"""The review agent: a small state machine over a chat provider with a tool surface.

The room posts a participant message; `respond` runs the provider in a tool-calling loop (opening ->
pick a trial -> present it -> agree? -> on disagree draft + apply a criterion change -> regrade ->
next). Its five tools (list_trials, read_trial, record_review, propose_change, apply_change) are the
only way it touches the dataset, the reviews table, or Harbor. Trust and the current trial live in
the room state (a role="draft" scratch message + the reviews table) so the UI reads them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .. import store
from ..harbor import run as run_mod
from ..interview import rooms
from . import changes, opening, readback, regrade, replies, snapshot, trials
from .facts import _editable, _shared_tasks
from .prompt import SYSTEM, TOOLS
from .scratch import _Scratch

MAX_STEPS = 6

_log = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    """What the review agent says on one turn, plus any drafted / committed criterion changes."""

    say: str
    draft: list[dict] = field(default_factory=list)
    commit: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"say": self.say, "draft": self.draft, "commit": self.commit}



class ReviewAgent:
    def __init__(self, provider, conn, room: store.Room, settings, *,
                 regrader=run_mod.regrade, on_status=None) -> None:
        self.provider = provider
        self.conn = conn
        self.room = room
        self.settings = settings
        self.regrader = regrader
        self.on_status = on_status  # called with a state word while a regrade runs, then None
        self.dataset_dir = settings.review_dataset_dir
        self.jobs_dir = settings.review_jobs
        self.scratch = self._load_scratch()
        self._presented: dict | None = None  # the trial read this turn, for factual grounding

    # ---- opening -----------------------------------------------------------

    def open_statement(self) -> str:
        return opening.statement(self.dataset_dir, self.jobs_dir, self.room.topic,
                                 trials.counts(self._scan()))

    # ---- a turn ------------------------------------------------------------

    def respond(self, history: list[dict]) -> AgentTurn:
        if replies.wants_done(history):
            rooms.close(self.conn, self.room.id)
            return AgentTurn(say="Closing the room — thanks all.")
        if self.provider is None:
            return AgentTurn(say="Tell me when to start and I'll pull up the first trial.")
        self._scope_all = replies.wants_everywhere(history)
        self._wording = replies.wants_wording_change(history)
        say = self._ground(self._run_loop(history))
        self._save_scratch()
        return AgentTurn(say=say, draft=self.draft(), commit=self.committed())

    def _ground(self, say: str) -> str:
        """Guarantee the reply states the verifier's actual scores when a trial was just presented,
        so the narrative can't contradict the screen; if the model already stated it, trust it."""
        if not self._presented:
            return say
        token = replies.reward_pct(self._presented.get("reward"))
        if token in say:
            return say
        return f"{replies.grounding_line(self._presented)}\n\n{say}"

    def _run_loop(self, history: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM}, *replies.as_messages(history)]
        last_error: str | None = None
        for _ in range(MAX_STEPS):
            reply = self.provider.chat(messages, TOOLS)
            if not reply.tool_calls:
                # No content after a tool error must never be papered over — say what went wrong.
                return (reply.content or "").strip() or replies.tool_error_reply(last_error)
            messages.append({"role": "assistant", "content": reply.content,
                             "tool_calls": reply.tool_calls})
            applied, last_error = self._steps(messages, reply.tool_calls)
            if applied is not None:  # a successful apply ends the turn with a fixed read-out
                return replies.applied_reply(applied)
        return replies.tool_error_reply(last_error)  # hit MAX_STEPS -> surface the last error

    def _steps(self, messages: list[dict], calls: list[dict]) -> tuple[dict | None, str | None]:
        """Run the turn's tool calls, appending each result; stop at a successful apply.
        Returns (the apply result or None, the last tool error or None)."""
        last_error = None
        for call in calls:
            result = self._dispatch(call)
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "name": call.get("name"), "content": result})
            last_error = replies.error_of(result)
            applied = replies.applied_result(call, result)
            if applied is not None:
                return applied, last_error
        return None, last_error

    # ---- tool dispatch -----------------------------------------------------

    def _dispatch(self, call: dict) -> str:
        name = call.get("name") or ""
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        handler = {"list_trials": self._list_trials, "read_trial": self._read_trial,
                   "record_review": self._record_review, "propose_change": self._propose_change,
                   "apply_change": self._apply_change}.get(name)
        if handler is None:
            result = json.dumps({"error": f"unknown tool {name}"})
        else:
            try:
                result = json.dumps(handler(args), ensure_ascii=False, default=str)
            except (changes.ChangeError, OSError, ValueError, RuntimeError) as exc:
                result = json.dumps({"error": str(exc)})
        # Log every tool call (tasks/criteria, never secrets; truncated) to diagnose a failed apply.
        _log.info("review tool %s args=%s -> %s", name,
                  json.dumps(args, default=str)[:800], result[:800])
        return result

    def _scan(self):
        return trials.scan(self.jobs_dir, self.conn, self.dataset_dir)

    def _list_trials(self, args: dict) -> dict:
        which = args.get("filter")
        if which == "needs_review":
            return {"needs_review": trials.needs_review(self.dataset_dir)}
        refs = trials.filter_refs(self._scan(), which)
        return {"trials": [{"task": r.task, "trial": r.trial, "reward": r.reward,
                            "reward_pct": replies.reward_pct(r.reward), "category": r.category,
                            "reviewed": r.reviewed, "run": r.label} for r in refs[:20]]}

    def _read_trial(self, args: dict) -> dict:
        task, trial = args.get("task", ""), args.get("trial", "")
        detail = trials.read(self.dataset_dir, self.jobs_dir, task, trial)
        if detail is None:
            return self._read_gate_failure(task, trial)
        self.scratch.current = {"task": task, "trial": trial}
        task_dir = self.dataset_dir / "tasks" / task
        detail["editable"] = _editable(task_dir)
        detail["criteria_files"] = changes.existing_criteria_files(task_dir / "tests")
        detail["reward_pct"] = replies.reward_pct(detail.get("reward"))
        self._presented = detail  # ground this turn's reply in these scores
        return detail

    def _read_gate_failure(self, task: str, trial: str) -> dict:
        """A needs-review task has no trajectory: return why the gate set it aside (from its
        gate.json) so the reviewer says it, not the old "no trial" error the header pointed at."""
        for item in trials.needs_review(self.dataset_dir):
            if item.get("task") in (task, trial):
                self.scratch.current = {"task": item["task"], "trial": item["task"]}
                return {**item, "gate_failure": True, "reason": _gate_reason(item)}
        return {"error": f"no trial {trial} for {task}"}

    def _record_review(self, args: dict) -> dict:
        review = store.Review(task=args.get("task", ""), trial=args.get("trial", ""),
                              verdict=args.get("verdict", "agree"),
                              speaker=args.get("speaker", "agent"), note=args.get("note"))
        store.insert_review(self.conn, review)
        return {"recorded": review.verdict, "trust": self._trust()}

    def _task_arg(self, args: dict) -> str:
        """The task a tool acts on: the model's `task` when it names a real task dir, else the
        open trial's task. Models often send the job label ("refund-order") for the task name."""
        named = args.get("task") or ""
        if named and (self.dataset_dir / "tasks" / named / "task.toml").is_file():
            return named
        current = (self.scratch.current or {}).get("task", "")
        if not current:
            raise changes.ChangeError("no trial is open — read a trial first, then change it.")
        return current

    def _propose_change(self, args: dict) -> dict:
        task, change = self._task_arg(args), args.get("change")
        self._guard_instruction_change(change)
        changes.validate(self.dataset_dir / "tasks" / task, change)  # bad shape -> re-draft
        self.scratch.proposed = change
        self.scratch.readback = readback.describe(change)
        return {"readback": self.scratch.readback}

    def _guard_instruction_change(self, change) -> None:
        """A disagreement about a CHECK changes a criterion, never the instruction wording — an
        instruction edit moves no reward on the recorded trials, so it only reads as broken. The
        instruction is editable only when the person said the wording itself is wrong."""
        if getattr(self, "_wording", False):
            return
        if any(c.get("op") == "text" for c in changes.as_list(change)):
            raise changes.ChangeError(
                "this is a disagreement about a check — change a criterion (edit, add, remove, or "
                "a weight), not the instruction. Rewrite the instruction only if they say the "
                "wording itself is wrong.")

    def _apply_change(self, args: dict) -> dict:
        # Apply ONLY the proposal that was read back (self.scratch.proposed) — never a change the
        # model re-sends in args. A model that echoes a different `change` here (a wrong file or
        # index) would otherwise fail to apply after a valid read-back; the go-ahead applies what
        # the human just heard, nothing else.
        change = self.scratch.proposed
        if change is None:
            return {"error": "no change has been proposed yet — call propose_change and read it "
                             "back first, then apply."}
        task = self._task_arg(args)
        everywhere = bool(args.get("always")) and getattr(self, "_scope_all", False)
        targets = _shared_tasks(self.dataset_dir, task) if everywhere else [task]
        for name in targets:  # refuse an unvalidated/malformed change before writing any file
            changes.validate(self.dataset_dir / "tasks" / name, change)
        backups = {name: snapshot.snapshot_tests(self.dataset_dir / "tasks" / name)
                   for name in targets}
        for name in targets:
            changes.apply(self.dataset_dir / "tasks" / name, change)
        result = self._regrade_and_revert(targets, backups)
        self._note_applied(change, targets, result)
        return {"applied_to": targets, **result}

    def _regrade_and_revert(self, targets: list[str], backups: dict) -> dict:
        """Regrade after the change; put the files back when the regrade could not run at all (the
        host's build failed), or when it ran but broke the verifier on every task it touched."""
        try:
            result = self._regrade_current()
        except (RuntimeError, OSError) as exc:  # the regrade itself never ran — name why, roll back
            self._restore(targets, backups)
            return {"regrade_error": replies.first_line(str(exc)), "reverted": targets}
        if len(result.get("failed", [])) >= len(targets) > 0:  # broke the verifier everywhere
            self._restore(targets, backups)
            result["reverted"] = targets
        return result

    def _restore(self, targets: list[str], backups: dict) -> None:
        for name, files in backups.items():
            snapshot.restore_tests(self.dataset_dir / "tasks" / name, files)

    # ---- helpers -----------------------------------------------------------

    def _regrade_current(self) -> dict:
        current = self.scratch.current or {}
        job = (current.get("trial") or "").split("/", 1)[0]
        if not job:
            return {"deltas": [], "note": "no current trial to regrade"}
        host = self.settings.harbor_host or "local"
        self._emit_status(f"regrading on {host}")  # a regrade is a ~30s remote job, not "thinking"
        try:
            return regrade.regrade_job(self.dataset_dir, self.jobs_dir, job, self.settings,
                                       runner=self.regrader)
        finally:
            self._emit_status(None)

    def _emit_status(self, state: str | None) -> None:
        if self.on_status is not None:
            self.on_status(state)

    def _note_applied(self, change, targets: list[str], result: dict) -> None:
        self.scratch.applied = (self.scratch.applied or []) + [
            {"summary": readback.describe(change), "tasks": targets,
             "deltas": result.get("deltas", [])}]
        self.scratch.proposed = None
        self.scratch.readback = ""

    def _trust(self) -> dict:
        reviews = store.list_reviews(self.conn)
        seen: dict[tuple[str, str], str] = {}
        for r in reviews:
            seen[(r.task, r.trial)] = r.verdict
        reviewed = len(seen)
        agreed = sum(1 for v in seen.values() if v == "agree")
        score = agreed / reviewed if reviewed else None
        return {"agreed": agreed, "reviewed": reviewed, "score": score}

    # ---- scratch persistence (a role="draft" room message) -----------------

    def _load_scratch(self) -> _Scratch:
        for m in reversed(store.list_room_messages(self.conn, self.room.id)):
            if m.role == "draft":
                try:
                    return _Scratch(**json.loads(m.text))
                except (json.JSONDecodeError, TypeError):
                    return _Scratch()
        return _Scratch()

    def _save_scratch(self) -> None:
        rooms.post(self.conn, self.room.id, "agent", "draft",
                   json.dumps(self.scratch.to_dict(), ensure_ascii=False))

    # ---- room-state views --------------------------------------------------

    def draft(self) -> list:
        proposed = self.scratch.proposed
        if proposed is None:
            return []
        return proposed if isinstance(proposed, list) else [proposed]

    def committed(self) -> list:
        return self.scratch.applied or []

    def review_state(self) -> dict:
        refs = self._scan()
        counts = trials.counts(refs)
        counts["stale"] = trials.stale_count(self.jobs_dir, self.dataset_dir)
        return {"trust": self._trust(), "current": self._current_detail(),
                "counts": counts,
                "proposed": {"change": self.draft(), "readback": self.scratch.readback}
                if self.scratch.proposed is not None else None,
                "needs_review": len(trials.needs_review(self.dataset_dir))}

    def _current_detail(self) -> dict | None:
        current = self.scratch.current
        if not current:
            return None
        return trials.read(self.dataset_dir, self.jobs_dir, current["task"], current["trial"])


def _gate_reason(item: dict) -> str:
    """Why the gate set a needs-review task aside, in plain words the reviewer can say aloud."""
    reasons = {
        "oracle": (f"oracle scored {item.get('oracle')} — the recorded conversation does not pass "
                   "its own criteria"),
        "nop": f"an empty agent scored {item.get('nop')} — the task passes with no work",
    }
    return (reasons.get(item.get("failed_side")) or item.get("reason")
            or "the gate could not grade this task")
