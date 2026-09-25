"""The review agent: a small state machine over a chat provider with a tool surface.

The room posts a participant message; `respond` runs the provider in a tool-calling loop (opening ->
pick a trial -> present it -> agree? -> on disagree draft + apply a criterion change -> regrade ->
next). Its five tools (list_trials, read_trial, record_review, propose_change, apply_change) are the
only way it touches the dataset, the reviews table, or Harbor. Trust and the current trial live in
the room state (a role="draft" scratch message + the reviews table) so the UI reads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .. import store
from ..harbor import run as run_mod
from ..interview import rooms
from . import changes, regrade, replies, trials
from .facts import (
    _baseline_counts,
    _editable,
    _job_labels,
    _join,
    _shared_tasks,
    _task_count,
)
from .prompt import SYSTEM, TOOLS

MAX_STEPS = 6


@dataclass
class AgentTurn:
    """What the review agent says on one turn, plus any drafted / committed criterion changes."""

    say: str
    draft: list[dict] = field(default_factory=list)
    commit: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"say": self.say, "draft": self.draft, "commit": self.commit}


@dataclass
class _Scratch:
    current: dict | None = None   # {"task", "trial"}
    proposed: object = None       # a change (object or list) awaiting confirmation
    readback: str = ""
    applied: list = None          # summaries of applied changes this room

    def to_dict(self) -> dict:
        return {"current": self.current, "proposed": self.proposed,
                "readback": self.readback, "applied": self.applied or []}


class ReviewAgent:
    def __init__(self, provider, conn, room: store.Room, settings, *,
                 regrader=run_mod.regrade) -> None:
        self.provider = provider
        self.conn = conn
        self.room = room
        self.settings = settings
        self.regrader = regrader
        self.dataset_dir = settings.review_dataset_dir
        self.jobs_dir = settings.review_jobs
        self.scratch = self._load_scratch()
        self._presented: dict | None = None  # the trial read this turn, for factual grounding

    # ---- opening -----------------------------------------------------------

    def open_statement(self) -> str:
        jobs_done = _job_labels(self.dataset_dir)
        counts = _baseline_counts(self.dataset_dir)
        if not jobs_done and not counts:
            return (f"Let's review “{self.room.topic}”. I couldn't find a benchmark here "
                    "yet — run `touchstone survey` first, then reopen this room.")
        does = _join(jobs_done) or "several jobs"
        tail = (f"{counts['tasks']} tasks, your current setup passes {counts['passed']}."
                if counts else f"{_task_count(self.dataset_dir)} tasks.")
        return f"Your agent handles {does}; {tail} {self._offer()}"

    def _offer(self) -> str:
        """The opening's offer, chosen from what is actually there to review."""
        c = trials.counts(self._scan())
        if c["unsure"]:
            return "Want to walk through the trials the verifier was unsure about?"
        if c["disagree"]:
            return "Want to look at the tasks where the models disagree?"
        if c["unreviewed"]:
            return "Want to walk through the ones nobody has reviewed yet?"
        return "Everything passes — want to spot-check a few?"

    # ---- a turn ------------------------------------------------------------

    def respond(self, history: list[dict]) -> AgentTurn:
        if replies.wants_done(history):
            rooms.close(self.conn, self.room.id)
            return AgentTurn(say="Closing the room — thanks all.")
        if self.provider is None:
            return AgentTurn(say="Tell me when to start and I'll pull up the first trial.")
        say = self._ground(self._run_loop(history))
        self._save_scratch()
        return AgentTurn(say=say, draft=self.draft(), commit=self.committed())

    def _ground(self, say: str) -> str:
        """Guarantee the reply states the verifier's actual scores when a trial was just presented,
        so the narrative can never contradict the numbers on screen. If the model already stated the
        reward figure we trust its wording; otherwise we prepend the facts."""
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
                # No content after a tool error is the failure mode we must never paper over: say
                # what went wrong so the room isn't left with an empty "let me look at that".
                return (reply.content or "").strip() or replies.tool_error_reply(last_error)
            messages.append({"role": "assistant", "content": reply.content,
                             "tool_calls": reply.tool_calls})
            for call in reply.tool_calls:
                result = self._dispatch(call)
                last_error = replies.error_of(result)  # the latest tool call's error, or None
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "name": call.get("name"), "content": result})
        return replies.tool_error_reply(last_error)  # hit MAX_STEPS -> surface the last error

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
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(handler(args), ensure_ascii=False, default=str)
        except (changes.ChangeError, OSError, ValueError, RuntimeError) as exc:
            return json.dumps({"error": str(exc)})

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
            return {"error": f"no trial {trial} for {task}"}
        self.scratch.current = {"task": task, "trial": trial}
        detail["editable"] = _editable(self.dataset_dir / "tasks" / task)
        detail["reward_pct"] = replies.reward_pct(detail.get("reward"))
        self._presented = detail  # ground this turn's reply in these scores
        return detail

    def _record_review(self, args: dict) -> dict:
        review = store.Review(task=args.get("task", ""), trial=args.get("trial", ""),
                              verdict=args.get("verdict", "agree"),
                              speaker=args.get("speaker", "agent"), note=args.get("note"))
        store.insert_review(self.conn, review)
        return {"recorded": review.verdict, "trust": self._trust()}

    def _propose_change(self, args: dict) -> dict:
        task, change = args.get("task", ""), args.get("change")
        changes.validate(self.dataset_dir / "tasks" / task, change)  # bad shape -> re-draft
        self.scratch.proposed = change
        self.scratch.readback = changes.describe(change)
        return {"readback": self.scratch.readback}

    def _apply_change(self, args: dict) -> dict:
        task, change = args.get("task", ""), args.get("change")
        targets = _shared_tasks(self.dataset_dir, task) if args.get("always") else [task]
        for name in targets:  # refuse an unvalidated/malformed change before writing any file
            changes.validate(self.dataset_dir / "tasks" / name, change)
        for name in targets:
            changes.apply(self.dataset_dir / "tasks" / name, change)
        result = self._regrade_current()
        self._note_applied(change, targets, result)
        return {"applied_to": targets, **result}

    # ---- helpers -----------------------------------------------------------

    def _regrade_current(self) -> dict:
        current = self.scratch.current or {}
        job = (current.get("trial") or "").split("/", 1)[0]
        if not job:
            return {"deltas": [], "note": "no current trial to regrade"}
        return regrade.regrade_job(self.dataset_dir, self.jobs_dir, job, self.settings,
                                   runner=self.regrader)

    def _note_applied(self, change, targets: list[str], result: dict) -> None:
        self.scratch.applied = (self.scratch.applied or []) + [
            {"summary": changes.describe(change), "tasks": targets,
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
