"""The review agent: a small state machine over a chat provider with a tool surface.

The room posts a participant message; `respond` runs the provider in a tool-calling loop
(opening -> pick a trial -> present it -> ask agree? -> on disagree draft + apply a criterion change
-> regrade -> next) and returns what to say. The five tools are the only way it touches anything:

- ``list_trials(filter)``    which trials to walk, in priority order
- ``read_trial(task, trial)``  the instruction, trajectory, and criteria+scores (sets current)
- ``record_review(task, trial, verdict, note)``  an agree/disagree row; trust is the agreed share
- ``propose_change(task, change)``  read back a criterion change without writing it
- ``apply_change(task, change, always)``  write the file(s), regrade, and report what moved

Trust and the current trial live in the room state (a role="draft" scratch message + the reviews
table) so the UI reads them. The prompts are short and the agent speaks plain product language; it
never names files unless asked.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .. import store
from ..harbor import run as run_mod
from ..interview import rooms
from . import changes, regrade, trials

MAX_STEPS = 6


@dataclass
class AgentTurn:
    """What the review agent says on one turn, plus any drafted / committed criterion changes."""

    say: str
    draft: list[dict] = field(default_factory=list)
    commit: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"say": self.say, "draft": self.draft, "commit": self.commit}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_trials",
        "description": "List trials to review in priority order. filter: unsure|disagree|"
                       "unreviewed|needs_review|all.",
        "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "read_trial",
        "description": "Read one trial: instruction, the trajectory in plain words, and each "
                       "criterion's description and score. Sets it as the current trial.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}, "trial": {"type": "string"}},
                       "required": ["task", "trial"]}}},
    {"type": "function", "function": {
        "name": "record_review",
        "description": "Record whether the human agreed with the verifier on this trial.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"}, "trial": {"type": "string"},
            "verdict": {"type": "string", "description": "agree or disagree"},
            "note": {"type": "string"}}, "required": ["task", "trial", "verdict"]}}},
    {"type": "function", "function": {
        "name": "propose_change",
        "description": "Read back a criterion change (edit/add/remove a check, a weight, a judge "
                       "line, or instruction/persona wording) without writing it. `change` is one "
                       "object or a list of them.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}, "change": {}},
                       "required": ["task", "change"]}}},
    {"type": "function", "function": {
        "name": "apply_change",
        "description": "Write the change, regrade the trial's job, and report the new reward and "
                       "any other trials that moved. always=true applies it to every task with the "
                       "same job.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"}, "change": {},
            "always": {"type": "boolean"}}, "required": ["task", "change"]}}},
]

_SYSTEM = (
    "You are Touchstone's review agent. A product person is checking their AI agent's benchmark "
    "with you, by voice or text. Walk one trial at a time: call list_trials, then read_trial. When "
    "you present a trial, FIRST state the facts verbatim from read_trial: the verifier reward as a "
    "percentage, then each criterion with whether it passed or failed. ONLY THEN gloss what "
    "the user wanted and what the agent did in plain words, and ask whether they agree it passed. "
    "Never claim a pass or a fail the scores do not show; when the trajectory is empty, say the "
    "agent did nothing. On agree, call record_review (verdict 'agree'); on disagree, record_review "
    "(verdict 'disagree'), ask what should have counted, call propose_change and read it back, and "
    "only after they confirm call apply_change (always=true if the rule holds for every task "
    "like it); then say the new reward and anything else that moved. Speak in plain product "
    "language. Never mention file names, tables, or JSON unless they ask. Reply with your spoken "
    "message when you are not calling a tool."
)


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
        if _wants_done(history):
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
        token = _reward_pct(self._presented.get("reward"))
        if token in say:
            return say
        return f"{_grounding_line(self._presented)}\n\n{say}"

    def _run_loop(self, history: list[dict]) -> str:
        messages = [{"role": "system", "content": _SYSTEM}, *_as_messages(history)]
        say = "Let me look at that."
        for _ in range(MAX_STEPS):
            reply = self.provider.chat(messages, TOOLS)
            if not reply.tool_calls:
                return reply.content or say
            messages.append({"role": "assistant", "content": reply.content,
                             "tool_calls": reply.tool_calls})
            say = reply.content or say
            for call in reply.tool_calls:
                result = self._dispatch(call)
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "name": call.get("name"), "content": result})
        return say

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
                            "reward_pct": _reward_pct(r.reward), "category": r.category,
                            "reviewed": r.reviewed, "run": r.label} for r in refs[:20]]}

    def _read_trial(self, args: dict) -> dict:
        task, trial = args.get("task", ""), args.get("trial", "")
        detail = trials.read(self.dataset_dir, self.jobs_dir, task, trial)
        if detail is None:
            return {"error": f"no trial {trial} for {task}"}
        self.scratch.current = {"task": task, "trial": trial}
        detail["editable"] = _editable(self.dataset_dir / "tasks" / task)
        detail["reward_pct"] = _reward_pct(detail.get("reward"))
        self._presented = detail  # ground this turn's reply in these scores
        return detail

    def _record_review(self, args: dict) -> dict:
        review = store.Review(task=args.get("task", ""), trial=args.get("trial", ""),
                              verdict=args.get("verdict", "agree"),
                              speaker=args.get("speaker", "agent"), note=args.get("note"))
        store.insert_review(self.conn, review)
        return {"recorded": review.verdict, "trust": self._trust()}

    def _propose_change(self, args: dict) -> dict:
        change = args.get("change")
        readback = changes.describe(change)
        self.scratch.proposed = change
        self.scratch.readback = readback
        return {"readback": readback}

    def _apply_change(self, args: dict) -> dict:
        task, change = args.get("task", ""), args.get("change")
        targets = _shared_tasks(self.dataset_dir, task) if args.get("always") else [task]
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


# ---- module helpers --------------------------------------------------------


def _reward_pct(reward) -> str:
    return "unknown" if reward is None else f"{round(reward * 100)}%"


def _passed(score) -> bool:
    return score is True or score == 1


def _grounding_line(detail: dict) -> str:
    """The verifier's actual result in one line: reward %, each criterion pass/fail, empty note."""
    parts = [f"The verifier scored this {_reward_pct(detail.get('reward'))}."]
    crits = detail.get("criteria") or []
    if crits:
        marks = "; ".join(f"{c.get('description', '?')} — "
                          f"{'passed' if _passed(c.get('score')) else 'failed'}" for c in crits)
        parts.append(f"Checks: {marks}.")
    if not detail.get("trajectory"):
        parts.append("The agent did nothing that was recorded.")
    return " ".join(parts)


def _wants_done(history: list[dict]) -> bool:
    """The participants asked to close the room since the agent last spoke (`/done`)."""
    idx = max((i for i, m in enumerate(history) if m.get("role") == "assistant"), default=-1)
    return any(m.get("role") == "user" and m.get("text", "").strip().lower().startswith("/done")
               for m in history[idx + 1:])


def _as_messages(history: list[dict]) -> list[dict]:
    """Room messages -> chat messages: participants are users, the agent is the assistant."""
    out = []
    for m in history:
        role = "assistant" if m.get("role") == "assistant" else "user"
        who = m.get("speaker", "")
        text = m.get("text", "")
        out.append({"role": role, "content": f"{who}: {text}" if role == "user" else text})
    return out


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _job_labels(dataset_dir: Path) -> list[str]:
    groups = dataset_dir / "groups.json"
    if groups.is_file():
        try:
            data = json.loads(groups.read_text(encoding="utf-8"))
            return [g["label"] for g in data.get("groups", []) if g.get("label")]
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return _labels_from_tasks(dataset_dir)


def _labels_from_tasks(dataset_dir: Path) -> list[str]:
    labels: list[str] = []
    for task_dir in sorted((dataset_dir / "tasks").glob("*")):
        job = _touchstone_meta(task_dir).get("job")
        if job and job not in labels:
            labels.append(job)
    return labels


def _touchstone_meta(task_dir: Path) -> dict:
    return _read_toml(task_dir / "task.toml").get("metadata", {}).get("touchstone", {})


def _baseline_counts(dataset_dir: Path) -> dict | None:
    baseline = dataset_dir / "baseline.json"
    if not baseline.is_file():
        return None
    try:
        data = json.loads(baseline.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    rates = data.get("pass_rates") or {}
    return {"tasks": len(rates), "passed": len(data.get("passed", []))}


def _task_count(dataset_dir: Path) -> int:
    tasks = dataset_dir / "tasks"
    return sum(1 for d in tasks.glob("*") if (d / "task.toml").is_file()) if tasks.is_dir() else 0


def _join(items: list[str]) -> str:
    items = [i.lower() for i in items]
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _shared_tasks(dataset_dir: Path, task: str) -> list[str]:
    """Every task with the same job-to-be-done label as `task` (for an 'always' change)."""
    job = _touchstone_meta(dataset_dir / "tasks" / task).get("job")
    if not job:
        return [task]
    out = [d.name for d in sorted((dataset_dir / "tasks").glob("*"))
           if _touchstone_meta(d).get("job") == job]
    return out or [task]


def _editable(task_dir: Path) -> list[dict]:
    """The criteria the room can change, keyed by file + 1-based index (or reward dimension)."""
    out: list[dict] = []
    tests = task_dir / "tests"
    for py in sorted(tests.rglob("*.py")):
        try:
            calls = changes.parse_criteria(py)
        except changes.ChangeError:
            continue
        rel = py.relative_to(task_dir).as_posix()
        out += [{"file": rel, "index": i, "criterion": src} for i, src in enumerate(calls, 1)]
    reward = tests / "reward.toml"
    if reward.is_file():
        weights = _read_toml(reward).get("reward", [{}])[0].get("weights", {})
        out += [{"file": "tests/reward.toml", "dimension": dim, "weight": w}
                for dim, w in weights.items()]
    return out
