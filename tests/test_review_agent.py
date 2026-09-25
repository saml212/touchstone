"""The review agent's state machine: pick -> present -> agree/disagree -> change -> regrade."""

from __future__ import annotations

import json
from pathlib import Path

from touchstone import store
from touchstone.config import Settings
from touchstone.harbor import rewardkit
from touchstone.llm.base import Reply
from touchstone.review import trials
from touchstone.review.agent import ReviewAgent


class Seq:
    """A provider that returns a fixed sequence of replies, one per chat() call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0
        self.name = "seq"

    def chat(self, messages, tools=None, json=False, timeout=60):
        reply = self.replies[self.i]
        self.i += 1
        return reply


def _call(name, args):
    return Reply(tool_calls=[{"id": "1", "name": name, "arguments": json.dumps(args)}])


def _write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")


def _task(dataset: Path, name: str, *, reward_expected=183.18):
    d = dataset / "tasks" / name
    (d / "tests" / "correctness").mkdir(parents=True)
    _write(d / "instruction.md", "Refund my order B6991.")
    _write(d / "persona.md", "A customer wanting a refund.")
    _write(d / "task.toml",
           '[metadata.touchstone]\njob = "Issue a refund"\ntools = ["refund"]\n')
    rewardkit.write_criteria(d / "tests" / "correctness", "state",
                             [f"rk.sqlite_query_equals('s/state.db', 'SELECT refunded', "
                              f"{reward_expected})"])
    rewardkit.write_reward_toml(d / "tests", ["correctness", "safety"])


def _trial(job: Path, task: str, reward: float):
    d = job / f"{task}__x"
    _write(d / "result.json", {"task_name": task})
    _write(d / "verifier" / "reward.json", {"reward": reward, "correctness": reward})
    _write(d / "verifier" / "reward-details.json",
           {"correctness": {"components": [{"detail": {"criteria": [
               {"description": "refund recorded", "value": reward}]}}]}})
    _write(d / "agent" / "trajectory.json",
           {"steps": [{"source": "user", "message": "refund B6991"},
                      {"source": "agent", "message": "done",
                       "tool_calls": [{"function_name": "refund", "arguments": {"id": "B6991"}}]}]})


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "touchstone"
    _task(dataset, "issue-a-refund-1")
    _task(dataset, "issue-a-refund-2")
    _write(dataset / "groups.json",
           {"groups": [{"label": "Issue a refund", "slug": "issue-a-refund"}]})
    _write(dataset / "baseline.json",
           {"pass_rates": {"issue-a-refund-1": 0.75, "issue-a-refund-2": 1.0},
            "passed": ["issue-a-refund-2"]})
    job = dataset / "jobs" / "src"
    _write(job / "config.json", {})
    _trial(job, "issue-a-refund-1", 0.75)
    _trial(job, "issue-a-refund-2", 0.875)
    return dataset


def _settings(tmp_path: Path) -> Settings:
    return Settings(db_path=str(tmp_path / ".touchstone" / "touchstone.db"))


def _room(conn):
    return store.insert_room(conn, store.Room(task_id=None, topic="review"))


def test_open_statement_names_jobs_and_pass_count(tmp_path, conn):
    _dataset(tmp_path)
    agent = ReviewAgent(None, conn, _room(conn), _settings(tmp_path))
    opening = agent.open_statement()
    assert "issue a refund" in opening
    assert "2 tasks" in opening and "passes 1" in opening


def test_present_a_trial_sets_current(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    ref = trials.scan(tmp_path / "touchstone" / "jobs", conn)[0]  # unsure refund-1
    provider = Seq([
        _call("list_trials", {"filter": "unsure"}),
        _call("read_trial", {"task": ref.task, "trial": ref.trial}),
        Reply(content="You wanted a refund; the agent did it; the verifier was unsure. Agree?"),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    turn = agent.respond([{"role": "user", "speaker": "sam", "text": "start"}])
    assert "Agree" in turn.say
    fresh = ReviewAgent(None, conn, store.get_room(conn, room.id), _settings(tmp_path))
    state = fresh.review_state()
    assert state["current"]["task"] == ref.task
    assert state["counts"]["unsure"] == 2  # both refund trials scored between 0 and 1


def test_agree_records_review_and_moves_trust(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    ref = trials.scan(tmp_path / "touchstone" / "jobs", conn)[0]
    provider = Seq([
        _call("record_review", {"task": ref.task, "trial": ref.trial, "verdict": "agree"}),
        Reply(content="Recorded — you agreed."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    agent.respond([{"role": "user", "speaker": "sam", "text": "I agree"}])
    assert len(store.list_reviews(conn)) == 1
    assert agent.review_state()["trust"] == {"agreed": 1, "reviewed": 1, "score": 1.0}


def test_disagree_proposes_then_applies_and_regrades(tmp_path, conn):
    dataset = _dataset(tmp_path)
    room = _room(conn)
    # a regraded job where refund-1 now passes
    rg = dataset / "jobs" / "src-rg"
    _write(rg / "config.json", {})
    _trial(rg, "issue-a-refund-1", 1.0)
    _trial(rg, "issue-a-refund-2", 0.875)

    def fake_regrader(job_dir, tasks_path, *, settings):
        assert Path(job_dir).name == "src"
        return rg

    change = {"op": "edit", "file": "tests/correctness/state.py", "criterion": 1,
              "params": {"fn": "sqlite_query_equals",
                         "args": ["s/state.db", "SELECT refunded", 200.0]}}
    provider = Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "src/issue-a-refund-1__x"}),
        _call("propose_change", {"task": "issue-a-refund-1", "change": change}),
        Reply(content="I'll change the expected refund to 200 — say yes to apply."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path), regrader=fake_regrader)
    agent.respond([{"role": "user", "speaker": "sam", "text": "that's wrong"}])
    assert agent.review_state()["proposed"]["readback"]

    # confirm -> apply_change regrades and reports the move
    provider2 = Seq([
        _call("apply_change", {"task": "issue-a-refund-1", "change": change, "always": False}),
        Reply(content="Done — refund-1 now passes."),
    ])
    agent2 = ReviewAgent(provider2, conn, store.get_room(conn, room.id), _settings(tmp_path),
                         regrader=fake_regrader)
    agent2.respond([{"role": "user", "speaker": "sam", "text": "yes"}])
    # the file was actually rewritten
    state_py = (dataset / "tasks" / "issue-a-refund-1" / "tests" / "correctness" / "state.py")
    assert "200.0" in state_py.read_text()
    committed = agent2.committed()
    assert committed and committed[0]["deltas"][0]["task"] == "issue-a-refund-1"


def test_apply_always_hits_every_task_with_the_same_job(tmp_path, conn):
    dataset = _dataset(tmp_path)
    room = _room(conn)
    rg = dataset / "jobs" / "src-rg"
    _write(rg / "config.json", {})
    _trial(rg, "issue-a-refund-1", 1.0)

    change = {"op": "edit", "file": "tests/reward.toml", "criterion": "safety", "weight": 2}
    provider = Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "src/issue-a-refund-1__x"}),
        _call("apply_change", {"task": "issue-a-refund-1", "change": change, "always": True}),
        Reply(content="Applied to every refund task."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path),
                        regrader=lambda *a, **k: rg)
    agent.respond([{"role": "user", "speaker": "sam", "text": "always weight safety double"}])
    import tomllib
    for name in ("issue-a-refund-1", "issue-a-refund-2"):
        doc = tomllib.loads((dataset / "tasks" / name / "tests" / "reward.toml").read_text())
        assert doc["reward"][0]["weights"]["safety"] == 2.0


def test_presentation_is_grounded_in_the_actual_scores(tmp_path, conn):
    dataset = _dataset(tmp_path)
    # a trial the verifier scored 50% with a failed correctness check and an empty trajectory
    job = dataset / "jobs" / "weak"
    d = job / "issue-a-refund-1__z"
    _write(job / "config.json", {"agents": [{"name": "cust", "model_name": "gpt"}]})
    _write(d / "result.json", {"task_name": "issue-a-refund-1"})
    _write(d / "verifier" / "reward.json", {"reward": 0.5, "correctness": 0.0})
    _write(d / "verifier" / "reward-details.json",
           {"correctness": {"components": [{"detail": {"criteria": [
               {"description": "refund recorded", "value": 0.0}]}}]}})
    _write(d / "agent" / "trajectory.json", {"steps": []})  # agent did nothing
    room = _room(conn)
    provider = Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "weak/issue-a-refund-1__z"}),
        Reply(content="Do you agree this one passed?"),  # no numbers -> agent must ground it
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    say = agent.respond([{"role": "user", "speaker": "sam", "text": "next"}]).say
    assert "50%" in say                      # the reward is stated verbatim
    assert "failed" in say                   # the failing criterion is not glossed as a pass
    assert "did nothing" in say              # empty trajectory is called out


def test_unparseable_change_is_reported_not_crashed(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    bad = {"op": "edit", "file": "tests/reward.toml", "criterion": "nope", "weight": 1}
    provider = Seq([
        _call("apply_change", {"task": "issue-a-refund-1", "change": bad}),
        Reply(content="I couldn't apply that."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    turn = agent.respond([{"role": "user", "speaker": "sam", "text": "change it"}])
    assert "couldn't" in turn.say
    assert agent.committed() == []
