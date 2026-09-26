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
    # counted from the latest job (refund-1 0.75, refund-2 0.875) — both ran, neither passed
    assert "2 tasks" in opening and "passes 0 of the 2 run" in opening


def test_open_statement_flags_gated_tasks_the_baseline_never_ran(tmp_path, conn):
    """A baseline that covered only some of the gated tasks reads as "K of the R run (U not run)",
    the same shared counting the first-five sentence uses — never "passes 2 of 2"."""
    dataset = _dataset(tmp_path)
    # a third gated task exists, but the baseline (2 tasks) never ran it
    _task(dataset, "issue-a-refund-3")
    agent = ReviewAgent(None, conn, _room(conn), _settings(tmp_path))
    opening = agent.open_statement()
    assert "3 tasks" in opening and "passes 0 of the 2 run" in opening
    assert "1 not run yet" in opening


def test_open_statement_counts_run_tasks_from_the_latest_rewarded_job(tmp_path, conn):
    """The live tau-bench miscount: a job of six rewarded trials (none passing) read as "0 of the 0
    run (6 not run yet)" off an empty baseline. The opening must count run/passed from that job:
    every task ran, none passed -> "0 of the 6 run", nothing "not run"."""
    dataset = tmp_path / "touchstone"
    _write(dataset / "groups.json",
           {"groups": [{"label": "Exchange items", "slug": "exchange-items"}]})
    job = dataset / "jobs" / "2026-09-26__02-18-03"
    _write(job / "config.json", {})
    for i in range(1, 7):
        _task(dataset, f"exchange-items-{i}")
        _trial(job, f"exchange-items-{i}", 0.75)
    agent = ReviewAgent(None, conn, _room(conn), _settings(tmp_path))
    opening = agent.open_statement()
    assert "6 tasks" in opening and "passes 0 of the 6 run" in opening
    assert "not run" not in opening


def _errored_trial(job: Path, task: str, message: str):
    """A trial that raised inside Harbor: an exception_info, no verifier reward."""
    d = job / f"{task}__x"
    _write(d / "result.json", {"task_name": task,
                               "agent_info": {"name": "TouchstoneAgent",
                                              "model_info": {"name": "openai/gpt-4o-mini"}},
                               "exception_info": {"exception_type": "RuntimeError",
                                                  "exception_message": message}})


def test_open_statement_reports_an_errored_run_not_everything_passes(tmp_path, conn):
    """A bench run that errored on every task (Docker down) must not read as "everything passes":
    errored trials (an exception, no reward) count as errors in the opening."""
    dataset = tmp_path / "touchstone"
    _write(dataset / "groups.json",
           {"groups": [{"label": "Check order status", "slug": "check-order-status"}]})
    job = dataset / "jobs" / "run1"
    _write(job / "config.json", {})
    for i in range(1, 4):
        _errored_trial(job, f"check-order-status-{i}", "Docker daemon is not running.")
    agent = ReviewAgent(None, conn, _room(conn), _settings(tmp_path))
    opening = agent.open_statement()
    assert "everything passes" not in opening and "passes" not in opening
    assert "the last run errored on all 3 (Docker was not running)" in opening
    assert "touchstone bench" in opening


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


def test_regrade_emits_status_word_around_the_remote_job(tmp_path, conn):
    # While the ~30s regrade runs, the room shows "regrading on <host>", not a 30s "thinking".
    dataset = _dataset(tmp_path)
    room = _room(conn)
    rg = dataset / "jobs" / "src-rg"
    _write(rg / "config.json", {})
    _trial(rg, "issue-a-refund-1", 1.0)
    change = {"op": "edit", "file": "tests/correctness/state.py", "criterion": 1,
              "params": {"fn": "sqlite_query_equals",
                         "args": ["s/state.db", "SELECT refunded", 200.0]}}
    agent = ReviewAgent(Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "src/issue-a-refund-1__x"}),
        _call("propose_change", {"task": "issue-a-refund-1", "change": change}),
        Reply(content="say yes to apply."),
    ]), conn, room, _settings(tmp_path), regrader=lambda *a, **k: rg)
    agent.respond([{"role": "user", "speaker": "sam", "text": "that's wrong"}])

    seen: list = []
    agent2 = ReviewAgent(Seq([
        _call("apply_change", {"task": "issue-a-refund-1", "change": change, "always": False}),
        Reply(content="done."),
    ]), conn, store.get_room(conn, room.id), _settings(tmp_path),
        regrader=lambda *a, **k: rg, on_status=seen.append)
    agent2.respond([{"role": "user", "speaker": "sam", "text": "yes"}])
    assert seen == ["regrading on local", None]  # raised while regrading, then cleared


def test_apply_always_hits_every_task_with_the_same_job(tmp_path, conn):
    dataset = _dataset(tmp_path)
    room = _room(conn)
    rg = dataset / "jobs" / "src-rg"
    _write(rg / "config.json", {})
    _trial(rg, "issue-a-refund-1", 1.0)

    change = {"op": "edit", "file": "tests/reward.toml", "criterion": "safety", "weight": 2}
    provider = Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "src/issue-a-refund-1__x"}),
        _call("propose_change", {"task": "issue-a-refund-1", "change": change}),
        _call("apply_change", {"task": "issue-a-refund-1", "always": True}),
        Reply(content="Applied to every refund task."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path),
                        regrader=lambda *a, **k: rg)
    agent.respond([{"role": "user", "speaker": "sam", "text": "always weight safety double"}])
    import tomllib
    for name in ("issue-a-refund-1", "issue-a-refund-2"):
        doc = tomllib.loads((dataset / "tasks" / name / "tests" / "reward.toml").read_text())
        assert doc["reward"][0]["weights"]["safety"] == 2.0


def test_apply_change_names_the_cause_and_reverts_when_the_regrade_cannot_run(tmp_path, conn):
    """The regrade itself fails (the host's build failed) — the read-out must name the cause and
    say the files were put back, and the change must actually be rolled back on disk."""
    dataset = _dataset(tmp_path)
    room = _room(conn)
    reward = dataset / "tasks" / "issue-a-refund-1" / "tests" / "reward.toml"
    before = reward.read_text()

    def _boom(*_a, **_k):
        raise RuntimeError("docker compose build failed on host mini (RC=1)\nline two")

    change = {"op": "edit", "file": "tests/reward.toml", "criterion": "safety", "weight": 2}
    provider = Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "src/issue-a-refund-1__x"}),
        _call("propose_change", {"task": "issue-a-refund-1", "change": change}),
        _call("apply_change", {"task": "issue-a-refund-1"}),
        Reply(content="unused — the fixed read-out ends the turn"),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path), regrader=_boom)
    turn = agent.respond([{"role": "user", "speaker": "sam", "text": "double safety weight"}])
    assert "The regrade could not run: docker compose build failed on host mini" in turn.say
    assert "line two" not in turn.say  # only the first line
    assert "I put the files back as they were" in turn.say
    assert reward.read_text() == before  # the change was rolled back on disk


def test_apply_change_uses_the_proposed_change_not_a_model_supplied_one(tmp_path, conn):
    """The go-ahead applies the change that was read back, even if the model re-sends a different
    (wrong) change to apply_change — the classic 'file path doesn't exist' apply failure."""
    dataset = _dataset(tmp_path)
    room = _room(conn)
    rg = dataset / "jobs" / "src-rg"
    _write(rg / "config.json", {})
    _trial(rg, "issue-a-refund-1", 1.0)
    _trial(rg, "issue-a-refund-2", 0.875)

    proposed = {"op": "remove", "file": "tests/correctness/state.py", "criterion": 1}
    provider = Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "src/issue-a-refund-1__x"}),
        _call("propose_change", {"task": "issue-a-refund-1", "change": proposed}),
        Reply(content="I'll drop that check — say yes to apply."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path), regrader=lambda *a, **k: rg)
    agent.respond([{"role": "user", "speaker": "sam", "text": "that check is wrong"}])

    # The model now re-sends a DIFFERENT, non-existent change to apply_change; it must be ignored.
    wrong = {"op": "remove", "file": "tests/correctness/nope.py", "criterion": 3}
    provider2 = Seq([
        _call("apply_change", {"task": "issue-a-refund-1", "change": wrong, "always": False}),
        Reply(content="Done."),
    ])
    agent2 = ReviewAgent(provider2, conn, store.get_room(conn, room.id), _settings(tmp_path),
                         regrader=lambda *a, **k: rg)
    turn = agent2.respond([{"role": "user", "speaker": "sam", "text": "yes"}])
    # The proposed removal was applied (state.py now has no criteria), not the wrong change.
    state_py = dataset / "tasks" / "issue-a-refund-1" / "tests" / "correctness" / "state.py"
    assert "sqlite_query_equals" not in state_py.read_text()
    assert "Applied to issue-a-refund-1" in turn.say and "75% → 100%" in turn.say
    assert agent2.committed()


def test_apply_change_without_a_proposal_says_so(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    provider = Seq([
        _call("apply_change", {"task": "issue-a-refund-1"}),
        Reply(content="Nothing to apply."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path), regrader=lambda *a, **k: None)
    agent.respond([{"role": "user", "speaker": "sam", "text": "yes"}])
    assert agent.committed() == []


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


# ---- criterion-change robustness: validated shape, stored draft, surfaced errors ----------------

_GOOD_CHANGE = {"op": "edit", "file": "tests/correctness/state.py", "criterion": 1,
                "params": {"fn": "sqlite_query_equals",
                           "args": ["s/state.db", "SELECT refunded", 200.0]}}
_WRONG_SHAPE = {"action": "remove_criterion", "criterion": "total of 8 emails sent"}


def test_read_trial_exposes_a_stable_criterion_handle(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    provider = Seq([
        _call("read_trial", {"task": "issue-a-refund-1", "trial": "src/issue-a-refund-1__x"}),
        Reply(content="Here it is."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    agent.respond([{"role": "user", "speaker": "sam", "text": "show me one"}])
    editable = agent._presented["editable"]
    assert editable and editable[0]["handle"] == "tests/correctness/state.py:1"


def test_propose_rejects_wrong_shape_then_stores_the_right_one(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    provider = Seq([
        _call("propose_change", {"task": "issue-a-refund-1", "change": _WRONG_SHAPE}),  # error
        _call("propose_change", {"task": "issue-a-refund-1", "change": _GOOD_CHANGE}),  # stored
        Reply(content="I'll set the expected refund to 200 — say yes to apply."),
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    agent.respond([{"role": "user", "speaker": "sam", "text": "that's wrong"}])
    proposed = agent.review_state()["proposed"]["change"]
    assert proposed == [_GOOD_CHANGE]  # only the validated draft was stored, not the wrong shape


def test_apply_error_is_surfaced_not_filler(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    provider = Seq([
        _call("apply_change", {"task": "issue-a-refund-1", "change": _WRONG_SHAPE}),
        Reply(content=""),  # the model falls silent after the failed apply
    ])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    turn = agent.respond([{"role": "user", "speaker": "sam", "text": "yes apply it"}])
    assert "couldn't apply that change" in turn.say and "check by number" in turn.say
    assert turn.say != "Let me look at that."  # never the empty filler


def test_apply_max_steps_with_error_surfaces_the_reason(tmp_path, conn):
    _dataset(tmp_path)
    room = _room(conn)
    # every step keeps failing to apply the wrong shape -> MAX_STEPS -> still an explained reply
    provider = Seq([_call("apply_change", {"task": "issue-a-refund-1", "change": _WRONG_SHAPE})
                    for _ in range(10)])
    agent = ReviewAgent(provider, conn, room, _settings(tmp_path))
    turn = agent.respond([{"role": "user", "speaker": "sam", "text": "apply"}])
    assert "couldn't apply that change" in turn.say


def test_change_tools_fall_back_to_the_open_trial_task(tmp_path, monkeypatch):
    """A model that sends the job label as `task` still changes the trial it is looking at."""
    from touchstone.review import agent as review_agent

    ds = tmp_path / "touchstone"
    (ds / "tasks" / "refund-order-2" / "tests").mkdir(parents=True)
    (ds / "tasks" / "refund-order-2" / "task.toml").write_text("[metadata.touchstone]\n")
    ra = review_agent.ReviewAgent.__new__(review_agent.ReviewAgent)
    ra.dataset_dir = ds
    ra.scratch = review_agent._Scratch(current={"task": "refund-order-2", "trial": "j/t"})
    assert ra._task_arg({"task": "refund-order"}) == "refund-order-2"
    assert ra._task_arg({"task": "refund-order-2"}) == "refund-order-2"
    assert ra._task_arg({}) == "refund-order-2"


def test_applied_reply_reads_out_deltas_and_failures():
    from touchstone.review import replies

    text = replies.applied_reply({
        "applied_to": ["refund-order-2"], "job": "j2",
        "deltas": [{"task": "refund-order-2", "before": 1.0, "after": 0.875}],
        "failed": [{"task": "x", "error": "RegradeError: no artifact"}]})
    assert "Applied to refund-order-2." in text
    assert "refund-order-2 100% → 88%" in text or "refund-order-2 100% → 87%" in text
    assert "Could not regrade: x (RegradeError: no artifact)" in text


def test_apply_reverts_when_the_change_breaks_every_verifier(tmp_path):
    from touchstone.harbor import rewardkit
    from touchstone.review import agent as review_agent

    ds = tmp_path / "touchstone"
    task = ds / "tasks" / "t1"
    (task / "tests" / "correctness").mkdir(parents=True)
    (task / "task.toml").write_text("[metadata.touchstone]\n")
    rewardkit.write_criteria(task / "tests" / "correctness", "state", ["rk.file_exists('a')"])
    before = (task / "tests" / "correctness" / "state.py").read_text()
    ra = review_agent.ReviewAgent.__new__(review_agent.ReviewAgent)
    ra.dataset_dir = ds
    proposed = {"op": "remove", "file": "tests/correctness/state.py", "criterion": 1}
    ra.scratch = review_agent._Scratch(current={"task": "t1", "trial": "j/t"}, proposed=proposed)
    failed = [{"task": "t1", "error": "RewardFileNotFoundError: x"}]
    ra._regrade_current = lambda: {"job": "j2", "deltas": [], "failed": failed}
    out = ra._apply_change({"task": "t1"})
    assert out["reverted"] == ["t1"]
    assert (task / "tests" / "correctness" / "state.py").read_text() == before


def test_apply_scope_comes_from_the_persons_words(tmp_path):
    from touchstone.review import replies

    history = [{"role": "assistant", "text": "apply everywhere?"},
               {"role": "user", "speaker": "sam", "text": "Just this task."}]
    assert replies.wants_everywhere(history) is False
    history.append({"role": "user", "speaker": "sam", "text": "Actually, everywhere, always."})
    assert replies.wants_everywhere(history) is True
