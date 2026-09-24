import json
import subprocess
import sys
from dataclasses import dataclass

from touchstone import tasks
from touchstone.checks import Check


@dataclass
class _Ep:
    name: str
    id: str = "ep1"


@dataclass
class _Span:
    id: str


def _task(name="support-a1-01xyz", **kw):
    defaults = dict(
        episode_id="ep1",
        cut_span_id="01SPAN",
        tags=["resolved", "order_status"],
        context={"messages": [{"role": "user", "content": "whére is my ordér? café ☕"}],
                 "tools": [{"type": "function", "function": {"name": "order_status"}}]},
        reference={"content": "Looking it up.",
                   "tool_calls": [{"name": "order_status", "arguments": "{}"}]},
        checks=[Check(kind="tool_called", params={"name": "order_status"},
                      name="calls order_status", rule="must call order_status",
                      severity="hard", source="policy", because="every good run called it")],
    )
    defaults.update(kw)
    return tasks.Task(name=name, **defaults)


def test_task_name_is_deterministic_and_filesystem_safe():
    ep = _Ep(name="Support / A1094 #4")
    span = _Span(id="01M38QQRHS8S3KV4W0VKW82DBA")
    n1 = tasks.task_name(ep, span)
    n2 = tasks.task_name(ep, span)
    assert n1 == n2 == "support-a1094-4-01m38qqrhs8s3kv4w0vkw82dba"
    assert "/" not in n1 and " " not in n1


def test_write_creates_full_harbor_layout(tmp_path):
    d = tasks.write_task(tmp_path, _task())
    for rel in ("task.toml", "instruction.md", "context.json", "reference.json",
                "environment/Dockerfile", "solution/solve.sh", "tests/test.sh",
                "tests/verify.py", "tests/task.toml", "tests/reference.json",
                "tests/touchstone_checks/dsl.py", "tests/touchstone_checks/run.py"):
        assert (d / rel).exists(), rel
    doc = (d / "task.toml").read_text()
    assert 'name = "touchstone/support-a1-01xyz"' in doc
    assert "[[metadata.touchstone.check]]" in doc
    assert 'tool = "order_status"' in doc  # flat rename in the file


def test_write_read_roundtrip_unicode(tmp_path):
    task = _task(reference={"content": "café ☕ done", "tool_calls":
                            [{"name": "order_status", "arguments": "{}"}]})
    tasks.write_task(tmp_path, task)
    got = tasks.read_task(tmp_path / "tasks" / task.name)
    assert got.name == task.name
    assert got.episode_id == "ep1" and got.cut_span_id == "01SPAN"
    assert got.tags == ["resolved", "order_status"]
    assert got.context["messages"][0]["content"] == "whére is my ordér? café ☕"
    assert got.reference["content"] == "café ☕ done"
    assert len(got.checks) == 1
    c = got.checks[0]
    assert c.kind == "tool_called" and c.params == {"name": "order_status"}
    assert c.name == "calls order_status" and c.source == "policy"
    assert c.rule == "must call order_status" and c.because == "every good run called it"
    assert c.id == "calls order_status"  # name becomes the evaluator key


def test_list_and_get_and_active_filter(tmp_path):
    tasks.write_task(tmp_path, _task(name="a-01"))
    tasks.write_task(tmp_path, _task(name="b-02", tags=["failure"]))
    names = [t.name for t in tasks.list_tasks(tmp_path)]
    assert names == ["a-01", "b-02"]  # sorted
    assert [t.name for t in tasks.list_tasks(tmp_path, tag="failure")] == ["b-02"]
    assert tasks.get_task(tmp_path, "a-01").name == "a-01"
    assert tasks.get_task(tmp_path, "missing") is None


def test_gate_active_when_reference_passes_and_nop_fails(tmp_path):
    tasks.write_task(tmp_path, _task(name="ok-01"))
    assert tasks.read_task(tmp_path / "tasks" / "ok-01").status == "active"


def test_gate_needs_checks_when_empty_reply_passes(tmp_path):
    # only a safety check: an empty reply satisfies it, so the task measures nothing yet.
    safety = Check(kind="no_pii", params={"kinds": ["email"]}, name="no email", source="policy")
    tasks.write_task(tmp_path, _task(name="nop-01", checks=[safety]))
    got = tasks.read_task(tmp_path / "tasks" / "nop-01")
    assert got.status == "needs_checks" and "empty reply" in got.status_reason


def test_gate_needs_solution_when_reference_fails_its_own_check(tmp_path):
    leak = Check(kind="no_pii", params={"kinds": ["email"]}, name="no email", source="policy")
    ref = {"content": "email me at a@b.com", "tool_calls": []}
    hard = Check(kind="tool_called", params={"name": "order_status"}, name="calls it",
                 source="policy")
    tasks.write_task(tmp_path, _task(name="bad-ref", reference=ref, checks=[leak, hard]))
    got = tasks.read_task(tmp_path / "tasks" / "bad-ref")
    assert got.status == "needs_solution" and "recorded reply" in got.status_reason


def test_difficulty_cache_and_provenance_survive_rematerialisation(tmp_path):
    t = _task(name="diff-01")
    t.difficulty = {"openai:gpt-4o-mini": 0.5}
    t.parent_task = "diff-parent"
    t.generated_by = "claude-cli"
    tasks.write_task(tmp_path, t)
    # re-materialise from a fresh Task carrying no difficulty; the cache must survive.
    tasks.write_task(tmp_path, _task(name="diff-01"))
    got = tasks.read_task(tmp_path / "tasks" / "diff-01")
    assert got.difficulty == {"openai:gpt-4o-mini": 0.5}
    assert got.parent_task is None  # provenance is not preserved, only the measured cache
    # writing provenance persists it
    assert tasks.read_task(tmp_path / "tasks" / "diff-01").difficulty["openai:gpt-4o-mini"] == 0.5


def test_sync_preserves_interview_blocks_replaces_policy(tmp_path):
    interview = Check(kind="contains", params={"values": ["sorry"]}, name="apologises",
                      severity="soft", source="interview")
    policy_v1 = Check(kind="tool_called", params={"name": "order_status"}, name="v1",
                      source="policy")
    tasks.write_task(tmp_path, _task(name="room-01", checks=[interview, policy_v1]))
    # re-materialise with a different policy set; the interview block must survive.
    policy_v2 = Check(kind="tool_called", params={"name": "order_status"}, name="v2",
                      source="policy")
    tasks.write_task(tmp_path, _task(name="room-01", checks=[policy_v2]))
    got = tasks.read_task(tmp_path / "tasks" / "room-01")
    names = sorted(c.name for c in got.checks)
    assert "apologises" in names  # interview preserved
    assert "v2" in names and "v1" not in names  # policy replaced


def test_append_check(tmp_path):
    tasks.write_task(tmp_path, _task(name="app-01"))
    tasks.append_check(tmp_path, "app-01",
                       Check(kind="contains", params={"values": ["hi"]}, name="greets",
                             severity="soft", source="interview"))
    got = tasks.read_task(tmp_path / "tasks" / "app-01")
    assert "greets" in {c.name for c in got.checks}


def _run_verify(task_dir, output, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    reward_dir = tmp_path / "reward"
    out_path = tmp_path / "out.json"
    out_path.write_text(json.dumps(output))
    subprocess.run(
        [sys.executable, str(task_dir / "tests" / "verify.py")],
        check=False,
        env={"TOUCHSTONE_OUTPUT": str(out_path), "TOUCHSTONE_REWARD_DIR": str(reward_dir),
             "PATH": "/usr/bin:/bin"},
    )
    return (reward_dir / "reward.txt").read_text().strip()


def test_verify_subprocess_scores_reference_and_nop(tmp_path):
    d = tasks.write_task(tmp_path, _task(name="verify-01"))
    ref = json.loads((d / "reference.json").read_text())
    assert _run_verify(d, ref, tmp_path / "a") == "1.0"
    assert _run_verify(d, {"content": "", "tool_calls": []}, tmp_path / "b") == "0.0"


def test_instruction_renders_multimodal_parts_as_markers(tmp_path):
    task = _task(
        context={"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this"},
                {"type": "image", "image_url": "http://x/p.png"}]}],
                 "tools": []},
        reference={"content": "a cat", "tool_calls": []},
        checks=[Check(kind="contains", params={"values": ["cat"], "mode": "any"},
                      name="mentions cat", rule="says cat", severity="hard", source="policy",
                      because="ref does")],
    )
    d = tasks.write_task(tmp_path, task)
    instruction = (d / "instruction.md").read_text()
    assert "what is in this" in instruction and "[image]" in instruction
    # the text-based check still passes the reference gate through text_of
    assert tasks.read_task(d).status == "active"
