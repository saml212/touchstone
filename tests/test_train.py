"""Training datasets + backend adapters."""

import ast
import json

import pytest

from touchstone import train
from touchstone.bench import benchmark, runner
from touchstone.llm import Reply
from touchstone.train import InfraRequired, TrainConfig, datasets, trainer_for


@pytest.fixture
def seeded(project):
    """A demo project (mined, enabled, synced) with a benchmark over every active task."""
    conn, root = project(14)
    benchmark.create(root, "demo", all_tasks=True)
    ntasks = len(benchmark.resolve(root, "demo"))
    return conn, root, ntasks


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_prepare_writes_all_files_with_manifest_counts(seeded, tmp_path):
    conn, root, ntasks = seeded
    out = tmp_path / "train"
    bundle = train.prepare(conn, root, "demo", out)

    for name in ("sft.jsonl", "preference.jsonl", "rl_tasks.jsonl", "manifest.json"):
        assert (out / name).exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["target"] == "demo"
    assert manifest["counts"] == bundle.counts
    assert bundle.counts["rl_tasks"] == ntasks
    assert bundle.counts["tasks"] == ntasks


def test_sft_holds_only_references_that_pass_hard_checks(seeded, tmp_path):
    conn, root, ntasks = seeded
    out = tmp_path / "train"
    train.prepare(conn, root, "demo", out)
    rows = _read_jsonl(out / "sft.jsonl")
    # active tasks all have a passing reference, so every one contributes an SFT row.
    assert rows
    for row in rows:
        assert set(row) == {"task", "messages", "tools", "completion"}
        assert row["completion"]["role"] == "assistant"
        assert "@example.com" not in (row["completion"]["content"] or "")


def test_rl_tasks_carry_context_tools_and_serialized_checks(seeded, tmp_path):
    conn, root, ntasks = seeded
    out = tmp_path / "train"
    train.prepare(conn, root, "demo", out)
    rows = _read_jsonl(out / "rl_tasks.jsonl")
    assert len(rows) == ntasks
    with_checks = [r for r in rows if r["checks"]]
    assert with_checks
    sample = with_checks[0]["checks"][0]
    assert {"name", "kind", "params", "applies_to", "severity"} <= set(sample)


class _FixedProvider:
    def __init__(self, content):
        self.content = content

    def chat(self, messages, tools=None, json=False, timeout=60):
        return Reply(content=self.content, usage={"tokens_in": 1, "tokens_out": 1})


def test_preference_pairs_from_candidate_disagreement_and_reference(conn, root, tmp_path):
    from touchstone import tasks
    from touchstone.checks import Check

    # A text-discriminating task: "all sorted" passes the contains check, "no" fails.
    tasks.write_task(root, tasks.Task(
        name="pref-01", context={"messages": [{"role": "user", "content": "help"}], "tools": []},
        reference={"content": "all sorted, thanks!", "tool_calls": []},
        checks=[Check(kind="contains", params={"values": ["sorted"]}, name="says sorted",
                      source="manual")]))
    benchmark.create(root, "demo", all_tasks=True)
    runner.run(conn, root, "demo", "good", provider=_FixedProvider("all sorted"))
    runner.run(conn, root, "demo", "bad", provider=_FixedProvider("no"))

    out = tmp_path / "train"
    train.prepare(conn, root, "demo", out)
    rows = _read_jsonl(out / "preference.jsonl")
    assert rows, "expected preference pairs"
    for row in rows:
        assert set(row) == {"task", "prompt", "tools", "chosen", "rejected", "source"}
        assert row["chosen"] != row["rejected"]
    sources = {r["source"] for r in rows}
    assert "candidates" in sources  # good vs bad on the same task
    assert "reference" in sources   # reference chosen over a failing candidate


def test_reruns_overwrite_cleanly(seeded, tmp_path):
    conn, root, _ = seeded
    out = tmp_path / "train"
    train.prepare(conn, root, "demo", out)

    runner.run(conn, root, "demo", "bad", provider=_FixedProvider("no"))
    train.prepare(conn, root, "demo", out)

    from touchstone import tasks
    one_name = tasks.list_tasks(root, active_only=True)[0].name
    benchmark.create(root, "one", task_names=[one_name])
    bundle = datasets.prepare(conn, root, "one", out)  # same out dir, fewer rows
    assert bundle.counts["rl_tasks"] == 1
    assert len(_read_jsonl(out / "rl_tasks.jsonl")) == 1  # overwritten, not appended
    assert json.loads((out / "manifest.json").read_text())["target"] == "one"


def test_null_backend_writes_a_plan(seeded, tmp_path):
    conn, root, _ = seeded
    trainer = trainer_for("null")
    bundle = trainer.prepare(conn, root, "demo", tmp_path / "train")
    handle = trainer.submit(bundle, TrainConfig())
    assert handle.status == "planned"
    plan = (tmp_path / "train" / "train_plan.md").read_text()
    assert "demo" in plan
    assert "sft.jsonl" in plan and "rl_tasks.jsonl" in plan


def test_art_backend_writes_runnable_script_then_raises(seeded, tmp_path):
    conn, root, _ = seeded
    trainer = trainer_for("art")
    bundle = trainer.prepare(conn, root, "demo", tmp_path / "train")
    with pytest.raises(InfraRequired) as exc:
        trainer.submit(bundle, TrainConfig(base_model="Qwen/Qwen2.5-7B-Instruct"))

    script = tmp_path / "train" / "art_train.py"
    assert script.exists()
    assert "art_train.py" in str(exc.value)
    ast.parse(script.read_text())
    assert (tmp_path / "train" / "touchstone_checks" / "run.py").exists()
    assert "Qwen/Qwen2.5-7B-Instruct" in script.read_text()


def test_trl_backend_writes_config_then_raises(seeded, tmp_path):
    conn, root, _ = seeded
    trainer = trainer_for("trl")
    bundle = trainer.prepare(conn, root, "demo", tmp_path / "train")
    with pytest.raises(InfraRequired):
        trainer.submit(bundle, TrainConfig())
    assert (tmp_path / "train" / "trl_sft.yaml").exists()
    run_sh = tmp_path / "train" / "run_trl.sh"
    assert run_sh.exists() and run_sh.stat().st_mode & 0o111


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        trainer_for("nope")


def test_prepare_unknown_target_raises(seeded, tmp_path):
    conn, root, _ = seeded
    with pytest.raises(ValueError, match="no active tasks"):
        train.prepare(conn, root, "missing", tmp_path / "train")
