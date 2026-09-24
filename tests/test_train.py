"""Training datasets + backend adapters."""

import ast
import json

import pytest

from touchstone import store, train
from touchstone.bench import benchmark, runner
from touchstone.llm import Reply
from touchstone.mine import cut_tasks
from touchstone.train import InfraRequired, TrainConfig, datasets, trainer_for


@pytest.fixture
def seeded(demo_db):
    """A demo db with mined tasks, a positive check, a safety check, and one benchmark."""
    conn = demo_db(14)
    store.insert_check(conn, store.Check(
        name="polite", kind="contains",
        params={"values": ["sorted", "escalat"], "mode": "any"}, enabled=1))
    store.insert_check(conn, store.Check(name="clean", kind="no_pii", params={}, enabled=1))
    cut_tasks(conn, store.list_episodes(conn))
    bench = benchmark.create(conn, "demo", all_tasks=True)
    return conn, bench


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_prepare_writes_all_files_with_manifest_counts(seeded, tmp_path):
    conn, bench = seeded
    out = tmp_path / "train"
    bundle = train.prepare(conn, bench.id, out)

    for name in ("sft.jsonl", "preference.jsonl", "rl_tasks.jsonl", "manifest.json"):
        assert (out / name).exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["benchmark_id"] == bench.id
    assert manifest["counts"] == bundle.counts
    assert bundle.counts["rl_tasks"] == len(bench.task_ids)
    assert bundle.counts["tasks"] == len(bench.task_ids)


def test_sft_holds_only_references_that_pass_hard_checks(seeded, tmp_path):
    conn, bench = seeded
    out = tmp_path / "train"
    train.prepare(conn, bench.id, out)
    rows = _read_jsonl(out / "sft.jsonl")

    # A resolved episode that leaks PII fails the no_pii safety check, so its reference is excluded.
    assert rows and len(rows) < len(bench.task_ids)
    for row in rows:
        assert set(row) == {"task_id", "messages", "tools", "completion"}
        assert row["completion"]["role"] == "assistant"
        # no SFT completion may leak an email (the no_pii check is enabled)
        assert "@example.com" not in (row["completion"]["content"] or "")


def test_rl_tasks_carry_context_tools_and_serialized_checks(seeded, tmp_path):
    conn, bench = seeded
    out = tmp_path / "train"
    train.prepare(conn, bench.id, out)
    rows = _read_jsonl(out / "rl_tasks.jsonl")
    assert len(rows) == len(bench.task_ids)
    with_checks = [r for r in rows if r["checks"]]
    assert with_checks, "some tasks should carry checks"
    sample = with_checks[0]["checks"][0]
    assert {"id", "name", "kind", "params", "applies_to", "severity"} <= set(sample)


class _FixedProvider:
    """Returns a constant reply, so two runs can disagree on the same task."""

    def __init__(self, content):
        self.content = content

    def chat(self, messages, tools=None, json=False, timeout=60):
        return Reply(content=self.content, usage={"tokens_in": 1, "tokens_out": 1})


def test_preference_pairs_from_candidate_disagreement_and_reference(seeded, tmp_path):
    conn, bench = seeded
    # One candidate always says "sorted" (passes the contains check); one always says "" (fails).
    runner.run(conn, bench.id, "good", provider=_FixedProvider("all sorted"))
    runner.run(conn, bench.id, "bad", provider=_FixedProvider("no"))

    out = tmp_path / "train"
    train.prepare(conn, bench.id, out)
    rows = _read_jsonl(out / "preference.jsonl")
    assert rows, "expected preference pairs"
    for row in rows:
        assert set(row) == {"task_id", "prompt", "tools", "chosen", "rejected", "source"}
        assert row["chosen"] != row["rejected"]
    sources = {r["source"] for r in rows}
    assert "candidates" in sources  # good vs bad on the same task
    assert "reference" in sources   # reference chosen over a failing candidate


def test_preference_finds_runs_recorded_by_benchmark_name(seeded, tmp_path):
    """Runs store the benchmark key as passed to `bench run` (the CLI passes the name); preference
    gathering must still find them, so a name-keyed run yields reference-vs-failed pairs."""
    conn, bench = seeded
    runner.run(conn, bench.name, "bad", provider=_FixedProvider("no"))  # keyed by name, not id

    out = tmp_path / "train"
    train.prepare(conn, bench.id, out)
    rows = _read_jsonl(out / "preference.jsonl")
    assert rows and any(r["source"] == "reference" for r in rows)


def test_reruns_overwrite_cleanly(seeded, tmp_path):
    conn, bench = seeded
    out = tmp_path / "train"
    train.prepare(conn, bench.id, out)

    # Add a run so preference rows appear, then narrow to a single task and re-prepare.
    runner.run(conn, bench.id, "bad", provider=_FixedProvider("no"))
    train.prepare(conn, bench.id, out)  # preference now non-empty

    one = store.insert_benchmark(conn, store.Benchmark(name="one", task_ids=[bench.task_ids[0]]))
    bundle = datasets.prepare(conn, one.id, out)  # same out dir, fewer rows
    assert bundle.counts["rl_tasks"] == 1
    assert len(_read_jsonl(out / "rl_tasks.jsonl")) == 1  # overwritten, not appended
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["benchmark_id"] == one.id


def test_null_backend_writes_a_plan(seeded, tmp_path):
    conn, bench = seeded
    trainer = trainer_for("null")
    bundle = trainer.prepare(conn, bench.id, tmp_path / "train")
    handle = trainer.submit(bundle, TrainConfig())
    assert handle.status == "planned"
    plan = (tmp_path / "train" / "train_plan.md").read_text()
    assert bench.name in plan
    assert "sft.jsonl" in plan and "rl_tasks.jsonl" in plan


def test_art_backend_writes_runnable_script_then_raises(seeded, tmp_path):
    conn, bench = seeded
    trainer = trainer_for("art")
    bundle = trainer.prepare(conn, bench.id, tmp_path / "train")
    with pytest.raises(InfraRequired) as exc:
        trainer.submit(bundle, TrainConfig(base_model="Qwen/Qwen2.5-7B-Instruct"))

    script = tmp_path / "train" / "art_train.py"
    assert script.exists()
    assert "art_train.py" in str(exc.value)
    ast.parse(script.read_text())  # the generated script is valid Python
    assert (tmp_path / "train" / "touchstone_checks" / "run.py").exists()
    assert "Qwen/Qwen2.5-7B-Instruct" in script.read_text()


def test_trl_backend_writes_config_then_raises(seeded, tmp_path):
    conn, bench = seeded
    trainer = trainer_for("trl")
    bundle = trainer.prepare(conn, bench.id, tmp_path / "train")
    with pytest.raises(InfraRequired):
        trainer.submit(bundle, TrainConfig())
    assert (tmp_path / "train" / "trl_sft.yaml").exists()
    run_sh = tmp_path / "train" / "run_trl.sh"
    assert run_sh.exists() and run_sh.stat().st_mode & 0o111  # executable


def test_unknown_backend_raises(seeded):
    with pytest.raises(ValueError, match="unknown backend"):
        trainer_for("nope")


def test_prepare_unknown_benchmark_raises(seeded, tmp_path):
    conn, _bench = seeded
    with pytest.raises(ValueError, match="no benchmark"):
        train.prepare(conn, "missing", tmp_path / "train")
