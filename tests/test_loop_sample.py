"""Sample: run the student, generate gated teacher variants, and report the frontier."""

import json
from pathlib import Path

from touchstone import store, tasks
from touchstone.bench import benchmark
from touchstone.llm import ScriptedProvider
from touchstone.loop import sample


def _bench_of_active(root):
    names = [t.name for t in tasks.list_tasks(root, active_only=True)]
    benchmark.create(root, "bench", task_names=names)
    return names


def test_sample_generates_gated_variants_and_frontier(project):
    conn, root = project(12)
    active = _bench_of_active(root)
    # scripted teacher returns no usable JSON -> a mechanical paraphrase variant per failing task.
    result = sample(conn, root, "bench", "scripted", teacher_provider=ScriptedProvider(),
                    variants=1)

    assert result["student"] == "scripted"
    assert result["variants_created"], "the scripted student should fail and spawn variants"
    # every created variant is active, carries provenance, and is a real task dir
    for name in result["variants_created"]:
        v = tasks.get_task(root, name)
        assert v.status == "active" and v.parent_task in active and v.kind == "variant"
    # the frontier is non-empty (the scripted student fails most active tasks)
    assert result["frontier"]
    assert set(result["frontier"]) >= set(result["variants_created"])


def test_generation_json_has_hashes_not_contents(project):
    conn, root = project(12)
    _bench_of_active(root)
    result = sample(conn, root, "bench", "scripted", teacher_provider=ScriptedProvider(),
                    variants=1)
    name = result["variants_created"][0]
    gen = json.loads((Path(root) / "tasks" / name / "generation.json").read_text())
    assert len(gen["parent_context_sha256"]) == 64  # a sha-256 hex digest
    assert gen["validation"]["oracle_pass"] and gen["validation"]["nop_fail"]
    assert gen["teacher"] and gen["method"] == "paraphrase"
    # provenance carries hashes, never the source context/reference contents
    blob = json.dumps(gen)
    assert "messages" not in blob and "reference" not in blob


def test_sample_is_deterministic_with_scripted(project):
    conn, root = project(12)
    _bench_of_active(root)
    a = sample(conn, root, "bench", "scripted", teacher_provider=ScriptedProvider(), variants=1)
    b = sample(conn, root, "bench", "scripted", teacher_provider=ScriptedProvider(), variants=1)
    assert a["variants_created"] == b["variants_created"]
    assert a["frontier"] == b["frontier"]


def test_sample_records_difficulty_and_loop_state(project):
    conn, root = project(12)
    _bench_of_active(root)
    sample(conn, root, "bench", "scripted", teacher_provider=ScriptedProvider(), variants=1)
    # difficulty rows exist for the student
    assert store.list_difficulty(conn, model_spec="scripted")
    from touchstone.loop.frontier import read_loop_state

    state = read_loop_state(root, "bench")
    assert state["student"] == "scripted" and "last_sample" in state


def test_variant_identical_to_parent_is_rejected_as_duplicate(tmp_path):
    # A teacher perturbation that doesn't actually change the task (a paraphrase that
    # echoes the same user turn, a rename of a tool that isn't present) yields a variant
    # identical to its parent; it must be rejected by content hash, not kept as a task.
    import importlib
    sample_mod = importlib.import_module("touchstone.loop.sample")
    from touchstone.checks import Check
    from touchstone.llm import Rule, ScriptedProvider
    root = str(tmp_path)
    tasks.write_task(root, tasks.Task(
        name="p", context={"messages": [{"role": "user", "content": "where is my order"}],
                            "tools": []},
        reference={"content": "your order id is 42", "tool_calls": []},
        checks=[Check(kind="contains", params={"values": ["order"], "mode": "any"},
                      name="says order", severity="hard", source="policy")]))
    assert tasks.get_task(root, "p").status == "active"
    teacher = ScriptedProvider(rules=[Rule(
        substring="where is my order",
        content=json.dumps([{"method": "paraphrase", "user": "where is my order"}]))])
    created = sample_mod._generate_variants(root, teacher, ["p"], "teacher:x", 1)
    assert created == []
    assert tasks.get_task(root, "p--v0") is None
