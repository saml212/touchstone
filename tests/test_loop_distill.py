"""Distill: package exactly the student's frontier into training data with teacher demos."""

import json

from touchstone import tasks
from touchstone.bench import benchmark
from touchstone.llm import Rule, ScriptedProvider
from touchstone.loop import distill, sample


def _bench(root):
    names = [t.name for t in tasks.list_tasks(root, active_only=True)]
    benchmark.create(root, "bench", task_names=names)
    return names


def _teacher():
    return ScriptedProvider(rules=[Rule(substring="", content="All sorted — refund done.")])


def test_distill_restricts_dataset_to_the_frontier(project):
    conn, root = project(12)
    _bench(root)
    # Sample first so difficulty (and thus the frontier) exists for the student.
    sample(conn, root, "bench", "scripted", teacher_provider=ScriptedProvider(), variants=1)

    result = distill(conn, root, "bench", "scripted", teacher_provider=_teacher(), backend="null")
    front = set(result["frontier"])
    assert front, "the scripted student should have a non-empty frontier"

    # rl_tasks.jsonl is exactly the frontier tasks (dataset restricted to the frontier)
    rl_path = f"{result['out_dir']}/rl_tasks.jsonl"
    rl_tasks = {json.loads(line)["task"] for line in open(rl_path) if line.strip()}
    assert rl_tasks == front
    assert "Sample again" in result["plan"]
    assert result["status"] == "planned"  # the null backend wrote a plan


def test_distill_empty_frontier_is_a_clean_no_op(project):
    conn, root = project(12)
    _bench(root)
    # never sampled -> no difficulty rows -> empty frontier
    result = distill(conn, root, "bench", "openai:gpt-4o-mini", teacher_provider=_teacher())
    assert result["frontier"] == [] and result["status"] == "empty"
    assert "nothing to distill" in result["plan"]
