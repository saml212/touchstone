"""Task directories are Harbor tasks: verify their layout, the vendored verifier, and harbor-run."""

import json
import os
import subprocess
import sys

import pytest

from touchstone import tasks
from touchstone.bench import harbor_run, harbor_tasks_path
from touchstone.checks import Check

# The files a task dir carries, matching harbor/src/harbor/cli/template-task/.
EXPECTED_FILES = ["task.toml", "instruction.md", ".gitignore",
                  "environment/Dockerfile", "tests/test.sh", "tests/verify.py",
                  "solution/solve.sh"]


@pytest.fixture
def task_dir(project):
    """One written active task dir from a demo project."""
    _conn, root = project(6)
    active = tasks.list_tasks(root, active_only=True)
    assert active
    return tasks.tasks_dir(root) / active[0].name


def test_layout_matches_harbor_template(task_dir):
    for rel in EXPECTED_FILES:
        assert (task_dir / rel).exists(), f"missing {rel}"
    assert (task_dir / "tests" / "touchstone_checks" / "dsl.py").exists()
    assert (task_dir / "tests" / "touchstone_checks" / "run.py").exists()


def test_task_toml_is_harbor_superset(task_dir):
    toml = (task_dir / "task.toml").read_text()
    assert 'schema_version = "1.4"' in toml
    for section in ("[task]", "[metadata.touchstone]", "[verifier]", "[agent]", "[environment]"):
        assert section in toml
    assert "[[metadata.touchstone.check]]" in toml
    assert 'name = "touchstone/' in toml


def test_judge_checks_excluded_from_container_copy(project):
    _conn, root = project(6)
    task = tasks.Task(name="judge-01",
                      reference={"content": "sorted", "tool_calls": []},
                      checks=[Check(kind="contains", params={"values": ["sorted"]}, name="c",
                                    source="manual"),
                              Check(kind="judge", params={"rubric": "?"}, name="j", severity="soft",
                                    source="manual")])
    d = tasks.write_task(root, task)
    root_names = {b["kind"] for b in _blocks(d / "task.toml")}
    tests_names = {b["kind"] for b in _blocks(d / "tests" / "task.toml")}
    assert "judge" in root_names  # authored source of truth keeps it
    assert "judge" not in tests_names  # the container verifier can't run judge (no LLM)


def _blocks(toml_path):
    import tomllib
    doc = tomllib.loads(toml_path.read_text())
    return doc.get("metadata", {}).get("touchstone", {}).get("check", [])


def _run_verifier(task_dir, output_obj, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "output.json"
    out_json.write_text(json.dumps(output_obj))
    reward_dir = tmp_path / "reward"
    env = {"TOUCHSTONE_OUTPUT": str(out_json), "TOUCHSTONE_REWARD_DIR": str(reward_dir),
           "PATH": os.environ.get("PATH", "")}
    proc = subprocess.run([sys.executable, str(task_dir / "tests" / "verify.py")],
                          capture_output=True, text=True, env=env)
    reward = (reward_dir / "reward.txt").read_text().strip()
    return proc, reward


def test_verify_runs_standalone_scoring_reference_and_nop(task_dir, tmp_path):
    reference = json.loads((task_dir / "reference.json").read_text())
    passing, reward_pass = _run_verifier(task_dir, reference, tmp_path / "a")
    assert passing.returncode == 0, passing.stderr
    assert reward_pass == "1.0"

    empty = {"content": "", "tool_calls": []}
    failing, reward_fail = _run_verifier(task_dir, empty, tmp_path / "b")
    assert failing.returncode == 1
    assert reward_fail == "0.0"


def test_verify_survives_missing_and_nondict_output(task_dir, tmp_path):
    reward_dir = tmp_path / "r"

    def run(output_file):
        env = {"TOUCHSTONE_OUTPUT": str(output_file), "TOUCHSTONE_REWARD_DIR": str(reward_dir),
               "PATH": os.environ.get("PATH", "")}
        return subprocess.run([sys.executable, str(task_dir / "tests" / "verify.py")],
                              capture_output=True, text=True, env=env)

    missing = run(tmp_path / "nope.json")
    assert missing.returncode in (0, 1) and (reward_dir / "reward.txt").exists()

    nondict = tmp_path / "arr.json"
    nondict.write_text("[1, 2, 3]")
    arr = run(nondict)
    assert arr.returncode in (0, 1) and (reward_dir / "reward.txt").exists()


def test_harbor_tasks_path_points_at_tasks_dir(root):
    assert harbor_tasks_path(root).name == "tasks"


def test_harbor_run_prints_command_when_docker_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(harbor_run.shutil, "which", lambda name: None)
    lines: list[str] = []
    code = harbor_run.run_task(tmp_path, agent="claude", echo=lines.append)
    assert code == 1
    joined = "\n".join(lines)
    assert "harbor run -p" in joined and str(tmp_path) in joined
    assert "-a claude" in joined
    assert any("docker" in line for line in lines)
