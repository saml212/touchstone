import json
import subprocess
import sys

import pytest

from touchstone import store
from touchstone.bench import benchmark, harbor_export, harbor_run
from touchstone.demo import run_demo
from touchstone.mine import cut_tasks

# The files this export matches from harbor/src/harbor/cli/template-task/.
EXPECTED_FILES = ["task.toml", "instruction.md", ".gitignore",
                  "environment/Dockerfile", "tests/test.sh", "tests/test_outputs.py",
                  "solution/solve.sh"]


@pytest.fixture
def exported(traced, tmp_path):
    run_demo(n=6)
    conn = store.connect(traced)
    store.insert_check(conn, store.Check(
        name="polite", kind="contains",
        params={"values": ["sorted", "escalat"], "mode": "any"}, enabled=1))
    store.insert_check(conn, store.Check(name="clean", kind="no_pii", params={}, enabled=1))
    store.insert_check(conn, store.Check(
        name="judgey", kind="judge", params={"rubric": "nice?"}, severity="soft", enabled=1))
    cut_tasks(conn, store.list_episodes(conn))
    for t in store.list_tasks(conn):  # attach the judge check so we can prove it's dropped
        ids = list(t.check_ids or [])
        jid = next(c.id for c in store.list_checks(conn) if c.kind == "judge")
        if jid not in ids:
            ids.append(jid)
        store.update_task(conn, t.id, check_ids=ids)
    bench = benchmark.create(conn, "all", all_tasks=True)
    out = tmp_path / "harbor"
    dirs = harbor_export.export(conn, bench.id, out)
    yield conn, bench, out, dirs
    conn.close()


def test_layout_matches_harbor_template(exported):
    _conn, bench, _out, dirs = exported
    assert len(dirs) == len(bench.task_ids)
    for d in dirs:
        for rel in EXPECTED_FILES:
            assert (d / rel).exists(), f"missing {rel} in {d.name}"
        assert (d / "tests" / "touchstone_checks" / "dsl.py").exists()
        assert (d / "tests" / "touchstone_checks" / "run.py").exists()


def test_task_toml_has_template_fields(exported):
    _conn, _bench, _out, dirs = exported
    toml = (dirs[0] / "task.toml").read_text()
    assert 'schema_version = "1.4"' in toml
    for section in ("[metadata]", "[verifier]", "[agent]", "[environment]"):
        assert section in toml
    assert "build_timeout_sec" in toml


def test_judge_checks_dropped_from_export(exported):
    _conn, _bench, _out, dirs = exported
    for d in dirs:
        spec = json.loads((d / "tests" / "checks.json").read_text())
        assert all(c["kind"] != "judge" for c in spec["checks"])


def test_reexport_overwrites_cleanly(exported):
    conn, bench, out, dirs = exported
    stale = dirs[0] / "STALE.txt"
    stale.write_text("x")
    harbor_export.export(conn, bench.id, out)
    assert not stale.exists()  # the task dir was rebuilt from scratch


def _run_verifier(task_dir, output_obj, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "output.json"
    out_json.write_text(json.dumps(output_obj))
    reward_dir = tmp_path / "reward"
    env = {
        "TOUCHSTONE_OUTPUT": str(out_json),
        "TOUCHSTONE_REWARD_DIR": str(reward_dir),
        "TOUCHSTONE_CHECKS": str(task_dir / "tests" / "checks.json"),
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    proc = subprocess.run(
        [sys.executable, str(task_dir / "tests" / "test_outputs.py")],
        capture_output=True, text=True, env=env,
    )
    reward = (reward_dir / "reward.txt").read_text().strip()
    return proc, reward


def test_test_outputs_runs_standalone_and_writes_reward(exported, tmp_path):
    _conn, _bench, _out, dirs = exported
    task_dir = next(d for d in dirs if "contains" in (d / "tests" / "checks.json").read_text())

    passing, reward_pass = _run_verifier(
        task_dir, {"content": "All sorted, thanks!", "tool_calls": []}, tmp_path / "a")
    assert passing.returncode == 0, passing.stderr
    assert reward_pass == "1.0"

    failing, reward_fail = _run_verifier(
        task_dir, {"content": "nope", "tool_calls": []}, tmp_path / "b")
    assert failing.returncode == 1
    assert reward_fail == "0.0"


def test_harbor_run_prints_command_when_docker_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(harbor_run.shutil, "which", lambda name: None)
    lines: list[str] = []
    code = harbor_run.run_task(tmp_path, agent="claude", echo=lines.append)
    assert code == 1
    joined = "\n".join(lines)
    assert "harbor run -p" in joined and str(tmp_path) in joined
    assert "-a claude" in joined
    assert any("docker" in line for line in lines)


def test_reexport_removes_stale_task_dirs_but_not_foreign_files(exported, tmp_path):
    conn, _bench, out, dirs = exported
    tasks = store.list_tasks(conn)
    # a smaller benchmark whose export should prune the dirs of the tasks it drops
    smaller = benchmark.create(conn, "smaller", task_ids=[tasks[0].id, tasks[1].id])
    # a foreign file and dir the export must never touch
    (out / "notes.txt").write_text("keep me")
    (out / "mystuff").mkdir()
    (out / "mystuff" / "a.txt").write_text("keep me too")

    kept = harbor_export.export(conn, smaller.id, out)
    kept_names = {d.name for d in kept}
    for d in dirs:
        if d.name in kept_names:
            assert d.exists()
        else:
            assert not d.exists(), f"stale touchstone task dir {d.name} was not removed"
    assert (out / "notes.txt").exists()
    assert (out / "mystuff" / "a.txt").exists()
