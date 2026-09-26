"""Gate logic over fixture job rates: oracle 1 / nop 0 passes; oracle 0.5 or nop 1 fails, moves."""

import json
import tomllib
from pathlib import Path

import tomli_w

from touchstone.survey import gate


def _task(out: Path, name: str, gated: bool = False) -> None:
    d = out / "tasks" / name
    d.mkdir(parents=True)
    ts = {"episodes": ["e"], "job": name, "tools": ["t"]}
    if gated:
        ts.update({"oracle": 1.0, "nop": 0.0, "gated_at": "2026-09-25T00:00:00Z"})
    doc = {"schema_version": "1.3", "task": {"name": f"d/{name}"},
           "metadata": {"touchstone": ts}, "environment": {"docker_image": "img:1"}}
    (d / "task.toml").write_text(tomli_w.dumps(doc), encoding="utf-8")


def _patch(monkeypatch, oracle: dict, nop: dict, calls: list | None = None,
           seen: dict | None = None):
    monkeypatch.setattr(gate.run_mod, "build_image", lambda *a, **k: calls.append("build")
                        if calls is not None else None)

    def fake_run(out, agent, **k):
        if seen is not None:
            seen["jobs_dir"] = k.get("jobs_dir")
        return Path(agent)
    monkeypatch.setattr(gate.run_mod, "run", fake_run)
    monkeypatch.setattr(gate, "_rates", lambda job: oracle if job.name == "oracle" else nop)


def _env():
    return {"deps_ok": True, "image_tag": "img:1"}


def test_gate_pass_fail_and_move(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _task(out, "good")
    _task(out, "weak-oracle")
    _task(out, "trivial-nop")
    seen: dict = {}
    _patch(monkeypatch,
           oracle={"good": 1.0, "weak-oracle": 0.5, "trivial-nop": 1.0},
           nop={"good": 0.0, "weak-oracle": 0.0, "trivial-nop": 1.0}, seen=seen)
    result = gate.run_gate(tmp_path, _env(), settings=None)

    assert seen["jobs_dir"] == out / "jobs"  # job dirs under the dataset root, not cwd
    assert result["gated"] == ["good"]
    # per-task oracle/nop rates are carried for the report's Gate table
    assert result["rates"]["good"] == {"oracle": 1.0, "nop": 0.0}
    assert result["rates"]["weak-oracle"]["oracle"] == 0.5
    # passing task stays and records its gate result
    doc = tomllib.loads((out / "tasks" / "good" / "task.toml").read_text())
    assert doc["metadata"]["touchstone"]["oracle"] == 1.0
    assert doc["metadata"]["touchstone"]["nop"] == 0.0
    assert "gated_at" in doc["metadata"]["touchstone"]

    # failing tasks moved to needs-review with gate.json naming the failing side
    review = {r["name"]: r for r in result["needs_review"]}
    assert review["weak-oracle"]["failed_side"] == "oracle"
    assert review["trivial-nop"]["failed_side"] == "nop"
    assert not (out / "tasks" / "weak-oracle").exists()
    moved = json.loads((out / "needs-review" / "weak-oracle" / "gate.json").read_text())
    assert moved["oracle"] == 0.5


def test_gate_idempotent_when_all_gated(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _task(out, "done", gated=True)

    def _boom(*a, **k):
        raise AssertionError("harbor must not run when all tasks are gated")

    monkeypatch.setattr(gate.run_mod, "build_image", _boom)
    monkeypatch.setattr(gate.run_mod, "run", _boom)
    result = gate.run_gate(tmp_path, _env(), settings=None)
    assert result["gated"] == ["done"]


def test_gate_skipped_when_deps_unresolved(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _task(out, "good")
    monkeypatch.setattr(gate.run_mod, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no run")))
    env = {"deps_ok": False, "deps_reason": "no pyproject.toml", "image_tag": "img:1"}
    result = gate.run_gate(tmp_path, env, settings=None)
    assert "no pyproject" in result["skipped_gate"]
    assert (out / "tasks" / "good").exists()  # not moved


def test_gate_harbor_failure_moves_to_needs_review(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _task(out, "good")
    import subprocess
    monkeypatch.setattr(gate.run_mod, "build_image", lambda *a, **k: None)

    def _fail(*a, **k):
        raise subprocess.CalledProcessError(1, "harbor")

    monkeypatch.setattr(gate.run_mod, "run", _fail)
    result = gate.run_gate(tmp_path, _env(), settings=None)
    assert result["gated"] == []
    assert result["needs_review"][0]["failed_side"] == "harbor"
    assert (out / "needs-review" / "good" / "gate.json").exists()


def test_gate_force_reruns_gated(tmp_path, monkeypatch):
    out = tmp_path / "touchstone"
    _task(out, "done", gated=True)
    calls = []
    _patch(monkeypatch, oracle={"done": 1.0}, nop={"done": 0.0}, calls=calls)
    result = gate.run_gate(tmp_path, _env(), settings=None, force=True)
    assert result["gated"] == ["done"]
    assert calls == ["build"]  # force triggered a real (mocked) build + run


def test_call_captures_output_on_failure():
    import pytest

    from touchstone.harbor import run as run_mod
    with pytest.raises(RuntimeError) as exc:
        run_mod._call(["sh", "-c", "echo hello-stdout; echo boom-stderr >&2; exit 3"])
    msg = str(exc.value)
    assert "exit 3" in msg
    assert "hello-stdout" in msg and "boom-stderr" in msg


def test_gate_failure_reason_reaches_gate_json_and_report(tmp_path, monkeypatch):
    from touchstone.survey.report import needs_review_section
    out = tmp_path / "touchstone"
    _task(out, "good")
    monkeypatch.setattr(gate.run_mod, "build_image", lambda *a, **k: None)

    def _fail(*a, **k):
        raise RuntimeError("command failed (exit 1): harbor run\nValueError: no tasks\nline2")

    monkeypatch.setattr(gate.run_mod, "run", _fail)
    result = gate.run_gate(tmp_path, _env(), settings=None)
    stored = json.loads((out / "needs-review" / "good" / "gate.json").read_text())
    assert "no tasks" in stored["reason"]
    # the report shows only the first line of the reason
    md = needs_review_section(result)
    assert "command failed (exit 1)" in md
    assert "line2" not in md


def test_build_image_local(monkeypatch, tmp_path):
    from touchstone.config import Settings
    from touchstone.harbor import run as run_mod
    seen = []
    monkeypatch.setattr(run_mod, "_has_docker", lambda: True)
    monkeypatch.setattr(run_mod, "_exec", lambda cmd, *a, **k: (seen.append(cmd), (0, ""))[1])
    run_mod.build_image(tmp_path, "img:1", Settings(harbor_host=""))
    assert seen == [["docker", "build", "-t", "img:1", str(tmp_path)]]


def test_build_image_remote(monkeypatch, tmp_path):
    from touchstone.config import Settings
    from touchstone.harbor import run as run_mod
    seen = []
    monkeypatch.setattr(run_mod, "_has_docker", lambda: False)
    monkeypatch.setattr(run_mod, "remote_docker_daemon", lambda _s: (True, "27"))
    monkeypatch.setattr(run_mod, "_exec", lambda cmd, *a, **k: (seen.append(cmd), (0, ""))[1])
    run_mod.build_image(tmp_path, "img:1", Settings(harbor_host="h", harbor_remote_root="/r"))
    assert any(c[0] == "rsync" for c in seen)
    assert any(c[0] == "ssh" and "docker build -t img:1" in c[-1] for c in seen)
