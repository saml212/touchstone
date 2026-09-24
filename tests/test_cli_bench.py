import re

from typer.testing import CliRunner

from touchstone.cli import app

runner = CliRunner()

_ID = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")


def _bench_repo(tmp_path, monkeypatch):
    """A repo with a demo db, two enabled checks, mined tasks, and a benchmark id."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    runner.invoke(app, ["demo", "--n", "12"])
    runner.invoke(app, ["checks", "add", "--kind", "contains",
                        "--params", '{"values": ["sorted", "escalat"], "mode": "any"}',
                        "--name", "polite"])
    runner.invoke(app, ["checks", "add", "--kind", "no_pii", "--params", "{}", "--name", "clean"])
    runner.invoke(app, ["mine", "--no-llm"])
    out = runner.invoke(app, ["bench", "create", "demo", "--all"]).output
    return _ID.search(out).group()


def test_bench_create_run_report_flow(tmp_path, monkeypatch):
    bid = _bench_repo(tmp_path, monkeypatch)

    run = runner.invoke(app, ["bench", "run", bid, "-m", "scripted"])
    assert run.exit_code == 0, run.output
    assert "MODELS" in run.output and "scripted" in run.output

    runs = runner.invoke(app, ["bench", "runs"])
    assert runs.exit_code == 0 and "done" in runs.output

    report = runner.invoke(app, ["bench", "report", "--json"])
    assert report.exit_code == 0 and '"models"' in report.output


def test_bench_proof_between_two_runs(tmp_path, monkeypatch):
    bid = _bench_repo(tmp_path, monkeypatch)
    r1 = _ID.search(runner.invoke(app, ["bench", "run", bid, "-m", "scripted"]).output).group()
    r2 = _ID.search(runner.invoke(app, ["bench", "run", bid, "-m", "scripted"]).output).group()
    proof = runner.invoke(app, ["bench", "proof", r1, r2, "--json"])
    assert proof.exit_code == 0
    assert '"counts"' in proof.output and '"both_pass"' in proof.output


def test_bench_run_bad_benchmark_fails_cleanly(tmp_path, monkeypatch):
    _bench_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["bench", "run", "nope", "-m", "scripted"])
    assert result.exit_code == 1
    assert "run failed" in result.output


def test_export_harbor_and_atif(tmp_path, monkeypatch):
    bid = _bench_repo(tmp_path, monkeypatch)
    out = tmp_path / "harbor"
    result = runner.invoke(app, ["export", "harbor", bid, "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "exported" in result.output
    assert any((d / "task.toml").exists() for d in out.iterdir())

    # atif export of a real captured episode
    from touchstone import store
    conn = store.connect(str(tmp_path / ".touchstone" / "touchstone.db"))
    episode_id = store.list_episodes(conn)[0].id
    conn.close()
    atif = runner.invoke(app, ["export", "atif", episode_id, "--out", str(tmp_path / "e.json")])
    assert atif.exit_code == 0 and (tmp_path / "e.json").exists()


def test_export_atif_unknown_episode_fails(tmp_path, monkeypatch):
    _bench_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["export", "atif", "does-not-exist"])
    assert result.exit_code == 1


def test_harbor_run_reports_missing_docker(tmp_path, monkeypatch):
    bid = _bench_repo(tmp_path, monkeypatch)
    out = tmp_path / "harbor"
    runner.invoke(app, ["export", "harbor", bid, "--out", str(out)])
    task_dir = next(out.iterdir())
    result = runner.invoke(app, ["bench", "harbor-run", str(task_dir), "--agent", "claude"])
    assert result.exit_code == 1
    assert "harbor run -p" in result.output


def test_bench_proof_bad_run_fails_cleanly(tmp_path, monkeypatch):
    bid = _bench_repo(tmp_path, monkeypatch)
    r1 = _ID.search(runner.invoke(app, ["bench", "run", bid, "-m", "scripted"]).output).group()
    result = runner.invoke(app, ["bench", "proof", r1, "nope"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "\n" not in result.output.strip()
    assert "run" in result.output.lower()


def test_checks_eval_malformed_tool_calls_fails_cleanly(tmp_path, monkeypatch):
    _bench_repo(tmp_path, monkeypatch)
    from touchstone import store
    conn = store.connect(str(tmp_path / ".touchstone" / "touchstone.db"))
    cid = store.list_checks(conn)[0].id
    conn.close()
    result = runner.invoke(
        app, ["checks", "eval", cid, "--tool-calls", "5"], catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "\n" not in result.output.strip()
