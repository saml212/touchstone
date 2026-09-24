import re

from typer.testing import CliRunner

from touchstone.cli import app

runner = CliRunner()

_ID = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")


def _train_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    runner.invoke(app, ["demo", "--n", "12"])
    runner.invoke(app, ["checks", "add", "--kind", "contains",
                        "--params", '{"values": ["sorted", "escalat"], "mode": "any"}',
                        "--name", "polite"])
    runner.invoke(app, ["mine", "--no-llm"])
    runner.invoke(app, ["bench", "create", "demo", "--all"])
    return "demo"


def test_train_prepare_writes_datasets(tmp_path, monkeypatch):
    bid = _train_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["train", "prepare", bid])
    assert result.exit_code == 0, result.output
    assert "sft.jsonl" in result.output and "rl_tasks.jsonl" in result.output
    out_dir = tmp_path / ".touchstone" / "train" / "demo"
    assert (out_dir / "sft.jsonl").exists()
    assert (out_dir / "manifest.json").exists()


def test_train_prepare_custom_out(tmp_path, monkeypatch):
    bid = _train_repo(tmp_path, monkeypatch)
    out = tmp_path / "custom"
    result = runner.invoke(app, ["train", "prepare", bid, "--out", str(out)])
    assert result.exit_code == 0
    assert (out / "rl_tasks.jsonl").exists()


def test_train_submit_null_writes_plan(tmp_path, monkeypatch):
    bid = _train_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["train", "submit", bid, "--backend", "null"])
    assert result.exit_code == 0, result.output
    assert "planned" in result.output
    assert (tmp_path / ".touchstone" / "train" / "demo" / "train_plan.md").exists()


def test_train_submit_art_prints_infra_instruction(tmp_path, monkeypatch):
    bid = _train_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["train", "submit", bid, "--backend", "art"])
    assert result.exit_code == 0
    assert "art_train.py" in result.output
    assert (tmp_path / ".touchstone" / "train" / "demo" / "art_train.py").exists()


def test_train_submit_unknown_backend_fails(tmp_path, monkeypatch):
    bid = _train_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["train", "submit", bid, "--backend", "nope"])
    assert result.exit_code == 1
    assert "unknown backend" in result.output


def test_train_prepare_unknown_benchmark_fails(tmp_path, monkeypatch):
    _train_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["train", "prepare", "nope"])
    assert result.exit_code == 1
    assert "no active tasks" in result.output


def test_train_prepare_unwritable_out_fails_cleanly(tmp_path, monkeypatch):
    import os
    import stat

    bid = _train_repo(tmp_path, monkeypatch)
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    try:
        result = runner.invoke(
            app, ["train", "prepare", bid, "--out", str(ro / "sub")], catch_exceptions=False
        )
        assert result.exit_code == 1
        assert "\n" not in result.output.strip()
    finally:
        os.chmod(ro, stat.S_IRWXU)
