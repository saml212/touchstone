"""The CLI's own touchstone is put on PYTHONPATH for the replay/adapter subprocesses.

A customer venv may pin a different (or no) touchstone, so the survey subprocesses must import the
running CLI's code, not whatever the customer interpreter would resolve — otherwise a version drift
breaks the survey with "No module named 'touchstone.survey'".
"""

import os
import subprocess

from touchstone.config import Settings
from touchstone.survey import fidelity, package_entry, subproc


def test_with_cli_path_prepends_cli_dir():
    cli = subproc.cli_import_dir()
    assert cli and os.path.isdir(cli)
    assert subproc.with_cli_path({})["PYTHONPATH"] == cli
    assert subproc.with_cli_path({"PYTHONPATH": "/foo"})["PYTHONPATH"] == cli + os.pathsep + "/foo"


def test_import_dir_exposes_only_touchstone_not_the_clis_other_sdks():
    # The crux: for a wheel/uv-tool install the package's parent is the whole site-packages, so the
    # dir on PYTHONPATH must hold ONLY a touchstone symlink — otherwise the CLI's pydantic/httpx
    # would shadow the customer's own (their native pydantic_core then fails to load).
    import pathlib
    d = pathlib.Path(subproc.cli_import_dir())
    assert [p.name for p in d.iterdir()] == ["touchstone"]
    link = d / "touchstone"
    assert link.is_symlink() or link.is_dir()
    assert (link / "__init__.py").is_file()  # resolves to the real touchstone package


def _capture_env(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured.update(kw.get("env") or {})

        class _P:  # minimal stand-in for CompletedProcess
            returncode, stdout, stderr = 0, "[]", ""
        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_replay_subprocess_gets_cli_touchstone_on_pythonpath(tmp_path, monkeypatch):
    # The replay subprocess (fidelity._run_spec) runs in the customer interpreter; its PYTHONPATH
    # must lead with the CLI's touchstone dir so `import touchstone.survey.replay` is the CLI code.
    env = _capture_env(monkeypatch)
    fidelity._run_spec(tmp_path, {"tools": {}, "calls": []}, Settings())
    assert env["PYTHONPATH"].split(os.pathsep)[0] == subproc.cli_import_dir()


def test_adapter_subprocess_gets_cli_touchstone_before_the_repo(tmp_path, monkeypatch):
    # entry.py imports the customer's own modules from the repo, but touchstone itself must resolve
    # to the CLI: the CLI dir comes first, the repo second.
    env = _capture_env(monkeypatch)
    package_entry._run_entry(tmp_path, tmp_path / "entry.py", {}, "hi", Settings())
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == subproc.cli_import_dir() and parts[1] == str(tmp_path)
