"""A transient CLI API error is retried once; a real error or timeout is not."""

from __future__ import annotations

import pytest

from touchstone.llm import _cli
from touchstone.llm._http import ProviderError

_TRANSIENT = ('claude CLI exited with code 1: {"is_error":true,"api_error_status":400,'
              '"result":"API Error: 400 Tool reference \'x\' not found in available tools"}')


def _runner(outcomes):
    """A fake `_cli.run` that yields each outcome in turn (an Exception is raised)."""
    calls = {"n": 0}

    def run(cmd, *, label, cwd, env, timeout, stdin_text=None):
        calls["n"] += 1
        outcome = outcomes[calls["n"] - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return run, calls


def test_transient_error_retries_once_then_succeeds(monkeypatch):
    run, calls = _runner([ProviderError(_TRANSIENT), "OK"])
    monkeypatch.setattr(_cli, "run", run)
    out = _cli.run_retrying(["x"], label="claude CLI", cwd=".", env={}, timeout=1,
                            sleep=lambda _p: None)
    assert out == "OK" and calls["n"] == 2


def test_non_transient_error_is_not_retried(monkeypatch):
    run, calls = _runner([ProviderError("claude CLI exited with code 127: command not found")])
    monkeypatch.setattr(_cli, "run", run)
    with pytest.raises(ProviderError, match="command not found"):
        _cli.run_retrying(["x"], label="claude CLI", cwd=".", env={}, timeout=1,
                          sleep=lambda _p: None)
    assert calls["n"] == 1


def test_two_transient_errors_still_raise(monkeypatch):
    run, calls = _runner([ProviderError(_TRANSIENT), ProviderError(_TRANSIENT)])
    monkeypatch.setattr(_cli, "run", run)
    with pytest.raises(ProviderError, match="Tool reference"):
        _cli.run_retrying(["x"], label="claude CLI", cwd=".", env={}, timeout=1,
                          sleep=lambda _p: None)
    assert calls["n"] == 2
