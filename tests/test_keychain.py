import subprocess

from touchstone.llm import keychain


def _fake_run(stdout="", returncode=0):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return run


def test_env_first_no_subprocess(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")

    def boom(*a, **k):
        raise AssertionError("subprocess must not run when env is set")

    monkeypatch.setattr(subprocess, "run", boom)
    assert keychain.secret("MY_KEY", "rockie-openai-api-key") == "from-env"


def test_blank_env_falls_through_to_keychain(monkeypatch):
    monkeypatch.setenv("MY_KEY", "   ")
    monkeypatch.setattr(keychain.sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout="kc-value\n"))
    assert keychain.secret("MY_KEY", "svc") == "kc-value"


def test_keychain_used_when_env_absent(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setattr(keychain.sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout="secret\n"))
    assert keychain.secret("MY_KEY", "svc") == "secret"


def test_keychain_miss_returns_none(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setattr(keychain.sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", _fake_run(returncode=44))
    assert keychain.secret("MY_KEY", "svc") is None


def test_non_darwin_skips_keychain(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setattr(keychain.sys, "platform", "linux")

    def boom(*a, **k):
        raise AssertionError("keychain must not be queried off macOS")

    monkeypatch.setattr(subprocess, "run", boom)
    assert keychain.secret("MY_KEY", "svc") is None


def test_security_binary_missing_is_graceful(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setattr(keychain.sys, "platform", "darwin")

    def missing(*a, **k):
        raise FileNotFoundError("no security tool")

    monkeypatch.setattr(subprocess, "run", missing)
    assert keychain.secret("MY_KEY", "svc") is None
