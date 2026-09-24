from touchstone.config import load_settings


def test_agent_provider_defaults_to_codex_cli(tmp_path, monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_AGENT_PROVIDER", raising=False)
    assert load_settings(tmp_path / "missing.toml").agent_provider == "codex-cli"


def test_agent_provider_from_toml(tmp_path):
    cfg = tmp_path / "touchstone.toml"
    cfg.write_text('agent_provider = "openai:gpt-4o-mini"\n', encoding="utf-8")
    assert load_settings(cfg).agent_provider == "openai:gpt-4o-mini"


def test_agent_provider_env_overrides_toml(tmp_path, monkeypatch):
    cfg = tmp_path / "touchstone.toml"
    cfg.write_text('agent_provider = "openai:gpt-4o-mini"\n', encoding="utf-8")
    monkeypatch.setenv("TOUCHSTONE_AGENT_PROVIDER", "scripted")
    assert load_settings(cfg).agent_provider == "scripted"
