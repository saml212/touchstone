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


def test_keychain_prefix_defaults_to_touchstone(tmp_path, monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_KEYCHAIN_PREFIX", raising=False)
    s = load_settings(tmp_path / "missing.toml")
    assert s.keychain_prefix == "touchstone-"
    assert s.keychain_service("openai-api-key") == "touchstone-openai-api-key"


def test_keychain_prefix_from_toml_overrides_default(tmp_path):
    cfg = tmp_path / "touchstone.toml"
    cfg.write_text('keychain_prefix = "rockie-"\n', encoding="utf-8")
    assert load_settings(cfg).keychain_prefix == "rockie-"
