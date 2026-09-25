from touchstone.config import load_settings


def test_agent_provider_defaults_to_claude_cli(tmp_path, monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_AGENT_PROVIDER", raising=False)
    assert load_settings(tmp_path / "missing.toml").agent_provider == "claude-cli"


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


def test_speech_mode_defaults_to_local(tmp_path, monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_SPEECH_MODE", raising=False)
    s = load_settings(tmp_path / "missing.toml")
    assert s.speech_mode == "local"
    assert s.realtime_model == "gpt-realtime-2.1-mini"
    assert s.realtime_voice == "marin"


def test_speech_mode_from_toml_and_env(tmp_path, monkeypatch):
    cfg = tmp_path / "touchstone.toml"
    cfg.write_text(
        '[speech]\nmode = "realtime"\nrealtime_model = "gpt-realtime-2.1"\n', encoding="utf-8"
    )
    monkeypatch.delenv("TOUCHSTONE_SPEECH_MODE", raising=False)
    s = load_settings(cfg)
    assert s.speech_mode == "realtime"
    assert s.realtime_model == "gpt-realtime-2.1"
    monkeypatch.setenv("TOUCHSTONE_SPEECH_MODE", "local")
    assert load_settings(cfg).speech_mode == "local"


def test_harbor_defaults_to_local(tmp_path, monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_HARBOR_HOST", raising=False)
    s = load_settings(tmp_path / "missing.toml")
    assert s.harbor_host == "" and s.harbor_remote_root == "~/touchstone-harbor"


def test_harbor_from_toml_and_env(tmp_path, monkeypatch):
    cfg = tmp_path / "touchstone.toml"
    cfg.write_text('[harbor]\nhost = "10.0.0.5"\nremote_root = "/data/ts"\n', encoding="utf-8")
    monkeypatch.delenv("TOUCHSTONE_HARBOR_HOST", raising=False)
    s = load_settings(cfg)
    assert s.harbor_host == "10.0.0.5" and s.harbor_remote_root == "/data/ts"
    monkeypatch.setenv("TOUCHSTONE_HARBOR_HOST", "192.168.1.1")
    assert load_settings(cfg).harbor_host == "192.168.1.1"


def test_survey_defaults(tmp_path, monkeypatch):
    for env in ("TOUCHSTONE_SURVEY_PROVIDER", "TOUCHSTONE_SURVEY_MODEL",
                "TOUCHSTONE_SURVEY_PYTHON"):
        monkeypatch.delenv(env, raising=False)
    s = load_settings(tmp_path / "missing.toml")
    assert s.survey_provider == "claude-cli"
    assert s.survey_model == "" and s.survey_python == ""
    assert s.survey_fidelity_threshold == 0.8
    assert s.survey_names == []


def test_survey_from_toml_and_env(tmp_path, monkeypatch):
    cfg = tmp_path / "touchstone.toml"
    cfg.write_text(
        '[survey]\nprovider = "codex-cli"\nmodel = "gpt-5.6-sol"\n'
        'fidelity_threshold = 0.6\nnames = ["Jane Roe"]\n',
        encoding="utf-8",
    )
    for env in ("TOUCHSTONE_SURVEY_PROVIDER", "TOUCHSTONE_SURVEY_MODEL"):
        monkeypatch.delenv(env, raising=False)
    s = load_settings(cfg)
    assert s.survey_provider == "codex-cli" and s.survey_model == "gpt-5.6-sol"
    assert s.survey_fidelity_threshold == 0.6 and s.survey_names == ["Jane Roe"]
    monkeypatch.setenv("TOUCHSTONE_SURVEY_PROVIDER", "claude-cli")
    assert load_settings(cfg).survey_provider == "claude-cli"
