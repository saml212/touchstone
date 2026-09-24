import json

import pytest

from touchstone.llm import Rule, ScriptedProvider, provider_from_spec


def _msgs(text):
    return [{"role": "user", "content": text}]


def test_deterministic_same_input_same_output():
    p = ScriptedProvider()
    a = p.chat(_msgs("where is my order"))
    b = p.chat(_msgs("where is my order"))
    assert a.content == b.content and a.content.startswith("scripted-reply-")


def test_rule_returns_tool_call_with_string_arguments():
    rule = Rule("refund", tool_call={"name": "refund", "arguments": {"amt": 5}})
    p = ScriptedProvider(rules=[rule])
    r = p.chat(_msgs("please refund me"))
    assert r.tool_calls[0]["name"] == "refund"
    assert isinstance(r.tool_calls[0]["arguments"], str)
    assert json.loads(r.tool_calls[0]["arguments"]) == {"amt": 5}


def test_rule_can_emit_malformed_json():
    p = ScriptedProvider(rules=[Rule("break", malformed_json=True)])
    r = p.chat(_msgs("break the parser"))
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.content)


def test_json_mode_wraps_content():
    p = ScriptedProvider()
    r = p.chat(_msgs("hello"), json=True)
    assert json.loads(r.content)  # valid JSON object


def test_registry_scripted_and_unknown():
    assert isinstance(provider_from_spec("scripted"), ScriptedProvider)
    with pytest.raises(ValueError, match="unknown provider spec"):
        provider_from_spec("gemini:flash")


@pytest.mark.parametrize("spec", ["openai:", "anthropic:", "openai-compatible:http://x/v1:"])
def test_registry_rejects_empty_model(spec):
    with pytest.raises(ValueError, match="model"):
        provider_from_spec(spec)


def test_openai_missing_key_error_names_service_and_knob(monkeypatch):
    from touchstone.config import Settings
    from touchstone.llm import ProviderError, registry

    monkeypatch.setattr(registry, "secret", lambda *a, **k: None)
    with pytest.raises(ProviderError) as ei:
        registry.provider_from_spec("openai:gpt-4o", Settings())
    msg = str(ei.value)
    assert "OPENAI_API_KEY" in msg
    assert "touchstone-openai-api-key" in msg  # the exact keychain service it looked for
    assert "keychain_prefix" in msg


def test_anthropic_missing_key_error_names_service_and_knob(monkeypatch):
    from touchstone.config import Settings
    from touchstone.llm import ProviderError, registry

    monkeypatch.setattr(registry, "secret", lambda *a, **k: None)
    with pytest.raises(ProviderError) as ei:
        registry.provider_from_spec("anthropic:claude-3", Settings())
    msg = str(ei.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "touchstone-anthropic-api-key" in msg
    assert "keychain_prefix" in msg


def test_doctor_provider_rows_name_the_keychain_service(monkeypatch):
    from touchstone.config import Settings
    from touchstone.llm import registry

    monkeypatch.setattr(registry, "secret", lambda *a, **k: None)
    rows = {s.name: s for s in registry.provider_statuses(Settings())}
    assert "touchstone-openai-api-key" in rows["openai"].detail
    assert "touchstone-anthropic-api-key" in rows["anthropic"].detail
