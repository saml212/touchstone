"""TOUCHSTONE_MODEL rewrites the model on all three capture patches before the SDK call goes thru.

The fake SDKs record the `model` they were called with; the assertions check both that value and the
recorded span's model. A positional model (no keyword) is left untouched.
"""

import asyncio
import sys
import types

from _fakes import install_fake_anthropic, install_fake_openai

from touchstone import store
from touchstone.capture import context, spans
from touchstone.capture.litellm import install
from touchstone.capture.patch_anthropic import patch as patch_anthropic
from touchstone.capture.patch_openai import patch as patch_openai

OPENAI_RESPONSE = {"choices": [{"message": {"content": "ok"}}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
ANTHROPIC_RESPONSE = {"content": [{"type": "text", "text": "ok"}],
                      "usage": {"input_tokens": 1, "output_tokens": 1}}


class OpenAICompletions:
    seen: dict = {}

    def create(self, **kwargs):
        OpenAICompletions.seen = dict(kwargs)
        return OPENAI_RESPONSE


class OpenAIAsync:
    async def create(self, **kwargs):
        OpenAICompletions.seen = dict(kwargs)
        return OPENAI_RESPONSE


class AnthropicMessages:
    seen: dict = {}

    def create(self, **kwargs):
        AnthropicMessages.seen = dict(kwargs)
        return ANTHROPIC_RESPONSE


class AnthropicAsync:
    async def create(self, **kwargs):
        AnthropicMessages.seen = dict(kwargs)
        return ANTHROPIC_RESPONSE


def _last_span():
    conn = context.get_conn()
    eps = store.list_episodes(conn)
    return store.list_spans(conn, eps[-1].id)[-1]


def test_override_helper_only_touches_keyword_model(monkeypatch):
    monkeypatch.setenv("TOUCHSTONE_MODEL", "cheap-1")
    kwargs = {"model": "gpt-4o", "messages": []}
    spans.override_model(kwargs)
    assert kwargs["model"] == "cheap-1"
    positional = {"messages": []}  # model passed positionally -> no keyword to rewrite
    spans.override_model(positional)
    assert "model" not in positional


def test_openai_model_rewritten(traced, monkeypatch):
    install_fake_openai(monkeypatch, OpenAICompletions, OpenAIAsync)
    monkeypatch.setenv("TOUCHSTONE_MODEL", "openai-cheap")
    assert patch_openai() is True
    from openai.resources.chat import completions as c

    c.Completions().create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert OpenAICompletions.seen["model"] == "openai-cheap"  # SDK saw the forced model
    assert _last_span().model == "openai-cheap"  # and it was recorded


def test_anthropic_model_rewritten(traced, monkeypatch):
    install_fake_anthropic(monkeypatch, AnthropicMessages, AnthropicAsync)
    monkeypatch.setenv("TOUCHSTONE_MODEL", "claude-cheap")
    assert patch_anthropic() is True
    from anthropic.resources import messages as m

    m.Messages().create(model="claude-3", messages=[{"role": "user", "content": "hi"}])
    assert AnthropicMessages.seen["model"] == "claude-cheap"
    assert _last_span().model == "claude-cheap"


def _fake_litellm(monkeypatch):
    seen: dict = {}

    def completion(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    litellm = types.ModuleType("litellm")
    litellm.completion = completion
    litellm.callbacks = []
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    return litellm, seen


def test_litellm_model_rewritten(monkeypatch):
    litellm, seen = _fake_litellm(monkeypatch)
    monkeypatch.setenv("TOUCHSTONE_MODEL", "litellm-cheap")
    assert install() is True
    litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert seen["model"] == "litellm-cheap"


def test_no_env_leaves_model_untouched(monkeypatch):
    litellm, seen = _fake_litellm(monkeypatch)
    monkeypatch.delenv("TOUCHSTONE_MODEL", raising=False)
    assert install() is True
    litellm.completion(model="gpt-4o", messages=[])
    assert seen["model"] == "gpt-4o"


def test_async_openai_model_rewritten(traced, monkeypatch):
    install_fake_openai(monkeypatch, OpenAICompletions, OpenAIAsync)
    monkeypatch.setenv("TOUCHSTONE_MODEL", "openai-cheap-async")
    assert patch_openai() is True
    from openai.resources.chat import completions as c

    asyncio.run(c.AsyncCompletions().create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]))
    assert OpenAICompletions.seen["model"] == "openai-cheap-async"
