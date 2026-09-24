"""Fake `openai` / `anthropic` modules injected into sys.modules so the patches can be tested
without the real SDKs or any network."""

import sys
import types


def install_fake_openai(monkeypatch, completions_cls, async_cls,
                        responses_cls=None, async_responses_cls=None):
    completions = types.ModuleType("openai.resources.chat.completions")
    completions.Completions = completions_cls
    completions.AsyncCompletions = async_cls
    chat = types.ModuleType("openai.resources.chat")
    chat.completions = completions
    resources = types.ModuleType("openai.resources")
    resources.chat = chat
    openai = types.ModuleType("openai")
    openai.resources = resources
    mods = [
        ("openai", openai),
        ("openai.resources", resources),
        ("openai.resources.chat", chat),
        ("openai.resources.chat.completions", completions),
    ]
    if responses_cls is not None:
        responses = types.ModuleType("openai.resources.responses")
        responses.Responses = responses_cls
        responses.AsyncResponses = async_responses_cls
        resources.responses = responses
        mods.append(("openai.resources.responses", responses))
    for name, mod in mods:
        monkeypatch.setitem(sys.modules, name, mod)


def install_fake_anthropic(monkeypatch, messages_cls, async_cls):
    messages = types.ModuleType("anthropic.resources.messages")
    messages.Messages = messages_cls
    messages.AsyncMessages = async_cls
    resources = types.ModuleType("anthropic.resources")
    resources.messages = messages
    anthropic = types.ModuleType("anthropic")
    anthropic.resources = resources
    for name, mod in [
        ("anthropic", anthropic),
        ("anthropic.resources", resources),
        ("anthropic.resources.messages", messages),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
