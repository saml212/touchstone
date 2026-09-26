"""`touchstone.capture.paused()` suspends model + tool span recording (a simulated user, a judge,
an evaluator is not the agent under test). Calls outside the block still record."""

from _fakes import install_fake_openai

from touchstone import store
from touchstone.capture import context, paused
from touchstone.capture.patch_openai import patch as patch_openai

RESPONSE = {"choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class Completions:
    def create(self, **kwargs):
        return RESPONSE


class AsyncCompletions:
    async def create(self, **kwargs):
        return RESPONSE


def _model_spans(conn):
    spans = []
    for ep in store.list_episodes(conn):
        spans += [s for s in store.list_spans(conn, ep.id) if s.kind == "model"]
    return spans


def test_model_call_inside_paused_records_nothing(traced, monkeypatch):
    install_fake_openai(monkeypatch, Completions, AsyncCompletions)
    assert patch_openai() is True
    from openai.resources.chat import completions as c

    with paused():
        c.Completions().create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    conn = context.get_conn()
    assert _model_spans(conn) == []

    c.Completions().create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert len(_model_spans(conn)) == 1  # the call outside the block records


def test_tool_span_inside_paused_records_nothing(traced):
    @context.tool
    def lookup(x):
        return x

    with paused():
        lookup(1)
    conn = context.get_conn()
    tools = [s for ep in store.list_episodes(conn)
             for s in store.list_spans(conn, ep.id) if s.kind == "tool"]
    assert tools == []
    lookup(2)
    tools = [s for ep in store.list_episodes(conn)
             for s in store.list_spans(conn, ep.id) if s.kind == "tool"]
    assert len(tools) == 1


def test_tool_span_records_module(traced):
    @context.tool
    def lookup(x):
        return x

    lookup(3)
    conn = context.get_conn()
    tools = [s for ep in store.list_episodes(conn)
             for s in store.list_spans(conn, ep.id) if s.kind == "tool"]
    assert len(tools) == 1
    # the tool function's defining module is recorded so the survey can attribute a shared-name call
    assert tools[0].input.get("module") == lookup.__module__
