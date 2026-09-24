"""litellm CustomLogger callback that records an llm span. litellm is imported lazily and never
required; the logger works even when litellm is absent (base class falls back to object)."""

from __future__ import annotations

from datetime import datetime

from . import context
from .patch_openai import _extract  # litellm ModelResponse is OpenAI-shaped


def _base():
    try:
        from litellm.integrations.custom_logger import CustomLogger

        return CustomLogger
    except Exception:
        return object


def _iso(t) -> str:
    if isinstance(t, datetime):
        return t.isoformat()
    from .. import store

    return store.now()


class TouchstoneLogger(_base()):
    def _record(self, kwargs, response_obj, start_time, error):
        model = kwargs.get("model")
        messages = kwargs.get("messages") or []
        opt = kwargs.get("optional_params") or {}
        tools = opt.get("tools") or kwargs.get("tools")
        content, tool_calls, usage = ("", [], None)
        if response_obj is not None:
            content, tool_calls, usage = _extract(response_obj)
        err = error or (repr(kwargs["exception"]) if kwargs.get("exception") else None)
        context.add_span(
            "llm",
            model or "litellm",
            model=model,
            input={"messages": messages, "tools": tools or [], "params": {}},
            output={"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}},
            tokens_in=usage.get("tokens_in") if usage else None,
            tokens_out=usage.get("tokens_out") if usage else None,
            error=err,
            started_at=_iso(start_time),
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, None)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, "failure")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, None)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, "failure")


def install() -> bool:
    """Register the logger with litellm if it is installed. Returns False if litellm is absent."""
    try:
        import litellm
    except Exception:
        return False
    logger = TouchstoneLogger()
    litellm.callbacks = list(getattr(litellm, "callbacks", []) or []) + [logger]
    return True
