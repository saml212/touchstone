"""litellm CustomLogger callback that records an llm span. litellm is imported lazily and never
required; the logger works even when litellm is absent (base class falls back to object)."""

from __future__ import annotations

from datetime import datetime

from . import spans
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
        result = _extract(response_obj) if response_obj is not None else spans._ERROR
        err = error or (repr(kwargs["exception"]) if kwargs.get("exception") else None)
        try:
            spans.record("litellm", model, messages, tools, {}, result, err, _iso(start_time))
        except Exception as exc:  # capture must never break the caller
            spans._log.warning("touchstone capture failed: %r", exc)

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
