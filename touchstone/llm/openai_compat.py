"""OpenAI and any OpenAI-compatible `/v1` endpoint, over httpx directly.

We deliberately do NOT import the `openai` SDK: touchstone patches that SDK to capture the user's
own calls, and our provider calls must never be captured. Tool calling uses the function format;
`json=True` sets `response_format={"type": "json_object"}`.
"""

from __future__ import annotations

import json

import httpx

from ..messages import to_openai
from ._http import ProviderError, apost_json, post_json
from .base import Reply


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        retries: int = 2,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._transport = transport
        self.name = f"openai-compat:{model}"

    # -- request/response shaping -------------------------------------------

    @property
    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body(self, messages, tools, want_json) -> dict:
        body: dict = {"model": self.model, "messages": to_openai(messages)}
        if tools:
            body["tools"] = tools
        if want_json:
            body["response_format"] = {"type": "json_object"}
        return body

    def _parse(self, resp: httpx.Response) -> Reply:
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError("OpenAI endpoint returned a non-JSON response.") from exc
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            tool_calls.append(
                {"id": tc.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments") or ""}
            )
        usage = data.get("usage") or {}
        return Reply(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            usage={
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
            },
            raw=data,
        )

    # -- Provider protocol ---------------------------------------------------

    def chat(self, messages, tools=None, json=False, timeout=None) -> Reply:
        with httpx.Client(transport=self._transport) as client:
            resp = post_json(
                client,
                self._url,
                headers=self._headers(),
                body=self._body(messages, tools, json),
                timeout=timeout or self.timeout,
                label=f"OpenAI request to {self.model}",
                retries=self.retries,
                backoff=self.backoff,
            )
        return self._parse(resp)

    async def achat(self, messages, tools=None, json=False, timeout=None) -> Reply:
        async with httpx.AsyncClient(transport=self._transport) as client:
            resp = await apost_json(
                client,
                self._url,
                headers=self._headers(),
                body=self._body(messages, tools, json),
                timeout=timeout or self.timeout,
                label=f"OpenAI request to {self.model}",
                retries=self.retries,
                backoff=self.backoff,
            )
        return self._parse(resp)
