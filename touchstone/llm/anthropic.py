"""Anthropic Messages API over httpx (no `anthropic` SDK — see openai_compat for why).

Converts OpenAI-style messages/tools into Anthropic's shape: system text is extracted, tool
schemas become `input_schema`, assistant tool calls become `tool_use` blocks and tool results
become `tool_result` blocks in a user turn. Same retry/timeout behaviour as the OpenAI provider.
"""

from __future__ import annotations

import json

import httpx

from ..messages import to_anthropic
from ..messages_wire import to_anthropic_tools
from ._http import ProviderError, apost_json, post_json
from .base import Reply

API_VERSION = "2023-06-01"
_JSON_INSTRUCTION = "Respond with a single valid JSON object and nothing else."


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        retries: int = 2,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._transport = transport
        self.name = f"anthropic:{model}"

    @property
    def _url(self) -> str:
        return f"{self.base_url}/v1/messages"

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
        }

    def _body(self, messages, tools, want_json) -> dict:
        system, converted = to_anthropic(messages)
        if want_json:
            system = (system + "\n\n" + _JSON_INSTRUCTION).strip()
        body: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = to_anthropic_tools(tools)
        return body

    def _parse(self, resp: httpx.Response) -> Reply:
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError("Anthropic endpoint returned a non-JSON response.") from exc
        texts, tool_calls = [], []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    }
                )
        usage = data.get("usage") or {}
        return Reply(
            content="".join(texts),
            tool_calls=tool_calls,
            usage={
                "tokens_in": usage.get("input_tokens"),
                "tokens_out": usage.get("output_tokens"),
            },
            raw=data,
        )

    def chat(self, messages, tools=None, json=False, timeout=None) -> Reply:
        with httpx.Client(transport=self._transport) as client:
            resp = post_json(
                client,
                self._url,
                headers=self._headers(),
                body=self._body(messages, tools, json),
                timeout=timeout or self.timeout,
                label=f"Anthropic request to {self.model}",
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
                label=f"Anthropic request to {self.model}",
                retries=self.retries,
                backoff=self.backoff,
            )
        return self._parse(resp)
