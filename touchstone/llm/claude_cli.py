"""`claude-cli` provider: runs `claude -p --output-format json` as a subprocess.

Uses the Claude Code subscription, not an API key: ANTHROPIC_API_KEY is stripped from the
child's environment. System messages + tool schemas go on `--system-prompt`; the conversation is
fed on stdin; the CLI's `result` text is parsed back into a Reply (tool calls via the JSON
convention, JSON extraction).
"""

from __future__ import annotations

import json

from . import _cli
from ._http import ProviderError
from .base import Reply
from .prompt import parse_cli_result, split_for_cli

_MISSING = "The `claude` CLI is not on PATH; install Claude Code to use the claude-cli provider."


class ClaudeCLIProvider:
    def __init__(self, model: str | None = None, *, timeout: float = 180.0) -> None:
        _cli.require_binary("claude", _MISSING)
        self.model = model
        self.timeout = timeout
        self.name = f"claude-cli:{model}" if model else "claude-cli"

    def _cmd(self, system: str) -> list[str]:
        # Instructions travel on the CLI's real system prompt, never inside the user turn.
        cmd = ["claude", "-p", "--output-format", "json", "--permission-mode", "default"]
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def _run(self, messages, tools, want_json, timeout) -> Reply:
        system, prompt = split_for_cli(messages, tools)
        env = _cli.scrubbed_env("ANTHROPIC_API_KEY")
        with _cli.temp_dir() as cwd:
            proc = _cli.run_retrying(
                self._cmd(system),
                label="claude CLI",
                cwd=cwd,
                env=env,
                timeout=timeout or self.timeout,
                stdin_text=prompt,
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError("claude CLI did not return valid JSON.") from exc
        result_text = data.get("result", "") if isinstance(data, dict) else ""
        reply = parse_cli_result(result_text, want_json)
        reply.usage = _usage(data)
        return reply

    def chat(self, messages, tools=None, json=False, timeout=None) -> Reply:
        return self._run(messages, tools, json, timeout)

    async def achat(self, messages, tools=None, json=False, timeout=None) -> Reply:
        import asyncio

        return await asyncio.to_thread(self.chat, messages, tools, json, timeout)


def _usage(data) -> dict | None:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    return {"tokens_in": usage.get("input_tokens"), "tokens_out": usage.get("output_tokens")}
