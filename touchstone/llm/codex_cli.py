"""`codex-cli` provider: runs `codex exec` as a read-only subprocess.

Uses the Codex subscription: OPENAI_API_KEY is stripped from the child's environment so it is
never handed to codex. The prompt is passed as an argument and the final message is written to
`-o <outfile>`, which we read back and parse the same way as the claude-cli provider.
"""

from __future__ import annotations

import os

from . import _cli
from .base import Reply
from .prompt import parse_cli_result, serialize_messages

DEFAULT_MODEL = "gpt-5.6-sol"
_MISSING = "The `codex` CLI is not on PATH; install Codex to use the codex-cli provider."


class CodexCLIProvider:
    def __init__(self, model: str | None = None, *, timeout: float = 180.0) -> None:
        _cli.require_binary("codex", _MISSING)
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.name = f"codex-cli:{self.model}"

    def _run(self, messages, tools, want_json, timeout) -> Reply:
        prompt = serialize_messages(messages, tools)
        env = _cli.scrubbed_env("OPENAI_API_KEY")
        with _cli.temp_dir() as cwd:
            outfile = os.path.join(cwd, "out.txt")
            cmd = [
                "codex", "exec",
                "-m", self.model,
                "-s", "read-only",
                "--skip-git-repo-check",
                "-o", outfile,
                prompt,
            ]
            proc = _cli.run(
                cmd,
                label="codex CLI",
                cwd=cwd,
                env=env,
                timeout=timeout or self.timeout,
                stdin_text="",
            )
            result_text = _read(outfile) or (proc.stdout or "")
        return parse_cli_result(result_text, want_json)

    def chat(self, messages, tools=None, json=False, timeout=None) -> Reply:
        return self._run(messages, tools, json, timeout)

    async def achat(self, messages, tools=None, json=False, timeout=None) -> Reply:
        import asyncio

        return await asyncio.to_thread(self.chat, messages, tools, json, timeout)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""
