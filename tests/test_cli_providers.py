"""claude-cli / codex-cli providers, driven by fake `claude` / `codex` executables on PATH.

No real CLI is ever invoked: a tmp bin dir is prepended to PATH and the fakes emit canned output
(or hang / exit non-zero) under the control of FAKE_* env vars.
"""

import json
import os
import stat

import pytest

from touchstone.llm import ProviderError
from touchstone.llm.claude_cli import ClaudeCLIProvider
from touchstone.llm.codex_cli import CodexCLIProvider

CLAUDE_FAKE = '''#!/usr/bin/env python3
import os, sys, time, json
sys.stdin.read()
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
code = int(os.environ.get("FAKE_EXIT", "0"))
if code:
    sys.stderr.write(os.environ.get("FAKE_STDERR", "boom\\n"))
    sys.exit(code)
echo = os.environ.get("FAKE_ECHO_ENV")
if echo is not None:
    print(json.dumps({"result": "VAL[" + os.environ.get(echo, "<unset>") + "]"}))
    sys.exit(0)
sys.stdout.write(os.environ.get("FAKE_STDOUT", ""))
'''

CODEX_FAKE = '''#!/usr/bin/env python3
import os, sys, time
argv = sys.argv[1:]
out = None
for i, a in enumerate(argv):
    if a == "-o" and i + 1 < len(argv):
        out = argv[i + 1]
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
code = int(os.environ.get("FAKE_EXIT", "0"))
if code:
    sys.stderr.write(os.environ.get("FAKE_STDERR", "boom\\n"))
    sys.exit(code)
content = os.environ.get("FAKE_OUT", "")
echo = os.environ.get("FAKE_ECHO_ENV")
if echo is not None:
    content = "VAL[" + os.environ.get(echo, "<unset>") + "]"
if out:
    with open(out, "w") as fh:
        fh.write(content)
sys.stdout.write("codex done\\n")
'''


def _write_exe(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_exe(bindir / "claude", CLAUDE_FAKE)
    _write_exe(bindir / "codex", CODEX_FAKE)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    return bindir


@pytest.fixture
def empty_bin(tmp_path, monkeypatch):
    bindir = tmp_path / "empty"
    bindir.mkdir()
    monkeypatch.setenv("PATH", str(bindir))
    return bindir


# ---- claude-cli ------------------------------------------------------------


def test_claude_success(fake_bin, monkeypatch):
    monkeypatch.setenv(
        "FAKE_STDOUT", json.dumps({"result": "hello world",
                                   "usage": {"input_tokens": 5, "output_tokens": 2}})
    )
    r = ClaudeCLIProvider().chat([{"role": "user", "content": "hi"}])
    assert r.content == "hello world"
    assert r.usage == {"tokens_in": 5, "tokens_out": 2}


def test_claude_tool_call(fake_bin, monkeypatch):
    result = '{"tool_calls": [{"name": "refund", "arguments": {"amount": 5}}]}'
    monkeypatch.setenv("FAKE_STDOUT", json.dumps({"result": result}))
    r = ClaudeCLIProvider("sonnet").chat([{"role": "user", "content": "refund"}])
    assert r.tool_calls[0]["name"] == "refund"


def test_claude_json_extraction(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", json.dumps({"result": 'sure: {"a": 1} ok'}))
    r = ClaudeCLIProvider().chat([{"role": "user", "content": "hi"}], json=True)
    assert json.loads(r.content) == {"a": 1}


def test_claude_malformed_stdout(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", "this is not json at all")
    with pytest.raises(ProviderError, match="valid JSON"):
        ClaudeCLIProvider().chat([{"role": "user", "content": "hi"}])


def test_claude_nonzero_exit(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    monkeypatch.setenv("FAKE_STDERR", "auth failed\n")
    with pytest.raises(ProviderError, match="exited with code 3"):
        ClaudeCLIProvider().chat([{"role": "user", "content": "hi"}])


def test_claude_timeout(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "2")
    with pytest.raises(ProviderError, match="timed out"):
        ClaudeCLIProvider(timeout=0.4).chat([{"role": "user", "content": "hi"}])


def test_claude_missing_binary_raises_at_construction(empty_bin):
    with pytest.raises(ProviderError, match="not on PATH"):
        ClaudeCLIProvider()


def test_claude_strips_anthropic_key(fake_bin, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leak-me")
    monkeypatch.setenv("FAKE_ECHO_ENV", "ANTHROPIC_API_KEY")
    r = ClaudeCLIProvider().chat([{"role": "user", "content": "hi"}])
    assert r.content == "VAL[<unset>]"
    assert "leak-me" not in r.content


@pytest.mark.asyncio
async def test_claude_async(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_STDOUT", json.dumps({"result": "async ok"}))
    r = await ClaudeCLIProvider().achat([{"role": "user", "content": "hi"}])
    assert r.content == "async ok"


# ---- codex-cli -------------------------------------------------------------


def test_codex_success_reads_outfile(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_OUT", "codex answer")
    r = CodexCLIProvider().chat([{"role": "user", "content": "hi"}])
    assert r.content == "codex answer"


def test_codex_default_model(fake_bin):
    assert CodexCLIProvider().model == "gpt-5.6-sol"


def test_codex_tool_call(fake_bin, monkeypatch):
    monkeypatch.setenv(
        "FAKE_OUT", '{"tool_calls": [{"name": "escalate", "arguments": {"r": "x"}}]}'
    )
    r = CodexCLIProvider("gpt-5.6-sol").chat([{"role": "user", "content": "help"}])
    assert r.tool_calls[0]["name"] == "escalate"


def test_codex_nonzero_exit(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "2")
    with pytest.raises(ProviderError, match="exited with code 2"):
        CodexCLIProvider().chat([{"role": "user", "content": "hi"}])


def test_codex_timeout(fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "2")
    with pytest.raises(ProviderError, match="timed out"):
        CodexCLIProvider(timeout=0.4).chat([{"role": "user", "content": "hi"}])


def test_codex_missing_binary(empty_bin):
    with pytest.raises(ProviderError, match="not on PATH"):
        CodexCLIProvider()


def test_codex_strips_openai_key(fake_bin, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    monkeypatch.setenv("FAKE_ECHO_ENV", "OPENAI_API_KEY")
    r = CodexCLIProvider().chat([{"role": "user", "content": "hi"}])
    assert r.content == "VAL[<unset>]"
    assert "sk-leak" not in r.content
