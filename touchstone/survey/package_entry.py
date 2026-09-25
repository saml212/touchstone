"""Generate the packaged `entry.py` and run the adapter check that decides packaged vs replica.

The survey provider writes a `def run(user_message) -> str` that drives the customer's REAL agent
loop for one message, importing their own modules (never copying them). An adapter check then runs
that entry.py locally against the live simulator with `TOUCHSTONE_MODEL` unset: it must complete,
record at least one tool call, and exit 0. On failure it retries once with the error, then the
caller falls back to the replica agent. Split out of `package.py` to keep each file small.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .. import store
from ..config import Settings
from . import fidelity
from .envs import auth_env_names, placeholder_auth
from .package_spans import first_user_turn
from .provider import SurveyProvider
from .simulate import crossing_services
from .subproc import cli_import_dir
from .writes import atomic_write

_ENTRY_TIMEOUT = 180.0

RUN_SH = """\
#!/bin/bash
set -euo pipefail
# The repo root is /app; put it on the path so entry.py can import the customer's own modules, and
# run the sibling entry.py by location so the agent dir can live anywhere in the sandbox.
export PYTHONPATH="/app:${PYTHONPATH:-}"
python "$(dirname "$0")/entry.py"
"""

ENTRY_PROMPT = '''You are packaging an existing AI agent as a test-harness entrypoint. Read the
repository (Read/Grep/Glob only — write nothing) and reply with the FULL contents of one Python
file, `agent/entry.py`, and nothing else (no prose, no markdown fences).

Map of this codebase:
{map}

entry.py must:
- `import touchstone` and call `touchstone.trace()` first, so the agent's model calls are captured.
- define `def run(user_message: str) -> str` that drives the customer's REAL agent loop for exactly
  one user message, the way production does, by importing the customer's own modules (never copy
  their code into entry.py). Build the model client exactly as the code does. Take the model id from
  the environment variable TOUCHSTONE_MODEL when it is set, otherwise the code's own default. Return
  the final assistant text.
- under `if __name__ == "__main__":` read the entire message from standard input, call run(), print
  the reply to stdout, and write {{"reply": <the reply>}} as JSON to the path in the environment
  variable TOUCHSTONE_OUTPUT, defaulting to "/app/output.json" when it is unset.

The repo root is already on sys.path, so `import <the customer's module>` works from entry.py.
Do not read argv. Do not hardcode a model id. Do not swallow errors — let them exit non-zero.
Return only the contents of entry.py.'''


def _map_digest(map_data: dict) -> str:
    keep = {k: map_data.get(k) for k in ("entrypoints", "system_prompts", "model_call", "tools")}
    return json.dumps(keep, indent=2)[:6000]


_CODE_START = ("import ", "from ", "#!", '"""', "'''", "#", "def ", "@", "__", "async ")


def _first_code_line(lines: list[str]) -> int:
    """Index of the first line that looks like Python, so a chatty provider preamble is dropped."""
    for i, line in enumerate(lines):
        if line.strip().startswith(_CODE_START):
            return i
    return 0


def _strip_fence(text: str) -> str:
    """Pull the Python out of a provider reply: the first ``` fenced block if present, else the
    body after any prose preamble. Providers sometimes narrate before the code or fence it."""
    text = text.strip()
    if "```" in text:
        body = text.split("```", 2)[1]
        lines = body.splitlines()
        if lines and lines[0].strip().lower() in ("python", "py"):
            lines = lines[1:]
        return "\n".join(lines).strip() + "\n"
    lines = text.splitlines()
    return "\n".join(lines[_first_code_line(lines):]).strip() + "\n"


def _generate_entry(provider: SurveyProvider, repo: Path, prompt: str, error: str | None) -> str:
    full = prompt if error is None else (
        f"{prompt}\n\nYour previous entry.py failed: {error}\nReturn a corrected entry.py.")
    return _strip_fence(provider.run(full, repo))


def _entry_cmd(repo: Path, entry_path: Path, settings: Settings) -> list[str]:
    if settings.survey_python:
        return [settings.survey_python, str(entry_path)]
    return ["uv", "run", "--project", str(repo), "python", str(entry_path)]


def _run_entry(repo: Path, entry_path: Path, env_extra: dict, message: str, settings: Settings):
    env = {**os.environ, **env_extra}
    env.pop("TOUCHSTONE_MODEL", None)  # the adapter check runs on the code's own default model
    # CLI's touchstone first (running code, not the customer's pin), then the repo (its modules).
    env["PYTHONPATH"] = os.pathsep.join(
        [cli_import_dir(), str(repo), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    return subprocess.run(_entry_cmd(repo, entry_path, settings), cwd=str(repo), input=message,
                          capture_output=True, text=True, timeout=_ENTRY_TIMEOUT, env=env)


def _tail(proc) -> str:
    return "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-30:])


def _span_has_tool(span) -> bool:
    if span.kind == "tool":
        return True
    return bool((span.output or {}).get("message", {}).get("tool_calls"))


def _has_tool_call(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    conn = store.connect(db_path)
    try:
        return any(_span_has_tool(s) for ep in store.list_episodes(conn)
                   for s in store.list_spans(conn, ep.id))
    finally:
        conn.close()


def _sim_mounts(map_data: dict, env_result: dict, out: Path) -> list[dict]:
    from .simulate import service_host

    envs = env_result.get("base_url_envs", {})
    return [{"sim_dir": out / "simulators" / s["name"], "env": envs.get(s["name"]),
             "host": service_host(s)} for s in crossing_services(map_data)]


def _run_and_check(agent_dir: Path, repo: Path, base_urls: dict, sim_hosts: dict, message: str,
                   settings: Settings) -> tuple[bool, str]:
    # Placeholder-fill unset auth env the repo names, so entry.py can build a token-needing client.
    auth = {n: v for n, v in placeholder_auth(auth_env_names(repo)).items() if n not in os.environ}
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "touchstone.db"
        env = {**auth, **base_urls, "TOUCHSTONE_DB": str(db),
               "TOUCHSTONE_OUTPUT": str(Path(tmp) / "out.json")}
        if sim_hosts:  # constant-host services: entry.py's trace() installs the net shim
            env["TOUCHSTONE_SIMULATORS"] = json.dumps(sim_hosts)
        proc = _run_entry(repo, agent_dir / "entry.py", env, message, settings)
        if proc.returncode != 0:
            return False, f"exit {proc.returncode}: {_tail(proc)}"
        if not _has_tool_call(db):
            return False, "entry.py recorded no tool call"
        return True, ""


def _adapter_check(agent_dir: Path, repo: Path, map_data: dict, env_result: dict, out: Path,
                   message: str, settings: Settings) -> tuple[bool, str]:
    try:
        with fidelity.simulators_running(_sim_mounts(map_data, env_result, out)) as (base_urls,
                                                                                     sim_hosts):
            return _run_and_check(agent_dir, repo, base_urls, sim_hosts, message, settings)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return False, f"adapter check error: {str(exc)[:400]}"


def _drop_packaged(agent_dir: Path) -> None:
    for name in ("entry.py", "run.sh"):
        path = agent_dir / name
        if path.exists():
            path.unlink()


def try_packaged(agent_dir: Path, repo: Path, conn, map_data: dict, env_result: dict,
                 provider: SurveyProvider, out: Path, settings: Settings) -> dict:
    """Generate entry.py + run.sh and adapter-check them; return {mode, ok, flag}. Drops the files
    and returns replica mode when the check fails twice or there is no runnable entrypoint."""
    message = first_user_turn(conn)
    if not map_data.get("entrypoints") or not message:
        return {"mode": "replica", "ok": False, "flag": "no runnable entrypoint or recorded turn"}
    prompt = ENTRY_PROMPT.format(map=_map_digest(map_data))
    error: str | None = None
    for _ in range(2):
        atomic_write(agent_dir / "entry.py", _generate_entry(provider, repo, prompt, error))
        atomic_write(agent_dir / "run.sh", RUN_SH)
        ok, reason = _adapter_check(agent_dir, repo, map_data, env_result, out, message, settings)
        if ok:
            return {"mode": "packaged", "ok": True, "flag": None}
        error = reason
    _drop_packaged(agent_dir)
    return {"mode": "replica", "ok": False, "flag": f"adapter check failed: {error}"}
