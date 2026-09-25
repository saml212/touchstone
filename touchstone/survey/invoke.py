"""Generate agent/invoke.py: call the customer's real tools by NAME, the way their code does.

Some agents don't expose each tool as a free function `f(**model_args)` — the model's tool name is
dispatched through a helper that needs a constructed client (`execute_function(client, name, args)`)
or maps to a bound method. The replica replay's `import_path(**arguments)` can't drive those. So the
survey provider writes `agent/invoke.py` with `invoke(name, arguments) -> object` that imports the
customer's modules and constructs whatever the tool needs from the environment (never copying code),
then calls it. It is adapter-checked against the live simulator: one recorded call per tool must
dispatch without raising. When it passes, fidelity, task criteria, and the replica dispatch route
through it; when it is absent or fails, they fall back to the mapped import path. Nothing is
target-specific — the generated file naturally carries the customer's own specifics.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..config import Settings
from . import fidelity
from .provider import SurveyProvider
from .recordings import ToolEvent
from .simulate import _tool_source, crossing_services, service_host
from .writes import atomic_write

_PROBE_TIMEOUT = 150.0

INVOKE_PROMPT = '''You are writing ONE Python file, agent/invoke.py, that lets a harness call an
existing AI agent's tools by name. Reply with the FULL contents of invoke.py and nothing else (no
prose, no markdown fences).

Map of the agent's tools (name, import path, the service each calls) and its model call:
{map}

Tool source (how the code actually calls each tool):
{tool_source}

invoke.py must:
- define `def invoke(name: str, arguments: dict) -> object` that calls the customer's REAL tool
  `name` with keyword `arguments`, EXACTLY as the agent's own code calls it. Import the customer's
  own modules (never copy their code). If the tool is a method on a client/session, or is dispatched
  through a helper (e.g. execute_function(client, name, arguments)), construct that client/helper
  same way the code does, reading base URLs and API keys from os.environ so the harness can point
  them at a simulator. Return whatever the tool returns.
- raise KeyError (or ValueError) if `name` is unknown. Do NOT swallow tool errors or read argv or
  stdin. The repo root is already on sys.path, so importing the customer's modules works.
Return only the contents of invoke.py.'''


def _map_digest(map_data: dict) -> str:
    keep = {"tools": map_data.get("tools"), "services": map_data.get("services"),
            "model_call": map_data.get("model_call")}
    return json.dumps(keep, indent=2)[:6000]


def _generate(provider: SurveyProvider, repo: Path, map_data: dict, tools: list[dict]) -> str:
    from .package import _strip_fence

    prompt = INVOKE_PROMPT.format(map=_map_digest(map_data), tool_source=_tool_source(repo, tools))
    return _strip_fence(provider.run(prompt, repo))


def _mounts(map_data: dict, out: Path) -> list[dict]:
    return [{"sim_dir": out / "simulators" / s["name"], "env": s.get("base_url_env"),
             "host": service_host(s)} for s in crossing_services(map_data)]


def _invoke_cmd(repo: Path, settings: Settings, invoke_path: Path, name: str,
                args_json: str) -> list[str]:
    base = ([settings.survey_python] if settings.survey_python
            else ["uv", "run", "--project", str(repo), "python"])
    return [*base, "-m", "touchstone.survey.replay", "--invoke-one", str(invoke_path), name,
            args_json]


def _run_probe(repo: Path, invoke_path: Path, call: ToolEvent, env: dict,
               settings: Settings) -> tuple[bool, str]:
    args = json.dumps(call.arguments if isinstance(call.arguments, dict) else {}, default=str)
    proc = subprocess.run(_invoke_cmd(repo, settings, invoke_path, call.tool, args),
                          cwd=str(repo), capture_output=True, text=True, timeout=_PROBE_TIMEOUT,
                          env=env)
    if proc.returncode != 0:
        return False, f"invoke.py raised for {call.tool}: {(proc.stderr or proc.stdout)[-400:]}"
    try:
        got = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, f"invoke.py output for {call.tool} was not JSON"
    if isinstance(got, dict) and "__error__" in got:
        return False, f"invoke.py error for {call.tool}: {got['__error__']}"
    return True, ""


def _probe(repo: Path, invoke_path: Path, calls: list[ToolEvent], base_urls: dict, sim_hosts: dict,
           settings: Settings) -> tuple[bool, str]:
    env = {**os.environ, **base_urls}
    if sim_hosts:
        env["TOUCHSTONE_SIMULATORS"] = json.dumps(sim_hosts)
    seen: set = set()
    for call in calls:
        if call.tool in seen:
            continue
        seen.add(call.tool)
        ok, detail = _run_probe(repo, invoke_path, call, env, settings)
        if not ok:
            return False, detail
    return True, ""


def _adapter_check(repo: Path, map_data: dict, out: Path, invoke_path: Path,
                   calls: list[ToolEvent], settings: Settings) -> tuple[bool, str]:
    try:
        with fidelity.simulators_running(_mounts(map_data, out)) as (base_urls, sim_hosts):
            return _probe(repo, invoke_path, calls, base_urls, sim_hosts, settings)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return False, f"adapter check error: {str(exc)[:400]}"


def build_invoke(repo: Path, map_data: dict, provider: SurveyProvider, events: list[ToolEvent],
                 out: Path, settings: Settings, force: bool = False) -> dict:
    """Write agent/invoke.py and adapter-check it. Returns {ok, path, flag}. Kept only when the
    check passes (one recorded call per tool dispatches without raising); else removed so callers
    fall back to the mapped import path."""
    path = out / "agent" / "invoke.py"
    if path.exists() and not force:
        return {"ok": True, "path": str(path), "flag": None}
    names = {t["name"] for t in map_data.get("tools", [])}
    calls = [e for e in events if e.tool in names]
    if not calls:
        return {"ok": False, "path": None, "flag": "no recorded tool calls to drive invoke.py"}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, _generate(provider, repo, map_data, list(map_data.get("tools", []))))
    ok, flag = _adapter_check(repo, map_data, out, path, calls, settings)
    if ok:
        return {"ok": True, "path": str(path), "flag": None}
    path.unlink()
    return {"ok": False, "path": None, "flag": f"invoke.py adapter check failed: {flag}"}
