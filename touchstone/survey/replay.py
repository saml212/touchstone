"""Replay recorded tool calls through the customer's REAL tool functions.

Run inside the customer's environment (`python -m touchstone.survey.replay <spec.json>`), so the
tools hit the code and dependency versions they run in production — only the service base URL is
swapped for the simulator's. The base-url env var is set BEFORE any tool module is imported, because
a module may build its HTTP client at import time. Output is a JSON list on stdout.

Spec: {"base_url_env": str|null, "base_url": str, "tools": {name: "module:function"},
       "calls": [{"tool": name, "arguments": <obj|value>}]}
`base_urls` ({env: url}) may point several services at their simulators at once; it is applied
before the single `base_url_env`/`base_url` pair (both are optional).

`--one <module:function> '<json arguments>'` replays a single call and prints its JSON result — the
form the packaged/replica agent execs per tool call. It reads the service base URL from the process
environment the caller set (no spec), so the tool module resolves its URL at import time as usual.
"""

from __future__ import annotations

import importlib
import json
import os
import sys


def _apply_base_urls(spec: dict) -> None:
    for env, url in (spec.get("base_urls") or {}).items():
        if env:
            os.environ[env] = url
    env = spec.get("base_url_env")
    if env:
        os.environ[env] = spec["base_url"]


def _apply_simulators(spec: dict) -> None:
    """Install the net shim for a service with a constant (non-env) base URL, so a tool that
    hard-codes its host is rewritten to the simulator instead of hitting the real service."""
    sims = spec.get("simulators")
    if sims:
        os.environ["TOUCHSTONE_SIMULATORS"] = json.dumps(sims)
    from .netshim import install_from_env

    install_from_env()


def _load(import_path: str):
    module_name, _, function = import_path.partition(":")
    return getattr(importlib.import_module(module_name), function)


def _invoke(fn, arguments):
    if isinstance(arguments, dict):
        return fn(**arguments)
    return fn(arguments)  # non-object arguments are passed raw


def replay(spec: dict) -> list[dict]:
    _apply_base_urls(spec)
    _apply_simulators(spec)
    tools = spec["tools"]
    results = []
    for call in spec["calls"]:
        name = call["tool"]
        try:
            got = _invoke(_load(tools[name]), call.get("arguments"))
        except Exception as exc:  # noqa: BLE001 — any tool failure is data, not a crash
            got = {"__error__": f"{type(exc).__name__}: {exc}"}
        results.append({"tool": name, "got": got})
    return results


def replay_one(import_path: str, arguments_json: str):
    """Run a single recorded call through its real tool function and return the raw result."""
    arguments = json.loads(arguments_json) if arguments_json else {}
    try:
        return _invoke(_load(import_path), arguments)
    except Exception as exc:  # noqa: BLE001 — a tool failure is data, not a crash
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--one":
        from .netshim import install_from_env

        install_from_env()  # honour TOUCHSTONE_SIMULATORS the caller set for a constant-host tool
        json.dump(replay_one(argv[1], argv[2] if len(argv) > 2 else ""), sys.stdout, default=str)
        return
    with open(argv[0], encoding="utf-8") as fh:
        spec = json.load(fh)
    json.dump(replay(spec), sys.stdout, default=str)


if __name__ == "__main__":
    main()
