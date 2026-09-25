"""Replay recorded tool calls through the customer's REAL tool functions.

Run inside the customer's environment (`python -m touchstone.survey.replay <spec.json>`), so the
tools hit the code and dependency versions they run in production — only the service base URL is
swapped for the simulator's. The base-url env var is set BEFORE any tool module is imported, because
a module may build its HTTP client at import time. Output is a JSON list on stdout.

Spec: {"base_url_env": str|null, "base_url": str, "tools": {name: "module:function"},
       "calls": [{"tool": name, "arguments": <obj|value>}]}
"""

from __future__ import annotations

import importlib
import json
import os
import sys


def _load(import_path: str):
    module_name, _, function = import_path.partition(":")
    return getattr(importlib.import_module(module_name), function)


def _invoke(fn, arguments):
    if isinstance(arguments, dict):
        return fn(**arguments)
    return fn(arguments)  # non-object arguments are passed raw


def replay(spec: dict) -> list[dict]:
    env = spec.get("base_url_env")
    if env:
        os.environ[env] = spec["base_url"]
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


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        spec = json.load(fh)
    json.dump(replay(spec), sys.stdout, default=str)


if __name__ == "__main__":
    main()
