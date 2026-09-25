"""Generate a simulator for each service a tool reaches over the network.

Per service: gather the tool source that calls it, the real service source when it lives in the
repo, any OpenAPI docs, and up to 30 scrubbed recorded calls; ask the survey provider for a FastAPI
`app.py`, a `seed.json`, and a `README.md`; write them; measure fidelity. If fidelity is below the
threshold, regenerate once with the failing examples appended and keep whichever scored higher.
Services sharing a base-url env var collapse to one simulator.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from ..config import Settings
from ..llm.prompt import extract_json
from . import fidelity
from .provider import SurveyProvider
from .recordings import ToolEvent
from .scrub import Scrubber
from .writes import atomic_write

_SKIP_DIRS = {".venv", ".git", "__pycache__", "touchstone", ".touchstone", "node_modules"}
_MAX_SOURCE = 6000
_MAX_EXAMPLES = 30

SIM_PROMPT = """You are generating a SIMULATOR of a network service so an AI agent can be tested
offline. Reply with ONE JSON object and nothing else (no markdown fences):
{{"app.py": "<full python source>", "seed.json": <json seed state>, "README.md": "<text>"}}

app.py requirements:
- A FastAPI app runnable as `python app.py <port>`: read the port from sys.argv and call
  uvicorn.run(app, host="127.0.0.1", port=port).
- Reproduce EXACTLY the same routes, path templates, request and response bodies, and HTTP status
  codes as the real service. Do not invent routes that are not used.
- Emit the RAW upstream response body — the bytes the tool receives BEFORE it parses them. Each
  recorded call's "returned" value is what the TOOL FUNCTION returned AFTER parsing that body (read
  "Tool source" to see how). Return the raw body the tool source parses into "returned", reproducing
  the upstream wire shape, NOT the parsed result: if the tool returns response.json() unchanged the
  body equals "returned"; if it extracts nested fields (e.g. `data["current"]["temp_c"]`) the body
  must nest them so (a `current` object here), else the real tool raises KeyError on replay.
- Back it with a SQLite file named state.db beside app.py.
- Load seed.json into state.db on startup AND on `POST /__reset` (drop and recreate tables first).
- `GET /__health` returns 200 with a JSON body.
- Use only the standard library, fastapi, uvicorn, and pydantic.

seed.json is the initial state derived from the recorded calls. README.md explains what this
simulates, which routes, and how it was derived.

## Service: {name}  (kind={kind}, base_url_env={env})
## Routes observed
{routes}
## Tool source (the code that calls the service)
{tool_source}
## Response keys the tool reads (static analysis of the tool source) — the raw response body the
## simulator returns MUST provide these nested paths, or the real tool raises KeyError on replay
{keys}
## Real service source (same repo, may be copied closely)
{service_source}
## API docs / OpenAPI
{docs}
## Recorded calls (scrubbed; tool, arguments, and the tool's PARSED return — reconstruct the raw
## response body from the tool source above)
{examples}
{hint}
Return only the JSON object."""


def crossing_services(map_data: dict) -> list[dict]:
    """Services a tool actually reaches over the network, one per base-url env var (else per name).

    A service no tool calls (commonly the model SDK the agent thinks with) is not a boundary to
    simulate — the model is swapped by setting — so it is dropped rather than given an empty sim."""
    called = {c for t in map_data.get("tools", []) for c in (t.get("calls") or [])}
    seen: set = set()
    out: list[dict] = []
    for service in map_data.get("services", []):
        if service.get("name") not in called:
            continue
        key = service.get("base_url_env") or f"name:{service.get('name')}"
        if key not in seen:
            seen.add(key)
            out.append(service)
    return out


def service_tools(map_data: dict, service: dict) -> list[dict]:
    name = service.get("name")
    return [t for t in map_data.get("tools", []) if name in (t.get("calls") or [])]


def service_host(service: dict) -> str | None:
    """The host in the service's constant base URL, so the net shim can rewrite it when there is no
    base_url_env to override (e.g. `http://api.weatherapi.com/v1` -> `api.weatherapi.com`)."""
    from urllib.parse import urlsplit

    default = service.get("base_url_default")
    return urlsplit(default).hostname if default else None


def _replay_ctx(service: dict, tools: list[dict]) -> dict:
    return {"base_url_env": service.get("base_url_env"), "host": service_host(service),
            "kind": service.get("kind"),
            "tools": {t["name"]: t["import_path"] for t in tools}}


# ---- prompt materials ------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:_MAX_SOURCE]
    except OSError:
        return ""


def _tool_source(repo: Path, tools: list[dict]) -> str:
    files = dict.fromkeys(t.get("file") for t in tools if t.get("file"))
    blocks = [f"### {f}\n{_read(repo / f)}" for f in files if _read(repo / f)]
    return "\n\n".join(blocks) or "(tool source not found)"


def _const_str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _subscript_path(node: ast.Subscript) -> list[str]:
    """The dot path of constant string keys read off a subscript chain (data['a']['b'] -> a.b)."""
    keys: list[str] = []
    while isinstance(node, ast.Subscript):
        key = _const_str(node.slice)
        if key is None:
            break
        keys.append(key)
        node = node.value
    return list(reversed(keys))


def response_key_paths(source: str) -> list[str]:
    """Nested key paths the tool source reads off an HTTP response (subscript chains), so the
    simulator knows the raw body shape the recordings never show. Longest paths only."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            keys = _subscript_path(node)
            if keys:
                paths.add(".".join(keys))
    return sorted(p for p in paths if not any(o != p and o.startswith(p + ".") for o in paths))


def _route_needles(service: dict) -> list[str]:
    needles = []
    for call in service.get("calls", []):
        template = call.get("path_template", "")
        needles.append(template.split("{", 1)[0].rstrip("/") or template)
    return [n for n in needles if n]


def _iter_py(repo: Path):
    for path in repo.rglob("*.py"):
        if not any(part in _SKIP_DIRS for part in path.relative_to(repo).parts):
            yield path


def _service_source(repo: Path, service: dict) -> str:
    needles = _route_needles(service)
    for path in _iter_py(repo):
        text = _read(path)
        if text and any(n in text for n in needles):
            return f"### {path.relative_to(repo)}\n{text}"
    return "(not found in repo)"


def _openapi(repo: Path) -> str:
    for pattern in ("openapi.json", "openapi.yaml", "openapi.yml", "*openapi*"):
        for path in repo.rglob(pattern):
            if path.is_file():
                return _read(path)
    return "(none)"


def _base_path(service: dict) -> str:
    """The path prefix the simulator must serve under. A constant-base-URL service is reached by the
    net shim, which rewrites only scheme+host and keeps the path, so the base URL's own path (e.g.
    /v1) is part of every request and the routes must include it. An env-configurable service is
    repointed by replacing the whole base URL (path included), so its routes are the templates.
    """
    from urllib.parse import urlsplit

    if service.get("base_url_env"):
        return ""
    return urlsplit(service.get("base_url_default") or "").path.rstrip("/")


def _routes_text(service: dict) -> str:
    bp = _base_path(service)
    lines = [f"{c.get('method')} {bp}{c.get('path_template')} (from {c.get('from_tool')})"
             for c in service.get("calls", [])]
    return "\n".join(lines) or "(none recorded)"


def _examples(calls: list[ToolEvent], scrub: Scrubber) -> list[dict]:
    return [{"tool": c.tool, "arguments": scrub.scrub(c.arguments),
             "returned": scrub.scrub(c.output)} for c in calls[:_MAX_EXAMPLES]]


def _keys_text(source: str) -> str:
    paths = response_key_paths(source)
    return "\n".join(f"- {p}" for p in paths) if paths else "(none detected)"


def _prompt(repo: Path, service: dict, tools: list[dict], examples: list[dict],
            hint: str = "") -> str:
    source = _tool_source(repo, tools)
    return SIM_PROMPT.format(
        name=service.get("name"), kind=service.get("kind"),
        env=service.get("base_url_env"), routes=_routes_text(service),
        tool_source=source, keys=_keys_text(source),
        service_source=_service_source(repo, service),
        docs=_openapi(repo), examples=json.dumps(examples, indent=2, ensure_ascii=False),
        hint=hint)


# ---- generate + write ------------------------------------------------------


def _parse_files(text: str) -> dict:
    raw = extract_json(text)
    if raw is None:
        raise ValueError("simulator answer contained no JSON object")
    data = json.loads(raw)
    if not isinstance(data, dict) or "app.py" not in data:
        raise ValueError("simulator answer missing 'app.py'")
    return data


def _as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)


_LOOPBACK = "127.0.0.1"


def _force_loopback(app_source: str) -> str:
    """A generated simulator must bind loopback only: during fidelity/capture it runs on the host as
    a plain subprocess, so a model-emitted `host="0.0.0.0"` would expose the seeded service on the
    LAN. Rewrite any bind-all address to 127.0.0.1 — the literal 0.0.0.0 and a quoted "::" have no
    other use in this generated FastAPI app (a `[::2]` slice is unquoted, so it survives)."""
    src = app_source.replace("0.0.0.0", _LOOPBACK)
    return re.sub(r"""(host\s*=\s*)(['"])::\2""", rf"\1\g<2>{_LOOPBACK}\2", src)


def _write_sim(sim_dir: Path, files: dict) -> None:
    atomic_write(sim_dir / "app.py", _force_loopback(_as_text(files.get("app.py", ""))))
    atomic_write(sim_dir / "seed.json", _as_text(files.get("seed.json", {})))
    atomic_write(sim_dir / "README.md", _as_text(files.get("README.md", "")))


def _snapshot(sim_dir: Path) -> dict:
    return {n: (sim_dir / n).read_text(encoding="utf-8")
            for n in ("app.py", "seed.json", "README.md") if (sim_dir / n).exists()}


def _restore(sim_dir: Path, snapshot: dict) -> None:
    for name, text in snapshot.items():
        atomic_write(sim_dir / name, text)


_KEYERROR = re.compile(r"KeyError: ['\"]([^'\"]+)['\"]")


def _missing_keys(failures: list[dict]) -> list[str]:
    """Keys the real tool looked for but the simulator's body did not provide (from KeyError)."""
    found: list[str] = []
    for f in failures:
        err = (f.get("got") or {}).get("__error__", "") if isinstance(f.get("got"), dict) else ""
        found += _KEYERROR.findall(err)
    return sorted(set(found))


def _failure_hint(result: dict) -> str:
    failures = result.get("failures", [])[:5]
    missing = _missing_keys(result.get("failures", []))
    miss = (f"\nThe tool raised KeyError for these keys — the response body MUST include them at "
            f"the right nesting: {missing}\n" if missing else "")
    return ("## Your previous simulator failed these examples; fix them:\n"
            + json.dumps(failures, indent=2, ensure_ascii=False) + miss)


def _measure(sim_dir, repo, calls, ctx, settings, scrub):
    return fidelity.measure_service(sim_dir, repo, calls, ctx, settings, scrub)


def _retry(provider, repo, service, tools, examples, sim_dir, calls, ctx, settings, scrub, prev):
    snapshot = _snapshot(sim_dir)
    files = _parse_files(provider.run(_prompt(repo, service, tools, examples,
                                              _failure_hint(prev)), repo))
    _write_sim(sim_dir, files)
    new = _measure(sim_dir, repo, calls, ctx, settings, scrub)
    if new["score"] >= prev["score"]:
        return new
    _restore(sim_dir, snapshot)  # keep the better (previous) simulator
    return prev


def generate_simulator(repo: Path, provider: SurveyProvider, service: dict, tools: list[dict],
                       events: list[ToolEvent], sim_root: Path, scrub: Scrubber,
                       settings: Settings, force: bool = False) -> dict:
    """Write simulators/<service>/ and return its fidelity result. Reuses files unless force."""
    sim_dir = sim_root / service["name"]
    ctx = _replay_ctx(service, tools)
    names = {t["name"] for t in tools}
    calls = [e for e in events if e.tool in names]
    if (sim_dir / "app.py").exists() and not force:
        return _measure(sim_dir, repo, calls, ctx, settings, scrub)
    examples = _examples(calls, scrub)
    _write_sim(sim_dir, _parse_files(provider.run(_prompt(repo, service, tools, examples), repo)))
    result = _measure(sim_dir, repo, calls, ctx, settings, scrub)
    if result["score"] < settings.survey_fidelity_threshold:
        result = _retry(provider, repo, service, tools, examples, sim_dir, calls, ctx,
                        settings, scrub, result)
    return result
