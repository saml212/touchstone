"""Map the repo: run the survey agent once and get one validated `map.json`.

The agent reads the code (Read/Grep/Glob only) and answers with a single JSON document describing
entrypoints, system prompts, tools, the model call site, network services, and tool schemas. The
answer is validated against `MAP_SCHEMA`; an invalid answer is retried once with the validation
error, then fails loudly. `map.json` is cached — reused unless `force`.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import ValidationError, validate

from ..llm.prompt import extract_json
from .provider import SurveyProvider
from .writes import atomic_write_json

_INT_OR_NULL = {"type": ["integer", "null"]}
_STR_OR_NULL = {"type": ["string", "null"]}

MAP_SCHEMA = {
    "type": "object",
    "required": ["entrypoints", "tools", "services", "model_call"],
    "properties": {
        "entrypoints": {"type": "array", "items": {"type": "string"}},
        "system_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file", "line", "text"],
                "properties": {"file": {"type": "string"}, "line": _INT_OR_NULL,
                               "text": {"type": "string"}},
            },
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "import_path", "file", "line", "calls"],
                "properties": {
                    "name": {"type": "string"}, "import_path": {"type": "string"},
                    "file": {"type": "string"}, "line": _INT_OR_NULL,
                    "calls": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "model_call": {
            "type": "object",
            "required": ["file", "line", "sdk", "model_setting"],
            "properties": {"file": {"type": "string"}, "line": _INT_OR_NULL,
                           "sdk": {"type": "string"}, "model_setting": {"type": "string"}},
        },
        "services": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "kind", "base_url_env", "base_url_default", "calls"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"enum": ["http", "sdk", "db"]},
                    "base_url_env": _STR_OR_NULL, "base_url_default": _STR_OR_NULL,
                    "calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["method", "path_template", "from_tool"],
                            "properties": {"method": {"type": "string"},
                                           "path_template": {"type": "string"},
                                           "from_tool": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "schemas": {"type": ["object", "array"]},
    },
}

MAP_PROMPT = """You are mapping an AI-agent codebase so it can be copied into a test sandbox.
Read the repository (use Read, Grep, and Glob only — do not write anything) and reply with ONE
JSON object and nothing else (no prose, no markdown fences). Use this exact shape:

{
  "entrypoints": ["<file:function that starts the agent>"],
  "system_prompts": [{"file": "<path>", "line": <int>, "text": "<first 200 chars>"}],
  "tools": [{"name": "<tool name the model calls>", "import_path": "<module:function>",
             "file": "<path>", "line": <int>, "calls": ["<service name it talks to>"]}],
  "model_call": {"file": "<path>", "line": <int>, "sdk": "<openai|anthropic|...>",
                 "model_setting": "<how the model id is chosen>"},
  "services": [{"name": "<short name>", "kind": "http|sdk|db",
                "base_url_env": "<env var holding the base url, or null>",
                "base_url_default": "<default base url, or null>",
                "calls": [{"method": "<GET|POST|...>", "path_template": "/path/{id}",
                           "from_tool": "<tool name>"}]}],
  "schemas": {"<tool name>": {<the JSON schema the model sees for that tool>}}
}

A "tool" is a function the model can call. A "service" is anything a tool reaches over the
network: an HTTP API, a database driver, or a vendor SDK. A tool that reaches no service has an
empty "calls" list. Do not list the model/LLM SDK the agent calls to think (e.g. the OpenAI or
Anthropic client behind model_call) as a service unless a TOOL calls it — the model is swapped by
setting, not simulated.

"base_url_env" is the environment variable that holds the service's BASE URL or host, and nothing
else. An API key, token, or secret env var (e.g. WEATHER_API_KEY, STRIPE_API_KEY, X_API_KEY) is NOT
a base_url_env — if only the key comes from the environment and the host is written into the source,
set "base_url_env" to null and put the hard-coded host in "base_url_default".

A tool that reaches a database in-process — a SQL driver or ORM, or an in-memory store loaded from
JSON/dict files that is passed to every tool — is a "db" service, not "http". For a db service,
"base_url_env" is the env var the code reads for the database path or URL (else null), and
"base_url_default" is the literal it uses (":memory:", a file path, or a URL like postgresql://...).

Report every tool and every service you find. Keep "text" to 200 characters. Return only the JSON
object."""


def _parse_and_validate(text: str) -> dict:
    raw = extract_json(text)
    if raw is None:
        raise ValueError("the map answer contained no JSON object")
    data = json.loads(raw)  # JSONDecodeError is a ValueError
    validate(data, MAP_SCHEMA)
    return data


def _obtain(provider: SurveyProvider, repo: Path, prompt: str) -> dict:
    text = provider.run(prompt, repo)
    try:
        return _parse_and_validate(text)
    except (ValueError, ValidationError) as first:
        retry = (f"{prompt}\n\nYour previous answer was invalid: {first}\n"
                 "Return only the corrected JSON object.")
        return _parse_and_validate(provider.run(retry, repo))


def build_map(repo: Path, provider: SurveyProvider, out_dir: Path, force: bool = False) -> dict:
    """Load `map.json` if present (unless force); otherwise run the agent and write it."""
    path = out_dir / "map.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    data = _obtain(provider, repo, MAP_PROMPT)
    atomic_write_json(path, data)
    return data
