"""Simulator for a `kind == "db"` service: a SQLite `state.db` the customer's real tools read.

Unlike an HTTP simulator there is no `app.py` server and no port. The survey provider returns ONE
JSON object describing the database in one of two shapes:

  relational      {"schema.sql": "<CREATE TABLE ...>", "seed.json": {"<table>": [ {row}, ... ]},
                   "README.md": "..."}
  document store  {"collections": {"orders": {"<id>": {...doc...}}, ...}, "README.md": "..."}

Touchstone (never the model) materializes `simulators/<name>/state.db` from those files. A document
store becomes one table per collection `(id TEXT PRIMARY KEY, doc TEXT)` with the document in
`doc`, so task criteria stay `sqlite_query_equals(..., json_extract(doc,'$.field'), ...)`. A service
whose tools cannot run against SQLite is written as `UNSUPPORTED.md` and scored 0 with the reason.

`python -m touchstone.survey.db_sim <sim_dir>` (re)builds state.db from the written files;
`start.sh` runs it in the image before replay.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from ..llm.prompt import extract_json
from .db_service import state_db
from .writes import atomic_write, atomic_write_json

DB_SIM_PROMPT = """You are generating a SIMULATOR of a DATABASE so an AI agent can be tested
offline.
The agent's tools read and write this database in-process (a SQL driver/ORM, or an in-memory store
loaded from JSON documents) — there is NO HTTP server. Reply with ONE JSON object and nothing else
(no markdown fences), in ONE of these two shapes:

Relational (the tools run SQL):
{{"schema.sql": "<CREATE TABLE ... for sqlite>", "seed.json": {{"<table>": [ {{<row>}}, ... ]}},
  "README.md": "<text>"}}

Document store (the tools work on dicts of JSON documents, e.g. data["orders"][id]):
{{"collections": {{"<collection>": {{"<id>": {{<document>}}, ...}}, ...}}, "README.md": "<text>"}}

Rules:
- Pick document-store shape when the tools index a dict of documents (data["orders"][order_id]);
  pick relational shape when the tools run SQL. Use ONLY sqlite-compatible SQL.
- The seed/collections MUST contain every row or document any recorded call below returned or
  touched, with ids and values EXACTLY as recorded (already scrubbed), plus enough additional
  consistent rows that list/search tools return plausible results.
- Do not invent tables or fields the tools never read. Keep README.md to what this simulates and how
  it was derived.
- If the tools CANNOT run against SQLite (e.g. raw psycopg with no configurable URL, a non-SQL
  engine), reply instead with {{"UNSUPPORTED": "<one-sentence reason>"}}.

## Service: {name}  (env var the tools read for the database: {env})
## Tool source (the code that reaches the database)
{tool_source}
## Schema source found in the repo (CREATE TABLE / models / migrations / seed JSON)
{schema_source}
## Recorded calls (scrubbed; tool, arguments, and the tool's returned value)
{examples}
{hint}
Return only the JSON object."""


# ---- provider answer -> files ----------------------------------------------


def parse_files(text: str) -> dict:
    """Parse the answer into one of {schema.sql,...}, {collections:...}, {UNSUPPORTED:...}."""
    raw = extract_json(text)
    if raw is None:
        raise ValueError("db simulator answer contained no JSON object")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("db simulator answer was not a JSON object")
    if _unsupported_reason(data) or "collections" in data or "schema.sql" in data:
        return data
    raise ValueError("db answer has no 'schema.sql', 'collections', nor 'UNSUPPORTED'")


def _unsupported_reason(data: dict) -> str | None:
    for key in ("UNSUPPORTED", "unsupported"):
        if isinstance(data.get(key), str) and data[key].strip():
            return data[key].strip()
    return None


def _as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)


def write_sim(sim_dir: Path, files: dict) -> None:
    """Write the db simulator's source files (never an app.py); UNSUPPORTED is its own file."""
    sim_dir.mkdir(parents=True, exist_ok=True)
    _clear(sim_dir)
    reason = _unsupported_reason(files)
    if reason:
        write_unsupported(sim_dir, reason)
        return
    atomic_write(sim_dir / "README.md", _as_text(files.get("README.md", "")))
    if "collections" in files:
        atomic_write_json(sim_dir / "collections.json", files["collections"])
        return
    atomic_write(sim_dir / "schema.sql", _as_text(files.get("schema.sql", "")))
    atomic_write_json(sim_dir / "seed.json", files.get("seed.json", {}))


def write_unsupported(sim_dir: Path, reason: str) -> None:
    sim_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(sim_dir / "UNSUPPORTED.md", reason.strip() + "\n")


def unsupported(sim_dir: Path) -> str | None:
    path = Path(sim_dir) / "UNSUPPORTED.md"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


_ARTIFACTS = ("schema.sql", "seed.json", "collections.json", "UNSUPPORTED.md", "state.db")


def _clear(sim_dir: Path) -> None:
    for name in _ARTIFACTS:
        (sim_dir / name).unlink(missing_ok=True)


# ---- materialize state.db --------------------------------------------------


def materialize(sim_dir: str | Path) -> Path | None:
    """Build a fresh state.db from the written files. Returns its path, or None when unsupported."""
    sim_dir = Path(sim_dir)
    db = state_db(sim_dir)
    if unsupported(sim_dir):
        db.unlink(missing_ok=True)
        return None
    db.unlink(missing_ok=True)
    conn = sqlite3.connect(str(db))
    try:
        if (sim_dir / "collections.json").is_file():
            _materialize_collections(conn, _read_json(sim_dir / "collections.json"))
        else:
            _materialize_relational(conn, sim_dir)
        conn.commit()
    finally:
        conn.close()
    return db


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _materialize_relational(conn, sim_dir: Path) -> None:
    schema = (sim_dir / "schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema)
    seed = _read_json(sim_dir / "seed.json") if (sim_dir / "seed.json").is_file() else {}
    for table, rows in seed.items():
        for row in rows or []:
            _insert_row(conn, table, row)


def _insert_row(conn, table: str, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    columns = ", ".join(cols)
    values = [_scalar(row[c]) for c in cols]
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)


def _scalar(value):
    """SQLite stores scalars; a nested object/array is stored as its JSON text."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _materialize_collections(conn, collections: dict) -> None:
    for name, docs in collections.items():
        conn.execute(f"CREATE TABLE {name} (id TEXT PRIMARY KEY, doc TEXT)")
        for doc_id, doc in _doc_items(docs):
            conn.execute(f"INSERT INTO {name} (id, doc) VALUES (?, ?)",
                         [str(doc_id), json.dumps(doc, ensure_ascii=False, sort_keys=True)])


def _doc_items(docs):
    """A collection as (id, document) pairs, accepting {id: doc} or a list of documents."""
    if isinstance(docs, dict):
        return list(docs.items())
    return [(d.get("id") if isinstance(d, dict) else i, d) for i, d in enumerate(docs or [])]


def is_document_store(sim_dir: str | Path) -> bool:
    return (Path(sim_dir) / "collections.json").is_file()


# ---- prompt materials + generation -----------------------------------------


_SCHEMA_NEEDLES = ("create table", "sqlalchemy", "declarative_base", "table(", "column(")


def _schema_py_blocks(repo: Path) -> list[str]:
    from .simulate import _iter_py, _read

    blocks: list[str] = []
    for path in _iter_py(repo):
        text = _read(path)
        if text and any(n in text.lower() for n in _SCHEMA_NEEDLES):
            blocks.append(f"### {path.relative_to(repo)}\n{text}")
    return blocks


def _schema_json_blocks(repo: Path) -> list[str]:
    blocks: list[str] = []
    for path in sorted(repo.rglob("*.json")):
        rel = path.relative_to(repo)
        parts = set(rel.parts)
        if "data" in parts and not parts & {".git", "touchstone", ".touchstone"}:
            blocks.append(f"### {rel}\n{path.read_text(encoding='utf-8', errors='replace')}")
    return blocks


def _schema_source(repo: Path) -> str:
    """Files that describe the database shape: CREATE TABLE / ORM models / migrations, and small
    JSON data files a document store loads. Capped like a service source block."""
    from .simulate import _MAX_SOURCE

    blocks = _schema_py_blocks(repo) + _schema_json_blocks(repo)
    return "\n\n".join(blocks)[:_MAX_SOURCE] or "(no schema source found in repo)"


def _db_prompt(repo: Path, service: dict, tool_source: str, examples: list[dict],
               hint: str = "") -> str:
    from .db_service import env_name

    return DB_SIM_PROMPT.format(
        name=service.get("name"), env=env_name(service), tool_source=tool_source,
        schema_source=_schema_source(repo),
        examples=json.dumps(examples, indent=2, ensure_ascii=False), hint=hint)


_SNAPSHOT = ("schema.sql", "seed.json", "collections.json", "README.md", "UNSUPPORTED.md")


def _snapshot(sim_dir: Path) -> dict:
    return {n: (sim_dir / n).read_text(encoding="utf-8")
            for n in _SNAPSHOT if (sim_dir / n).is_file()}


def _restore(sim_dir: Path, snapshot: dict) -> None:
    _clear(sim_dir)
    for name, text in snapshot.items():
        atomic_write(sim_dir / name, text)


def _failure_hint(result: dict) -> str:
    failures = result.get("failures", [])[:5]
    return ("## Your previous db simulator failed these examples; fix the seed/schema so the tools "
            "return the expected values:\n" + json.dumps(failures, indent=2, ensure_ascii=False))


def _generate(provider, repo: Path, service: dict, tool_source: str, examples, hint=""):
    return parse_files(provider.run(_db_prompt(repo, service, tool_source, examples, hint), repo))


def generate_db_simulator(repo, provider, service: dict, tools: list[dict], events, sim_root: Path,
                          scrub, settings, force: bool = False) -> dict:
    """Write simulators/<name>/ for a db service and return its fidelity result (one retry)."""
    from . import fidelity
    from .simulate import _examples, _replay_ctx, _tool_source

    sim_dir = sim_root / service["name"]
    ctx = _replay_ctx(service, tools)
    calls = [e for e in events if e.tool in {t["name"] for t in tools}]
    if _has_files(sim_dir) and not force:
        return fidelity.measure_service(sim_dir, repo, calls, ctx, settings, scrub)
    tool_source = _tool_source(repo, tools)
    examples = _examples(calls, scrub)
    write_sim(sim_dir, _generate(provider, repo, service, tool_source, examples))
    result = fidelity.measure_service(sim_dir, repo, calls, ctx, settings, scrub)
    if unsupported(sim_dir) or result["score"] >= settings.survey_fidelity_threshold:
        return result
    return _retry(provider, repo, service, tools, examples, sim_dir, calls, ctx, settings, scrub,
                  result)


def _retry(provider, repo, service, tools, examples, sim_dir, calls, ctx, settings, scrub, prev):
    from . import fidelity
    from .simulate import _tool_source

    snapshot = _snapshot(sim_dir)
    write_sim(sim_dir, _generate(provider, repo, service, _tool_source(repo, tools), examples,
                                 _failure_hint(prev)))
    new = fidelity.measure_service(sim_dir, repo, calls, ctx, settings, scrub)
    if new["score"] >= prev["score"]:
        return new
    _restore(sim_dir, snapshot)  # keep the better (previous) simulator
    return prev


def _has_files(sim_dir: Path) -> bool:
    return any((sim_dir / n).is_file()
               for n in ("schema.sql", "collections.json", "UNSUPPORTED.md"))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m touchstone.survey.db_sim <sim_dir>")
    materialize(sys.argv[1])


if __name__ == "__main__":
    main()
