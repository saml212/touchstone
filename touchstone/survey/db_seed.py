"""Deterministic seed sources for a `kind == "db"` simulator — never the model's own data.

A db simulator's `state.db` is built from a real source, in priority order:

1. **data files in the repo** — the files the code loads as its store (JSON dict-of-docs, JSON list,
   CSV, SQLite). Each is copied — scrubbed — into `simulators/<name>/source/`, and `materialize`
   rebuilds `state.db` from those files with no model involved.
2. **recorded reads** — the documents the tools returned, keyed by id, written into `source/`.

`materialize_source` reads the already-scrubbed `source/` files (it runs in the image with no
scrubber and no raw data present), inferring each file's shape from its extension and contents:
`.json` dict → a document table, `.json` list → a document table keyed by each item's `id`, `.csv` →
a relational TEXT table, a SQLite file → copied verbatim as `state.db`. The table name is the file
stem, so the customer's `data["orders"]` maps to the `orders` table.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sqlite3
from pathlib import Path

from .minted_ids import scalar_leaves
from .writes import atomic_write, atomic_write_json

SOURCE = "source"
_SQLITE_EXT = {".db", ".sqlite", ".sqlite3"}
_ID_KEYS = ("id", "_id")


def source_dir(sim_dir: str | Path) -> Path:
    return Path(sim_dir) / SOURCE


def has_source(sim_dir: str | Path) -> bool:
    d = source_dir(sim_dir)
    return d.is_dir() and any(d.iterdir())


# ---- seeding (parent side, with the shared scrubber) ------------------------


_BRACE = re.compile(r"\{([^{}]*)\}")


def data_files_for(repo: Path, service: dict) -> list[str]:
    """The repo-relative data files to seed from: the map's explicit `data_files`, else derived from
    a `base_url_default` naming local files (one path, or a `{a,b,c}` brace list) that exist in the
    repo. A URL default yields nothing (there are no files to copy)."""
    explicit = [f for f in (service.get("data_files") or []) if (repo / f).is_file()]
    if explicit:
        return explicit
    default = service.get("base_url_default") or ""
    if "://" in default:
        return []
    return [c for c in _expand_braces(default) if (repo / c).is_file()]


def _expand_braces(pattern: str) -> list[str]:
    match = _BRACE.search(pattern)
    if not match:
        return [pattern]
    prefix, suffix = pattern[:match.start()], pattern[match.end():]
    return [f"{prefix}{option.strip()}{suffix}" for option in match.group(1).split(",")]


def copy_data_files(sim_dir: Path, repo: Path, data_files: list[str], scrub) -> list[str]:
    """Copy each repo data file into `source/`, scrubbed, and return the names copied. A missing
    file is skipped. Scrubbing runs on the file's CONTENTS so names/emails never enter source/."""
    dest = source_dir(sim_dir)
    _reset(dest)
    copied: list[str] = []
    for rel in data_files:
        src = repo / rel
        if src.is_file() and _copy_one(src, dest / src.name, scrub):
            copied.append(src.name)
    return copied


def _copy_one(src: Path, dst: Path, scrub) -> bool:
    ext = src.suffix.lower()
    if ext in _SQLITE_EXT:
        shutil.copy2(src, dst)
        _scrub_sqlite(dst, scrub)
        return True
    if ext == ".csv":
        atomic_write(dst, _scrub_csv_text(src.read_text(encoding="utf-8", errors="replace"), scrub))
        return True
    if ext == ".json":
        atomic_write_json(dst, scrub.scrub(json.loads(src.read_text(encoding="utf-8"))))
        return True
    return False


def _scrub_csv_text(text: str, scrub) -> str:
    return scrub.text(text)


def _scrub_sqlite(db: Path, scrub) -> None:
    conn = sqlite3.connect(str(db))
    try:
        for (table,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            _scrub_table(conn, table, scrub)
        conn.commit()
    finally:
        conn.close()


def _scrub_table(conn, table: str, scrub) -> None:
    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    rows = [dict(zip(cols, r, strict=False))
            for r in conn.execute(f'SELECT * FROM "{table}"').fetchall()]
    conn.execute(f'DELETE FROM "{table}"')
    for row in rows:
        scrubbed = {c: (scrub.text(v) if isinstance(v, str) else v) for c, v in row.items()}
        placeholders = ", ".join("?" for _ in cols)
        conn.execute(f'INSERT INTO "{table}" ({", ".join(cols)}) VALUES ({placeholders})',
                     [scrubbed[c] for c in cols])


def seed_recorded(sim_dir: Path, calls, scrub, collection: str) -> bool:
    """Seed `source/<collection>.json` from the documents the tools returned (scrubbed), keyed by
    the id the call carried or the result's own id. True when at least one document was found."""
    docs: dict[str, object] = {}
    for call in calls:
        doc = call.output
        if isinstance(doc, dict):
            key = _recorded_id(call)
            if key is not None:
                docs[str(key)] = scrub.scrub(doc)
    if not docs:
        return False
    dest = source_dir(sim_dir)
    _reset(dest)
    atomic_write_json(dest / f"{collection}.json", {scrub.text(k): v for k, v in docs.items()})
    return True


def _recorded_id(call):
    doc = call.output if isinstance(call.output, dict) else {}
    for key in _ID_KEYS:
        if doc.get(key) is not None:
            return doc[key]
    args = call.arguments if isinstance(call.arguments, dict) else {}
    for key, val in args.items():
        if key in _ID_KEYS or key.endswith("_id"):
            if isinstance(val, str | int) and val in scalar_leaves(call.output):
                return val
    return next((v for v in args.values() if isinstance(v, str | int)), None)


def _reset(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)


# ---- materialize (image side, no scrubber) ---------------------------------


def materialize_source(sim_dir: str | Path, db: Path) -> Path:
    """Build `state.db` from the already-scrubbed `source/` files. A SQLite source is copied whole;
    otherwise each file becomes one table (document or relational) inferred from its contents."""
    files = sorted(source_dir(sim_dir).iterdir())
    sqlite_src = next((f for f in files if f.suffix.lower() in _SQLITE_EXT), None)
    if sqlite_src:
        shutil.copy2(sqlite_src, db)
        return db
    conn = sqlite3.connect(str(db))
    try:
        for path in files:
            _load_file(conn, path)
        conn.commit()
    finally:
        conn.close()
    return db


def materialize_files(sim_dir: str | Path, db: Path) -> Path:
    """Build state.db from a provider-written seed: `collections.json` (document tables) or
    `schema.sql` + `seed.json` (relational). This is the path-c fallback shape."""
    sim_dir = Path(sim_dir)
    conn = sqlite3.connect(str(db))
    try:
        collections = sim_dir / "collections.json"
        if collections.is_file():
            for name, docs in json.loads(collections.read_text(encoding="utf-8")).items():
                _load_json(conn, name, docs)
        else:
            _materialize_relational(conn, sim_dir)
        conn.commit()
    finally:
        conn.close()
    return db


def _materialize_relational(conn, sim_dir: Path) -> None:
    schema = sim_dir / "schema.sql"
    if not schema.is_file():
        return
    conn.executescript(schema.read_text(encoding="utf-8"))
    seed_path = sim_dir / "seed.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else {}
    for table, rows in seed.items():
        for row in rows or []:
            _insert_row(conn, table, row)


def _insert_row(conn, table: str, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    values = [_scalar(row[c]) for c in cols]
    conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", values)


def _scalar(value):
    """SQLite stores scalars; a nested object/array is stored as its JSON text."""
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _load_file(conn, path: Path) -> None:
    if path.suffix.lower() == ".json":
        _load_json(conn, path.stem, json.loads(path.read_text(encoding="utf-8")))
    elif path.suffix.lower() == ".csv":
        _load_csv(conn, path.stem, path.read_text(encoding="utf-8"))


def _load_json(conn, table: str, data) -> None:
    conn.execute(f'CREATE TABLE "{table}" (id TEXT PRIMARY KEY, doc TEXT)')
    for doc_id, doc in _doc_items(data):
        conn.execute(f'INSERT OR REPLACE INTO "{table}" (id, doc) VALUES (?, ?)',
                     [str(doc_id), json.dumps(doc, ensure_ascii=False, sort_keys=True)])


def _doc_items(data):
    if isinstance(data, dict):
        return list(data.items())
    return [(_list_id(d, i), d) for i, d in enumerate(data or [])]


def _list_id(doc, index):
    if isinstance(doc, dict):
        for key in _ID_KEYS:
            if doc.get(key) is not None:
                return doc[key]
    return index


def _load_csv(conn, table: str, text: str) -> None:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return
    header = rows[0]
    cols = ", ".join(f'"{c}" TEXT' for c in header)
    conn.execute(f'CREATE TABLE "{table}" ({cols})')
    placeholders = ", ".join("?" for _ in header)
    for row in rows[1:]:
        padded = (row + [None] * len(header))[:len(header)]
        conn.execute(f'INSERT INTO "{table}" VALUES ({placeholders})', padded)
