"""Helpers shared by every stage that wires a `kind == "db"` service.

A db service's tools read and write a database in-process (a SQL driver/ORM, or an in-memory store
loaded from JSON documents) instead of calling an HTTP API. There is no server and no port: the
tools are pointed at a materialized `simulators/<name>/state.db` through one environment variable —
the one the map found (`base_url_env`), else a synthesized `TOUCHSTONE_DB_<NAME>`. The value is the
database file's path, or `sqlite:///<path>` when the code reads a URL.
"""

from __future__ import annotations

import re
from pathlib import Path

CONTAINER_SIM_ROOT = "/app/simulators"


def is_db(service: dict | None) -> bool:
    return bool(service) and service.get("kind") == "db"


def _synth(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", name or "db").strip("_").upper() or "DB"
    return f"TOUCHSTONE_DB_{slug}"


def env_name(service: dict) -> str:
    """The env var the db tools read for the database: the mapped one, else a synthesized name."""
    return service.get("base_url_env") or _synth(service.get("name", "db"))


def is_url(service: dict) -> bool:
    """The code reads a database URL (e.g. postgresql://...) rather than a bare file path."""
    return "://" in (service.get("base_url_default") or "")


def value_for(db_path: str | Path, *, url: bool) -> str:
    """The env value for a state.db path: the bare path, or sqlite:///<path> for a URL tool."""
    p = str(db_path)
    return f"sqlite:///{p}" if url else p


def env_value(db_path: str | Path, service: dict) -> str:
    """The value env_name is set to: the state.db path, or sqlite:///<path> for a URL tool."""
    return value_for(db_path, url=is_url(service))


def container_db_path(name: str) -> str:
    """Where state.db lives inside the environment image."""
    return f"{CONTAINER_SIM_ROOT}/{name}/state.db"


def state_db(sim_dir: str | Path) -> Path:
    return Path(sim_dir) / "state.db"
