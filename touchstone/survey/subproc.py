"""Make the CLI's own touchstone importable in a survey subprocess — without shadowing its SDKs.

The replay and packaged-adapter subprocesses run in the customer's interpreter/venv (so their real
tools hit the SDK versions they run in production). But that venv may pin a DIFFERENT, older
touchstone — and then `import touchstone.survey.replay` loads stale code and the survey breaks
("No module named 'touchstone.survey'" when the pin predates the package).

The fix must expose ONLY the touchstone package, not the whole install: for a wheel/uv-tool install
`Path(touchstone.__file__).parents[1]` is the entire site-packages, so putting it on PYTHONPATH
would also shadow the customer's own SDKs (e.g. their pydantic loading the CLI's `pydantic_core`,
which fails). So point PYTHONPATH at a small dir that holds only a `touchstone` symlink to the
running package; `import touchstone` resolves to the CLI's code while every other import still
falls through to the customer's venv.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import touchstone


def _package_dir() -> Path:
    return Path(touchstone.__file__).resolve().parent


def cli_import_dir() -> str:
    """A directory exposing only the running touchstone package (by symlink) for a subprocess's
    PYTHONPATH. Falls back to the package's parent when symlinks are unavailable — safe for a source
    checkout, whose parent holds no sibling SDK packages to shadow."""
    pkg = _package_dir()
    key = hashlib.sha1(str(pkg).encode()).hexdigest()[:12]
    link_dir = Path(tempfile.gettempdir()) / f"touchstone-clipath-{key}"
    link = link_dir / "touchstone"
    try:
        link_dir.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(pkg, target_is_directory=True)
        return str(link_dir)
    except OSError:
        return str(pkg.parent)


def with_cli_path(env: dict) -> dict:
    """`env` with the CLI's touchstone-only dir prepended onto PYTHONPATH (returns a new dict)."""
    parts = [cli_import_dir(), env.get("PYTHONPATH", "")]
    out = dict(env)
    out["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
    return out
