"""Make the CLI's own touchstone importable in a survey subprocess.

The replay and packaged-adapter subprocesses run in the customer's interpreter/venv (so their real
tools hit the SDK versions they run in production). But that venv may pin a DIFFERENT, older
touchstone — and then `import touchstone.survey.replay` loads stale code and the survey breaks
("No module named 'touchstone.survey'" when the pin predates the package). Prepend PYTHONPATH with
the directory holding the running CLI's touchstone package so the survey always uses its own code;
the customer's venv still supplies their own SDKs. Same idea as vendoring touchstone into the image.
"""

from __future__ import annotations

import os
from pathlib import Path

import touchstone


def cli_src_dir() -> str:
    """The directory CONTAINING the running touchstone package (so `import touchstone` finds it)."""
    return str(Path(touchstone.__file__).resolve().parents[1])


def with_cli_path(env: dict) -> dict:
    """`env` with the CLI's touchstone dir prepended onto PYTHONPATH (returns a new dict)."""
    parts = [cli_src_dir(), env.get("PYTHONPATH", "")]
    out = dict(env)
    out["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
    return out
