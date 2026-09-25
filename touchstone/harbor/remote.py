"""Stage the touchstone package for shipping to a remote Harbor host.

When `harbor run` executes over SSH and the agent is a touchstone custom agent, the package has to
be importable on the host. `harbor/run.py` rsyncs a `touchstone-src` dir there and runs harbor with
`uvx --with <dir>`; this module produces that local dir from either install layout — a source
checkout (which already has a `pyproject.toml`) or a wheel install (which does not, so a minimal
hatchling pyproject is generated). Kept apart from `run.py` so the SSH/rsync orchestration reads on
its own and each file stays within the size budget.
"""

from __future__ import annotations

import importlib.metadata as metadata
import shutil
from pathlib import Path

import touchstone

_DIST = "touchstone-bench"  # the distribution that ships the touchstone package


def _src_paths() -> tuple[Path, Path]:
    """(directory that would hold pyproject.toml, the touchstone package dir). For a source checkout
    the first has a pyproject; for a wheel install it is site-packages and has none."""
    package_dir = Path(touchstone.__file__).resolve().parent
    return package_dir.parent, package_dir


def _runtime_deps() -> list[str]:
    """The distribution's runtime dependencies (extras dropped), so `uvx --with <dir>` resolves."""
    out: list[str] = []
    try:
        reqs = metadata.requires(_DIST) or []
    except metadata.PackageNotFoundError:
        return out
    for req in reqs:
        if "extra ==" not in req:
            out.append(req.split(";")[0].strip())
    return out


def _dist_version() -> str:
    try:
        return metadata.version(_DIST)
    except metadata.PackageNotFoundError:
        return "0.0.0"


def generated_pyproject() -> str:
    """A minimal hatchling pyproject so `uvx --with <dir>` can build touchstone from a wheel install
    (which ships no pyproject). Lists the running version, the runtime deps, and the package."""
    deps = "".join(f'    "{d}",\n' for d in _runtime_deps())
    return ('[project]\n'
            f'name = "{_DIST}"\n'
            f'version = "{_dist_version()}"\n'
            f'dependencies = [\n{deps}]\n\n'
            '[build-system]\n'
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            '[tool.hatch.build.targets.wheel]\n'
            'packages = ["touchstone"]\n')


def _stage_src(stage: Path) -> Path:
    """The local dir to rsync as `touchstone-src`: the checkout root when it already has a
    pyproject, else a staged dir holding only the touchstone package plus a generated pyproject."""
    source_root, package_dir = _src_paths()
    if (source_root / "pyproject.toml").is_file():
        return source_root
    shutil.copytree(package_dir, stage / "touchstone", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "pyproject.toml").write_text(generated_pyproject(), encoding="utf-8")
    return stage
