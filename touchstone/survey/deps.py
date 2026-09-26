"""Resolve a snapshotted repo's own runtime dependencies for the Harbor environment image.

Sources are tried in order (uv.lock, pyproject `[project].dependencies`, requirements*.txt,
setup.cfg `install_requires`, setup.py `install_requires`) so a repo that declares its deps any of
these ways still gets a real requirements.txt and a runnable gate — not `deps_ok=False`. setup.py is
read with `ast`; it is never executed. `is_package` decides whether the snapshot is itself an
installable distribution (so the image installs it editable).
"""

from __future__ import annotations

import ast
import configparser
import re
import tomllib
from pathlib import Path

_VCS_RE = re.compile(r"@\s*(git|http|file)|://|git\+|^-e\b|^\.{0,2}/")


def pyproject_deps(repo: Path) -> list[str] | None:
    path = repo / "pyproject.toml"
    if not path.is_file():
        return None
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies")
    return list(deps) if isinstance(deps, list) else None


def _requirements_deps(repo: Path) -> list[str] | None:
    """Merge every requirements*.txt (comments stripped) — a repo may split runtime/dev files."""
    paths = sorted(repo.glob("requirements*.txt"))
    if not paths:
        return None
    lines = []
    for path in paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                lines.append(line)
    return lines or None


def _uv_root(packages: list[dict]) -> dict | None:
    for pkg in packages:
        src = pkg.get("source", {})
        if src.get("editable") == "." or src.get("virtual") == "." or "workspace" in src:
            return pkg
    return None


def _uv_lock_deps(repo: Path) -> list[str] | None:
    """The repo's own direct deps from uv.lock, pinned to the resolved version."""
    path = repo / "uv.lock"
    if not path.is_file():
        return None
    packages = tomllib.loads(path.read_text(encoding="utf-8")).get("package", [])
    root = _uv_root(packages)
    if root is None:
        return None
    versions = {p.get("name"): p.get("version") for p in packages}
    deps = []
    for dep in root.get("dependencies", []):
        name = dep.get("name")
        if name:
            ver = versions.get(name)
            deps.append(f"{name}=={ver}" if ver else name)
    return deps or None


def _setup_cfg_deps(repo: Path) -> list[str] | None:
    path = repo / "setup.cfg"
    if not path.is_file():
        return None
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if not parser.has_option("options", "install_requires"):
        return None
    deps = [line.strip() for line in parser.get("options", "install_requires").splitlines()
            if line.strip()]
    return deps or None


def _is_setup_call(node: ast.Call) -> bool:
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "setup") or (
        isinstance(func, ast.Attribute) and func.attr == "setup")


def _kw_list(call: ast.Call, name: str) -> list[str] | None:
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.List):
            return [el.value for el in kw.value.elts
                    if isinstance(el, ast.Constant) and isinstance(el.value, str)]
    return None


def _setup_py_deps(repo: Path) -> list[str] | None:
    """`install_requires` read statically with `ast` — setup.py is never executed."""
    path = repo / "setup.py"
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_setup_call(node):
            deps = _kw_list(node, "install_requires")
            if deps:
                return deps
    return None


_SOURCES = (_uv_lock_deps, pyproject_deps, _requirements_deps, _setup_cfg_deps, _setup_py_deps)


def resolve_deps(repo: Path) -> tuple[list[str] | None, str]:
    """Resolvable (non-VCS, non-URL, non-path) runtime deps, or (None, reason) when none exist."""
    for source in _SOURCES:
        deps = source(repo)
        if deps is not None:
            return [d for d in deps if not _VCS_RE.search(d)], "ok"
    return None, ("no dependency source (uv.lock, pyproject.toml, requirements*.txt, "
                  "setup.cfg, setup.py)")


def is_package(repo: Path) -> bool:
    """The repo builds an installable distribution (setup.py/setup.cfg, or a build-system)."""
    if (repo / "setup.py").is_file() or (repo / "setup.cfg").is_file():
        return True
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        return "build-system" in tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return False
