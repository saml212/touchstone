"""Snapshot the customer's system into a Harbor environment image.

Writes `touchstone/environment/`: the repo copied read-only (git-tracked files, so `.gitignore` is
honoured; secrets and the touchstone output are skipped), the generated simulators, the touchstone
package vendored onto PYTHONPATH (so the customer's `import touchstone` resolves offline — it is not
on PyPI), a `requirements.txt` of the repo's resolvable dependencies plus the simulator runtime, and
a single generic `Dockerfile`. Harbor overrides the container command with `sleep infinity`, so this
image only has to hold the code; `solution/solve.sh` starts the simulators at run time.

One image serves every task (`task.toml` points at it via `[environment].docker_image`). Ports are
`8000 + index` per network-crossing service. A repo whose dependencies cannot be resolved is flagged
(`deps_ok = False`) so the gate is skipped for it; the tasks are still written.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import touchstone

from .simulate import crossing_services
from .writes import atomic_write

_SKIP_DIRS = {".git", ".touchstone", "touchstone", ".venv", "node_modules", "__pycache__"}
_SECRET_RE = re.compile(r"(^\.env$|^\.env\.|\.pem$|\.key$|^id_rsa|secret)", re.IGNORECASE)
_VCS_RE = re.compile(r"@\s*(git|http|file)|://|git\+|^-e\b|^\.{0,2}/")
_SIM_RUNTIME = ["fastapi", "uvicorn", "pydantic"]
_SIM_SKIP = {"state.db", ".sim.log"}

DOCKERFILE = """\
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/_touchstone
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY repo/ /app/
COPY simulators/ /app/simulators/
COPY _touchstone/ /app/_touchstone/
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system -r /app/requirements.txt
"""

# The one copy of "start a simulator" in the image. `start.sh <name> <port>` starts the sim in the
# background (nohup survives the exec), waits for /__health, and POSTs /__reset. It exports nothing:
# the caller sets the service's base-url env. Both solve.sh (oracle) and TouchstoneAgent.setup()
# call `bash /app/simulators/start.sh <name> <port>`.
START_SH = """\
#!/bin/bash
set -u
name="$1"
port="$2"
poke() {
  python - "$1" "${2:-GET}" <<'PY' 2>/dev/null
import sys, urllib.request as u
u.urlopen(u.Request(sys.argv[1], method=sys.argv[2]), timeout=2)
PY
}
nohup python "/app/simulators/$name/app.py" "$port" >"/tmp/ts-sim-$name.log" 2>&1 &
for _ in $(seq 1 100); do
  poke "http://127.0.0.1:$port/__health" && break
  sleep 0.2
done
poke "http://127.0.0.1:$port/__reset" POST || true
"""


def _is_secret(rel: str) -> bool:
    return any(_SECRET_RE.search(part) for part in Path(rel).parts)


def _tracked_files(repo: Path) -> list[str] | None:
    try:
        proc = subprocess.run(["git", "-C", str(repo), "ls-files"],
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in proc.stdout.splitlines() if line]


def _walk_files(repo: Path) -> list[str]:
    out = []
    for path in repo.rglob("*"):
        rel = path.relative_to(repo)
        if path.is_file() and not any(part in _SKIP_DIRS for part in rel.parts):
            out.append(rel.as_posix())
    return out


def _repo_files(repo: Path) -> list[str]:
    """Files to snapshot: git-tracked when possible (honours .gitignore), else a filtered walk."""
    files = _tracked_files(repo)
    if files is None:
        files = _walk_files(repo)
    return sorted(f for f in files
                  if not _is_secret(f) and not any(p in _SKIP_DIRS for p in Path(f).parts))


def _copy_repo(repo: Path, dest: Path, files: list[str]) -> None:
    for rel in files:
        src = repo / rel
        if not src.is_file():
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def _copy_simulators(sim_root: Path, dest: Path) -> None:
    if not sim_root.is_dir():
        return
    for path in sim_root.rglob("*"):
        if path.is_file() and path.name not in _SIM_SKIP:
            target = dest / path.relative_to(sim_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _vendor_touchstone(dest: Path) -> None:
    pkg = Path(touchstone.__file__).resolve().parent
    shutil.copytree(pkg, dest / "touchstone",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)


def _pyproject_deps(repo: Path) -> list[str] | None:
    path = repo / "pyproject.toml"
    if not path.is_file():
        return None
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies")
    return list(deps) if isinstance(deps, list) else None


def _requirements_deps(repo: Path) -> list[str] | None:
    path = repo / "requirements.txt"
    if not path.is_file():
        return None
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _resolve_deps(repo: Path) -> tuple[list[str] | None, str]:
    """Resolvable (non-VCS, non-URL, non-path) runtime deps, or (None, reason) when none exist."""
    deps = _pyproject_deps(repo)
    if deps is None:
        deps = _requirements_deps(repo)
    if deps is None:
        return None, "no pyproject.toml or requirements.txt"
    keep = [d for d in deps if not _VCS_RE.search(d)]
    return keep, "ok"


def _touchstone_deps() -> list[str]:
    """touchstone's own runtime deps, so `import touchstone` resolves inside the image (it is
    vendored on PYTHONPATH, not pip-installed, so its declared deps are not pulled otherwise)."""
    try:
        reqs = importlib.metadata.requires("touchstone-bench")
    except importlib.metadata.PackageNotFoundError:
        reqs = None
    if not reqs:  # running from source without installed metadata: read this repo's pyproject
        reqs = _pyproject_deps(Path(touchstone.__file__).resolve().parent.parent) or []
    return [r for r in reqs if ";" not in r]  # drop extras / environment markers


def _requirements_text(deps: list[str]) -> str:
    merged = list(dict.fromkeys([*deps, *_SIM_RUNTIME, *_touchstone_deps()]))
    return "\n".join(merged) + "\n"


def _sanitize_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]", "-", name.lower()).strip("-._") or "env"


def _image_tag(repo_name: str, env_dir: Path) -> str:
    digest = hashlib.sha1()
    for path in sorted(env_dir.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(env_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return f"touchstone-env-{_sanitize_tag(repo_name)}:{digest.hexdigest()[:12]}"


def _ports(map_data: dict) -> tuple[dict, dict]:
    ports, base_url_envs = {}, {}
    for i, service in enumerate(crossing_services(map_data)):
        name = service["name"]
        ports[name] = 8000 + i
        base_url_envs[name] = service.get("base_url_env")
    return ports, base_url_envs


def build_environment(repo: Path, map_data: dict, out: Path, force: bool = False) -> dict:
    """Write touchstone/environment/ and return the image tag, ports, and dependency status."""
    env_dir = out / "environment"
    deps, reason = _resolve_deps(repo)
    if not (env_dir / "Dockerfile").exists() or force:
        if env_dir.exists():
            shutil.rmtree(env_dir)
        (env_dir / "repo").mkdir(parents=True, exist_ok=True)
        _copy_repo(repo, env_dir / "repo", _repo_files(repo))
        _copy_simulators(out / "simulators", env_dir / "simulators")
        atomic_write(env_dir / "simulators" / "start.sh", START_SH)
        _vendor_touchstone(env_dir / "_touchstone")
        atomic_write(env_dir / "requirements.txt", _requirements_text(deps or []))
        atomic_write(env_dir / "Dockerfile", DOCKERFILE)
    ports, base_url_envs = _ports(map_data)
    return {"deps_ok": deps is not None, "deps_reason": reason, "ports": ports,
            "base_url_envs": base_url_envs, "image_tag": _image_tag(repo.name, env_dir),
            "services": sorted(ports)}
