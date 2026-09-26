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
from pathlib import Path

import touchstone

from .deps import is_package, pyproject_deps, resolve_deps
from .simulate import crossing_services, service_host
from .writes import atomic_write

_SKIP_DIRS = {".git", ".touchstone", "touchstone", ".venv", "node_modules", "__pycache__"}
_SECRET_RE = re.compile(r"(^\.env$|^\.env\.|\.pem$|\.key$|^id_rsa|secret)", re.IGNORECASE)
_SIM_RUNTIME = ["fastapi", "uvicorn", "pydantic"]
# The ACP server (touchstone.harbor.acp_server) runs the loop inside the sandbox on a simulated-user
# trial; it needs the acp package and the providers' httpx client. Baking them here means the image
# carries them at build time, so no fragile runtime `pip install` (which fails on an
# externally-managed Python, offline, or without uv) can leave a trial silently unscored.
_ACP_RUNTIME = ["agent-client-protocol", "httpx"]
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

# When the snapshotted repo is itself an installable package, install it (deps already handled by
# requirements.txt) so its own modules import the way they do in production, not just off the copied
# tree. `repo/` is copied into /app, so the project root inside the image is /app.
_EDITABLE_INSTALL = "RUN uv pip install --system --no-deps -e /app\n"

# The one copy of "start a simulator" in the image. `start.sh <name> <port>` starts the sim in the
# background (nohup survives the exec), waits for /__health, and POSTs /__reset. It exports nothing:
# the caller sets the service's base-url env. Both solve.sh (oracle) and TouchstoneAgent.setup()
# call `bash /app/simulators/start.sh <name> <port>`.
START_SH = """\
#!/bin/bash
set -u
name="$1"
port="${2:-0}"
sim="/app/simulators/$name"
if [ ! -f "$sim/app.py" ]; then
  # A db service has no server: (re)materialize state.db from schema.sql/seed.json (or
  # collections.json) and return. The net shim is not involved.
  python -m touchstone.survey.db_sim "$sim"
  exit 0
fi
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


# Auto-imported at interpreter startup (this dir is on PYTHONPATH in the image). When
# TOUCHSTONE_SIMULATORS names host->simulator base URLs, install the net shim so a tool that
# hard-codes its host is rewritten to the loopback simulator. Loads netshim by file path so the
# whole touchstone package is not imported into every python process in the container.
SITECUSTOMIZE = '''\
"""Touchstone environment: install the net shim from TOUCHSTONE_SIMULATORS when set."""
import os

if os.environ.get("TOUCHSTONE_SIMULATORS"):
    import importlib.util
    import pathlib

    _p = pathlib.Path(__file__).resolve().parent / "touchstone" / "survey" / "netshim.py"
    if _p.exists():
        _spec = importlib.util.spec_from_file_location("_touchstone_netshim", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.install_from_env()
'''


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


# The container path the replica dispatch runs invoke.py from (copied here when the survey generated
# and adapter-passed one). On PYTHONPATH root; loaded by file path, so it need not be a package.
IMAGE_INVOKE = "/app/_touchstone/invoke.py"


def _copy_invoke(out: Path, dest: Path) -> None:
    """Copy the generated agent/invoke.py into the image so the replica dispatch can call tools by
    name inside the sandbox. Absent when the survey wrote none (or its adapter check failed)."""
    src = out / "agent" / "invoke.py"
    if src.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / "invoke.py")


def _dockerfile(repo: Path) -> str:
    return DOCKERFILE + _EDITABLE_INSTALL if is_package(repo) else DOCKERFILE


def _touchstone_deps() -> list[str]:
    """touchstone's own runtime deps, so `import touchstone` resolves inside the image (it is
    vendored on PYTHONPATH, not pip-installed, so its declared deps are not pulled otherwise)."""
    try:
        reqs = importlib.metadata.requires("touchstone-bench")
    except importlib.metadata.PackageNotFoundError:
        reqs = None
    if not reqs:  # running from source without installed metadata: read this repo's pyproject
        reqs = pyproject_deps(Path(touchstone.__file__).resolve().parent.parent) or []
    return [r for r in reqs if ";" not in r]  # drop extras / environment markers


def _requirements_text(deps: list[str]) -> str:
    merged = list(dict.fromkeys([*deps, *_SIM_RUNTIME, *_ACP_RUNTIME, *_touchstone_deps()]))
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


def _ports(map_data: dict) -> dict:
    """Per-crossing-service wiring. A db service gets no port/host; its env var is the mapped one
    or a synthesized TOUCHSTONE_DB_<NAME>, and db_urls says whether its value is a URL."""
    from . import db_service

    ports, base_url_envs, hosts, kinds, db_urls = {}, {}, {}, {}, {}
    next_port = 8000
    for service in crossing_services(map_data):
        name = service["name"]
        kinds[name] = service.get("kind")
        if db_service.is_db(service):
            base_url_envs[name] = db_service.env_name(service)
            hosts[name], db_urls[name] = None, db_service.is_url(service)
        else:
            ports[name] = next_port
            next_port += 1
            base_url_envs[name] = service.get("base_url_env")
            hosts[name] = service_host(service)
    return {"ports": ports, "base_url_envs": base_url_envs, "hosts": hosts,
            "kinds": kinds, "db_urls": db_urls, "services": sorted(base_url_envs)}


def build_environment(repo: Path, map_data: dict, out: Path, force: bool = False) -> dict:
    """Write touchstone/environment/ and return the image tag, ports, and dependency status."""
    env_dir = out / "environment"
    deps, reason = resolve_deps(repo)
    if not (env_dir / "Dockerfile").exists() or force:
        if env_dir.exists():
            shutil.rmtree(env_dir)
        (env_dir / "repo").mkdir(parents=True, exist_ok=True)
        _copy_repo(repo, env_dir / "repo", _repo_files(repo))
        _copy_simulators(out / "simulators", env_dir / "simulators")
        atomic_write(env_dir / "simulators" / "start.sh", START_SH)
        _vendor_touchstone(env_dir / "_touchstone")
        _copy_invoke(out, env_dir / "_touchstone")
        atomic_write(env_dir / "_touchstone" / "sitecustomize.py", SITECUSTOMIZE)
        atomic_write(env_dir / "requirements.txt", _requirements_text(deps or []))
        atomic_write(env_dir / "Dockerfile", _dockerfile(repo))
    wiring = _ports(map_data)
    return {"deps_ok": deps is not None, "deps_reason": reason,
            "image_tag": _image_tag(repo.name, env_dir), **wiring}
