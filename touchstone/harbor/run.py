"""Run `harbor run` over a dataset or a single task, locally or on a remote Docker host.

Harbor bind-mounts local directories, so a remote Docker daemon alone is not enough: when the local
machine has no Docker daemon and `settings.harbor_host` is set, this rsyncs the dataset to the host,
runs Harbor there over SSH, and rsyncs the job directory back. Datasets always run as the implicit
dataset (`harbor run -p <root>/tasks`); `dataset.toml` is metadata only and is never passed to `-p`.
The exact command run is printed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..config import Settings, load_settings

# A generic login PATH so a non-interactive SSH shell finds harbor and docker on common hosts.
_REMOTE_PATH = 'export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"'


def _run_path(path: Path) -> Path:
    """The path passed to `harbor run -p`: a task dir as-is, a dataset root -> its tasks/."""
    if (path / "task.toml").is_file():
        return path
    if (path / "tasks").is_dir():
        return path / "tasks"
    return path


def _has_docker() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except OSError:
        return False


def _call(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _job_dirs(jobs_dir: Path) -> set[str]:
    return {d.name for d in jobs_dir.iterdir() if d.is_dir()} if jobs_dir.is_dir() else set()


def _newest_job(jobs_dir: Path, before: set[str]) -> Path:
    new = [d for d in jobs_dir.iterdir() if d.is_dir() and d.name not in before]
    pool = new or [d for d in jobs_dir.iterdir() if d.is_dir()]
    if not pool:
        raise FileNotFoundError(f"no job directory was created under {jobs_dir}")
    return max(pool, key=lambda d: d.stat().st_mtime)


def _is_custom_agent(agent: str) -> bool:
    """A `module.path:Class` import path (needs touchstone in harbor's env), not a built-in name."""
    return ":" in agent


def _repo_root() -> Path:
    """The touchstone repo root, so a custom agent can be loaded via `uvx --with <root>`."""
    return Path(__file__).resolve().parents[2]


def _harbor_cmd(run_path: str, agent: str, model: str | None, jobs_dir: str, n_concurrent: int,
                extra_args: list[str], with_path: str | None = None) -> list[str]:
    prefix = ["uvx", "--from", "harbor", "--with", with_path, "harbor", "run"] if with_path \
        else ["harbor", "run"]
    cmd = prefix + ["-p", run_path, "-a", agent, "-o", jobs_dir, "-n", str(n_concurrent), "-y"]
    if model:
        cmd += ["-m", model]
    return cmd + extra_args


def _run_local(path: Path, agent: str, model: str | None, jobs_dir: Path,
               n_concurrent: int, extra_args: list[str]) -> Path:
    # A custom agent must already be importable in the local `harbor` env; built-ins always work.
    jobs_dir.mkdir(parents=True, exist_ok=True)
    before = _job_dirs(jobs_dir)
    cmd = _harbor_cmd(str(_run_path(path).resolve()), agent, model, str(jobs_dir.resolve()),
                      n_concurrent, extra_args)
    _call(cmd)
    return _newest_job(jobs_dir, before)


def _run_remote(path: Path, agent: str, model: str | None, jobs_dir: Path, n_concurrent: int,
                extra_args: list[str], settings: Settings) -> Path:
    host, remote_root = settings.harbor_host, settings.harbor_remote_root
    sync_root = path.parent if path.name == "tasks" else path
    rel_run = _run_path(path).relative_to(sync_root).as_posix() or "."
    remote_path = f"{remote_root}/{sync_root.name}"

    _call(["rsync", "-az", "--delete", f"{sync_root}/", f"{host}:{remote_path}/"])
    with_path = None
    if _is_custom_agent(agent):  # ship touchstone so harbor can import the custom agent
        with_path = f"{remote_root}/touchstone"
        _call(["rsync", "-az", "--delete", f"{_repo_root()}/", f"{host}:{with_path}/"])
    remote_cmd = " ".join(
        _harbor_cmd(rel_run, agent, model, "jobs", n_concurrent, extra_args, with_path))
    _call(["ssh", host, f"{_REMOTE_PATH}; cd {remote_path} && {remote_cmd}"])

    jobs_dir.mkdir(parents=True, exist_ok=True)
    before = _job_dirs(jobs_dir)
    _call(["rsync", "-az", f"{host}:{remote_path}/jobs/", f"{jobs_dir}/"])
    return _newest_job(jobs_dir, before)


def run(path: str | Path, agent: str, *, model: str | None = None, jobs_dir: str | Path = "jobs",
        n_concurrent: int = 4, extra_args: list[str] | None = None,
        settings: Settings | None = None) -> Path:
    """Run Harbor over `path` (a task dir or dataset root) and return the created job directory."""
    path, jobs_dir = Path(path), Path(jobs_dir)
    extra_args = extra_args or []
    settings = settings or load_settings()
    if settings.harbor_host and not _has_docker():
        return _run_remote(path, agent, model, jobs_dir, n_concurrent, extra_args, settings)
    return _run_local(path, agent, model, jobs_dir, n_concurrent, extra_args)
