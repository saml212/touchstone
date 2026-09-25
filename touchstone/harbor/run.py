"""Run `harbor run` over a dataset or a single task, locally or on a remote Docker host.

Harbor bind-mounts local directories, so a remote Docker daemon alone is not enough: when the local
machine has no Docker daemon and `settings.harbor_host` is set, this rsyncs the dataset to the host,
runs Harbor there over SSH, and rsyncs the job directory back. Datasets always run as the implicit
dataset (`harbor run -p <root>/tasks`); `dataset.toml` is metadata only and is never passed to `-p`.
The exact command run is printed.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from ..config import Settings, load_settings
from ..llm.keychain import secret
from . import keys

# API-key env var -> the Settings attribute holding its Keychain service name.
_KEYCHAIN_ATTR = {"OPENAI_API_KEY": "keychain_openai", "ANTHROPIC_API_KEY": "keychain_anthropic"}

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


def _call(cmd: list[str], env: dict | None = None, stdin_data: str | None = None) -> None:
    """Run `cmd`, echo its combined output, and on failure raise a RuntimeError whose message is the
    last lines of that output, so a gate failure records what Harbor said, not just an exit code.

    `stdin_data` is fed on stdin (never argv, never printed) — the channel that forwards an API key
    to the remote shell. `env` replaces the child environment when given."""
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, input=stdin_data)
    output = (proc.stdout or "") + (proc.stderr or "")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-30:])
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n{tail}")


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


def _provider_key(model: str | None, settings: Settings) -> tuple[str, str] | None:
    """Resolve `(env var, value)` for the model provider's API key, or None when none is needed or
    found. Read on the laptop (env or Keychain); forwarded to harbor over stdin, never argv."""
    var = keys.provider_key_var(model)
    if not var:
        return None
    value = secret(var, settings.keychain_service(getattr(settings, _KEYCHAIN_ATTR[var])))
    return (var, value) if value else None


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


def _run_local(path: Path, agent: str, model: str | None, jobs_dir: Path, n_concurrent: int,
               extra_args: list[str], key: tuple[str, str] | None = None) -> Path:
    # A custom agent must already be importable in the local `harbor` env; built-ins always work.
    jobs_dir.mkdir(parents=True, exist_ok=True)
    before = _job_dirs(jobs_dir)
    cmd = _harbor_cmd(str(_run_path(path).resolve()), agent, model, str(jobs_dir.resolve()),
                      n_concurrent, extra_args)
    _call(cmd, env={**os.environ, key[0]: key[1]} if key else None)
    return _newest_job(jobs_dir, before)


def _remote_dataset_path(remote_root: str, sync_root: Path) -> str:
    """A per-checkout remote path so two customers' `touchstone/` roots never collide."""
    digest = hashlib.sha1(str(sync_root.resolve()).encode()).hexdigest()[:8]
    return f"{remote_root}/datasets/{sync_root.name}-{digest}"


def _remote_key_prefix(key: tuple[str, str] | None) -> tuple[str, str | None]:
    """Shell prefix that reads the key from stdin into the provider's env var, plus the stdin data.
    The value never appears in the command string (only `$TS_KEY`, expanded on the remote host)."""
    if not key:
        return "", None
    return f'read -r TS_KEY; export {key[0]}="$TS_KEY"; ', key[1] + "\n"


def _run_remote(path: Path, agent: str, model: str | None, jobs_dir: Path, n_concurrent: int,
                extra_args: list[str], settings: Settings,
                key: tuple[str, str] | None = None) -> Path:
    host, remote_root = settings.harbor_host, settings.harbor_remote_root
    sync_root = path.parent if path.name == "tasks" else path
    rel_run = _run_path(path).relative_to(sync_root).as_posix() or "."
    remote_path = _remote_dataset_path(remote_root, sync_root)

    # --exclude jobs so a --delete push never wipes job dirs the host still holds.
    _call(["rsync", "-az", "--delete", "--exclude", "jobs",
           f"{sync_root}/", f"{host}:{remote_path}/"])
    with_path = None
    if _is_custom_agent(agent):  # ship the touchstone repo so harbor can import the custom agent
        with_path = f"{remote_root}/touchstone-src"
        _call(["rsync", "-az", "--delete", f"{_repo_root()}/", f"{host}:{with_path}/"])
    remote_cmd = " ".join(
        _harbor_cmd(rel_run, agent, model, "jobs", n_concurrent, extra_args, with_path))
    prefix, stdin_data = _remote_key_prefix(key)
    _call(["ssh", host, f"{prefix}{_REMOTE_PATH}; cd {remote_path} && {remote_cmd}"],
          stdin_data=stdin_data)

    jobs_dir.mkdir(parents=True, exist_ok=True)
    before = _job_dirs(jobs_dir)
    _call(["rsync", "-az", f"{host}:{remote_path}/jobs/", f"{jobs_dir}/"])
    return _newest_job(jobs_dir, before)


def _build_remote(context_dir: Path, tag: str, settings: Settings) -> None:
    host, remote_root = settings.harbor_host, settings.harbor_remote_root
    remote_ctx = f"{remote_root}/env-build/{tag.replace(':', '-')}"
    _call(["rsync", "-az", "--delete", f"{context_dir}/", f"{host}:{remote_ctx}/"])
    _call(["ssh", host, f"{_REMOTE_PATH}; cd {remote_ctx} && docker build -t {tag} ."])


def build_image(context_dir: str | Path, tag: str, settings: Settings | None = None) -> None:
    """Build the environment image `tag` from `context_dir`. Builds on the SSH host when there is no
    local Docker daemon — the same host `harbor run` uses, so its daemon has the image."""
    context_dir = Path(context_dir)
    settings = settings or load_settings()
    if settings.harbor_host and not _has_docker():
        _build_remote(context_dir, tag, settings)
    else:
        _call(["docker", "build", "-t", tag, str(context_dir)])


def run(path: str | Path, agent: str, *, model: str | None = None, jobs_dir: str | Path = "jobs",
        n_concurrent: int = 4, extra_args: list[str] | None = None,
        settings: Settings | None = None) -> Path:
    """Run Harbor over `path` (a task dir or dataset root) and return the created job directory."""
    path, jobs_dir = Path(path), Path(jobs_dir)
    extra_args = extra_args or []
    settings = settings or load_settings()
    key = _provider_key(model, settings) if _is_custom_agent(agent) else None
    if settings.harbor_host and not _has_docker():
        return _run_remote(path, agent, model, jobs_dir, n_concurrent, extra_args, settings, key)
    return _run_local(path, agent, model, jobs_dir, n_concurrent, extra_args, key)
