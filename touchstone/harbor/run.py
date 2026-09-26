"""Run `harbor run` over a dataset or a single task, locally or on a remote Docker host.

When the local machine has no Docker daemon and `settings.harbor_host` is set, this rsyncs the
dataset to the host, runs Harbor over SSH (`harbor run -p <root>/tasks`), and rsyncs jobs back.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from ..config import Settings, load_settings
from ..ids import new_id
from ..llm.keychain import secret
from . import keys, remote

_KEYCHAIN_ATTR = {"OPENAI_API_KEY": "keychain_openai", "ANTHROPIC_API_KEY": "keychain_anthropic"}

_REMOTE_PATH = 'export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"'

# ssh/rsync options: bound the TCP connect and BatchMode=yes so an unreachable host fails fast.
_SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]


def _ssh_cmd(host: str, remote: str) -> list[str]:
    return ["ssh", *_SSH_OPTS, host, remote]


def _rsync_cmd(args: list[str]) -> list[str]:
    return ["rsync", "-e", "ssh " + " ".join(_SSH_OPTS), *args]


def _run_path(path: Path) -> Path:
    """The path passed to `harbor run -p`: a task dir as-is, a dataset root -> its tasks/."""
    if (path / "task.toml").is_file():
        return path
    if (path / "tasks").is_dir():
        return path / "tasks"
    return path


def docker_daemon() -> tuple[bool, str]:
    """(daemon up?, server version) for the local daemon via `docker info` (3 s timeout)."""
    if not shutil.which("docker"):
        return False, ""
    try:
        proc = subprocess.run(["docker", "info", "-f", "{{.ServerVersion}}"],
                              capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return proc.returncode == 0, proc.stdout.strip()


def remote_docker_daemon(settings: Settings) -> tuple[bool, str]:
    """(daemon up?, server version) on the harbor host — the same `docker info` probe over ssh."""
    cmd = _ssh_cmd(settings.harbor_host, f"{_REMOTE_PATH}; docker info -f '{{{{.ServerVersion}}}}'")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return proc.returncode == 0, proc.stdout.strip()


def _has_docker() -> bool:
    return docker_daemon()[0]


class DockerDaemonError(RuntimeError):
    """No Docker daemon is answering (local or harbor host); carries the one CLI-ready sentence."""


def _require_target(settings: Settings) -> str:
    """'local' or 'remote' — where a Docker-needing run executes, proving that daemon first."""
    if _has_docker():
        return "local"
    if settings.harbor_host and remote_docker_daemon(settings)[0]:
        return "remote"
    where = f"host {settings.harbor_host}" if settings.harbor_host else "local"
    raise DockerDaemonError(
        f"Docker daemon not running on {where}. Start Docker (colima start / Docker Desktop) "
        "or set [harbor] host in touchstone.toml.")


def _exec(cmd: list[str], env: dict | None = None,
          stdin_data: str | None = None) -> tuple[int, str]:
    """Run `cmd`, echo its combined output, and return (returncode, output). `stdin_data` feeds
    stdin (never argv) — how an API key reaches the remote shell; `env` replaces the child env."""
    print("$ " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, input=stdin_data)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"'{cmd[0]}' is not installed or not on PATH — it is required to reach the harbor "
            "host; install it (e.g. `brew install rsync openssh`) and retry") from exc
    output = (proc.stdout or "") + (proc.stderr or "")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return proc.returncode, output


def _call(cmd: list[str], env: dict | None = None, stdin_data: str | None = None) -> None:
    """Run `cmd`, echo output, raise on failure."""
    rc, output = _exec(cmd, env, stdin_data)
    if rc != 0:
        tail = "\n".join(output.splitlines()[-30:])
        raise RuntimeError(f"command failed (exit {rc}): {' '.join(cmd)}\n{tail}")


def _build_call(cmd: list[str], where: str, env: dict | None = None) -> None:
    """Run a docker build, retrying once — a shared Docker host under another job's build can fail
    transiently (seen on the mini; a manual rebuild then succeeded). On the second failure raise
    with the last 8 lines of build output so the reason is in the message, not just an exit code."""
    rc, output = 0, ""
    for attempt in (1, 2):
        rc, output = _exec(cmd, env)
        if rc == 0:
            return
        if attempt == 1:
            print(f"docker build failed on {where} (exit {rc}); retrying once")
    tail = "\n".join(output.splitlines()[-8:])
    raise RuntimeError(f"docker build failed on {where} (exit {rc}):\n{tail}")


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
    """`(env var, value)` for the provider's API key, else None; sent to harbor over stdin."""
    var = keys.provider_key_var(model)
    if not var:
        return None
    value = secret(var, settings.keychain_service(getattr(settings, _KEYCHAIN_ATTR[var])))
    return (var, value) if value else None


def _provider_keys(models: list[str | None], settings: Settings) -> list[tuple[str, str]]:
    """`(env var, value)` per model, deduped by var, valueless dropped: agent key plus user key."""
    found: dict[str, str] = {}
    for model in models:
        pair = _provider_key(model, settings)
        if pair and pair[0] not in found:
            found[pair[0]] = pair[1]
    return list(found.items())


def _sync_src(host: str, with_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local = remote._stage_src(Path(tmp))
        _call(_rsync_cmd(["-az", "--delete", f"{local}/", f"{host}:{with_path}/"]))


def _harbor_cmd(run_path: str, agent: str, model: str | None, jobs_dir: str, n_concurrent: int,
                extra_args: list[str], with_path: str | None = None) -> list[str]:
    prefix = ["uvx", "--from", "harbor", "--with", with_path, "harbor", "run"] if with_path \
        else ["harbor", "run"]
    cmd = prefix + ["-p", run_path, "-a", agent, "-o", jobs_dir, "-n", str(n_concurrent), "-y"]
    if model:
        cmd += ["-m", model]
    return cmd + extra_args


def _run_local(path: Path, agent: str, model: str | None, jobs_dir: Path, n_concurrent: int,
               extra_args: list[str], keys_: list[tuple[str, str]] | None = None) -> Path:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    before = _job_dirs(jobs_dir)
    cmd = _harbor_cmd(str(_run_path(path).resolve()), agent, model, str(jobs_dir.resolve()),
                      n_concurrent, extra_args)
    _call(cmd, env={**os.environ, **dict(keys_)} if keys_ else None)
    return _newest_job(jobs_dir, before)


def _remote_key_prefix(keys_: list[tuple[str, str]] | None) -> tuple[str, str | None]:
    """Read each key from stdin (one line per key, in order) into its env var; no value on argv."""
    if not keys_:
        return "", None
    prefix = "".join(f'IFS= read -r _tsk; export {var}="$_tsk"; ' for var, _ in keys_)
    return prefix, "".join(value + "\n" for _, value in keys_)


def _run_remote(path: Path, agent: str, model: str | None, jobs_dir: Path, n_concurrent: int,
                extra_args: list[str], settings: Settings,
                keys_: list[tuple[str, str]] | None = None) -> Path:
    host, remote_root = settings.harbor_host, settings.harbor_remote_root
    sync_root = path.parent if path.name == "tasks" else path
    rel_run = _run_path(path).relative_to(sync_root).as_posix() or "."
    remote_path = remote.remote_dataset_path(remote_root, sync_root)
    run_out = f"{remote_path}/runs/{new_id()}"  # this run's own output dir; nothing else lands here

    # --exclude jobs/runs so a --delete push never wipes job dirs or prior run outputs.
    _call(_rsync_cmd(["-az", "--delete", "--exclude", "jobs", "--exclude", "runs",
                      f"{sync_root}/", f"{host}:{remote_path}/"]))
    with_path = None
    if _is_custom_agent(agent):  # ship touchstone so harbor can import the custom agent
        with_path = f"{remote_root}/touchstone-src"
        _sync_src(host, with_path)
    # -o must be ABSOLUTE: a verifier's `docker compose cp` resolves a relative path wrong.
    remote_cmd = " ".join(
        _harbor_cmd(rel_run, agent, model, run_out, n_concurrent, extra_args, with_path))
    prefix, stdin_data = _remote_key_prefix(keys_)
    _call(_ssh_cmd(host, f"{prefix}{_REMOTE_PATH}; cd {remote_path} && {remote_cmd}"),
          stdin_data=stdin_data)

    jobs_dir.mkdir(parents=True, exist_ok=True)
    before = _job_dirs(jobs_dir)
    # Pull back ONLY this run's output dir — never the other jobs a shared host holds.
    _call(_rsync_cmd(["-az", f"{host}:{run_out}/", f"{jobs_dir}/"]))
    return _newest_job(jobs_dir, before)


def _build_remote(context_dir: Path, tag: str, settings: Settings) -> None:
    host, remote_root = settings.harbor_host, settings.harbor_remote_root
    remote_ctx = f"{remote_root}/env-build/{tag.replace(':', '-')}"
    _call(_rsync_cmd(["-az", "--delete", f"{context_dir}/", f"{host}:{remote_ctx}/"]))
    _build_call(_ssh_cmd(host, f"{_REMOTE_PATH}; cd {remote_ctx} && docker build -t {tag} ."),
                f"host {host}")


def build_image(context_dir: str | Path, tag: str, settings: Settings | None = None) -> None:
    """Build the image `tag` from `context_dir`, on the SSH host when there is no local Docker."""
    context_dir = Path(context_dir)
    settings = settings or load_settings()
    if _require_target(settings) == "remote":
        _build_remote(context_dir, tag, settings)
    else:
        _build_call(["docker", "build", "-t", tag, str(context_dir)], "local")


def _tasks_dir(path: Path) -> Path:
    return path if path.name == "tasks" else path / "tasks"


def _task_multi_turn(task_dir: Path) -> bool:
    try:
        doc = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return bool(doc.get("metadata", {}).get("touchstone", {}).get("multi_turn"))


def dataset_is_multi_turn(path: str | Path) -> bool:
    """True when the target task or any dataset task is multi-turn (needs the simulated user)."""
    p = Path(path)
    if (p / "task.toml").is_file():
        return _task_multi_turn(p)
    tasks = _tasks_dir(p)
    return any(_task_multi_turn(d) for d in tasks.glob("*") if (d / "task.toml").is_file())


def user_agent_for(model: str) -> str:
    """The simulated-user agent for the provider: codex for OpenAI, claude-code for Anthropic."""
    return "claude-code" if model.split("/", 1)[0] == "anthropic" else "codex"


def simulated_user_args(user_agent: str, user_model: str,
                        persona: str | Path | None = None) -> list[str]:
    """Harbor flags that put a simulated user on the other side of the agent (multi-turn tasks)."""
    agent = user_agent or user_agent_for(user_model)
    args = ["--user-agent", agent, "--user-model", user_model, "--bridge", "acp"]
    if persona:
        args += ["--user-persona-path", str(persona)]
    return args


def run(path: str | Path, agent: str, *, model: str | None = None,
        user_model: str | None = None, jobs_dir: str | Path = "jobs",
        n_concurrent: int = 4, extra_args: list[str] | None = None,
        settings: Settings | None = None) -> Path:
    """Run Harbor over `path` (a task dir or dataset root) and return the created job directory.
    `user_model` is the multi-turn simulated user's model; its key is forwarded with the agent's."""
    path, jobs_dir = Path(path), Path(jobs_dir)
    extra_args = extra_args or []
    settings = settings or load_settings()
    want = ([model] if _is_custom_agent(agent) else []) + ([user_model] if user_model else [])
    keys_ = _provider_keys(want, settings)
    if _require_target(settings) == "remote":
        return _run_remote(path, agent, model, jobs_dir, n_concurrent, extra_args, settings, keys_)
    return _run_local(path, agent, model, jobs_dir, n_concurrent, extra_args, keys_)


# Re-export `regrade` so `run_mod.regrade` stays the public entry point; its implementation lives in
# regrade_run to keep this file small. Imported at the bottom so run's helpers are defined first.
from .regrade_run import regrade  # noqa: E402,F401
