"""Run `harbor job regrade` over a job dir, locally or on the remote Docker host.

The verifier reruns against the recorded outputs (no agent, no key), so a criterion change is graded
in seconds. Split out of `run` so each file stays small; it reuses run's ssh/rsync/exec helpers, and
`run` re-exports `regrade` so `run_mod.regrade` stays the public entry point.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, load_settings
from ..ids import new_id
from . import remote, run


def _regrade_cmd(job_dir: str, tasks_path: str, out: str) -> list[str]:
    return ["harbor", "job", "regrade", job_dir, "-p", tasks_path, "-o", out]


def _regrade_local(job_dir: Path, tasks_path: Path) -> Path:
    """Regrade `job_dir` against the updated tasks; the new job lands beside the source."""
    out = job_dir.parent
    before = run._job_dirs(out)
    run._call(_regrade_cmd(str(job_dir.resolve()), str(run._run_path(tasks_path).resolve()),
                           str(out.resolve())))
    return run._newest_job(out, before)


def _regrade_remote(job_dir: Path, tasks_path: Path, settings: Settings) -> Path:
    host = settings.harbor_host
    sync_root = job_dir.parent.parent  # the dataset root holding tasks/ and jobs/
    rel_tasks = run._run_path(tasks_path).relative_to(sync_root).as_posix()
    remote_path = remote.remote_dataset_path(settings.harbor_remote_root, sync_root)
    run_out = f"{remote_path}/runs/{new_id()}"  # the regrade's own output dir, pulled back alone
    # Push the updated tasks (never the whole jobs/runs tree), then just the one source job dir.
    run._call(run._rsync_cmd(["-az", "--delete", "--exclude", "jobs", "--exclude", "runs",
                              f"{sync_root}/", f"{host}:{remote_path}/"]))
    run._call(run._rsync_cmd(["-az", f"{job_dir}/", f"{host}:{remote_path}/jobs/{job_dir.name}/"]))
    remote_cmd = " ".join(  # absolute -o for the same compose-cp reason as run's remote run
        _regrade_cmd(f"{remote_path}/jobs/{job_dir.name}", rel_tasks, run_out))
    run._call(run._ssh_cmd(host, f"{run._REMOTE_PATH}; cd {remote_path} && {remote_cmd}"))
    out = job_dir.parent
    before = run._job_dirs(out)
    # Pull back ONLY the regrade's output — never the other jobs under the remote dataset.
    run._call(run._rsync_cmd(["-az", f"{host}:{run_out}/", f"{out}/"]))
    return run._newest_job(out, before)


def regrade(job_dir: str | Path, tasks_path: str | Path, *,
            settings: Settings | None = None) -> Path:
    """Run `harbor job regrade` over `job_dir` and return the new job dir. From recorded
    artifacts (no agent, no key); runs on the SSH host when there is no local Docker."""
    job_dir, tasks_path = Path(job_dir), Path(tasks_path)
    settings = settings or load_settings()
    if run._require_target(settings) == "remote":
        return _regrade_remote(job_dir, tasks_path, settings)
    return _regrade_local(job_dir, tasks_path)
