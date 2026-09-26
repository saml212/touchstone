"""Build a dataset's shared environment image on the chosen Docker target, idempotently.

Every task's environment Dockerfile is `FROM` one shared image that only the survey's gate/baseline
builds (`run_gate` -> `build_image`). A survey that ran without Docker (preflight stop, --skip-gate,
an older version) never built it, so a later `harbor run` fails at the task build with "pull access
denied … repository does not exist". bench, baseline, regrade and gate all call
`ensure_dataset_image` first — one code path — so the image is present before the run. Split out
of run to keep that file small; it reuses run's target-selection and build helpers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import Settings, load_settings

# `run` is imported lazily inside the functions: run re-exports this module's names at its own
# module bottom, so a top-level `from . import run` here would be a circular import.


def _image_exists_local(tag: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", tag],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _image_exists_remote(tag: str, settings: Settings) -> bool:
    from . import run
    probe = f"{run._REMOTE_PATH}; docker image inspect {tag} >/dev/null"
    try:
        return subprocess.run(run._ssh_cmd(settings.harbor_host, probe),
                              capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_image(context_dir: str | Path, tag: str, settings: Settings | None = None) -> None:
    """Build the image `tag` from `context_dir`, on the SSH host when there is no local Docker.
    Idempotent: when `tag` already exists on the chosen target, no build runs."""
    from . import run
    context_dir = Path(context_dir)
    settings = settings or load_settings()
    if run._require_target(settings) == "remote":
        if not _image_exists_remote(tag, settings):
            run._build_remote(context_dir, tag, settings)
    elif not _image_exists_local(tag):
        run._build_call(["docker", "build", "-t", tag, str(context_dir)], "local")


def ensure_dataset_image(dataset: str | Path, settings: Settings | None = None) -> None:
    """Build the dataset's shared environment image on the chosen target when it is missing, so the
    `FROM <shared image>` in every task's environment Dockerfile resolves. No-op for a dataset with
    no environment/ dir to build from."""
    from ..survey.environment import dataset_image
    found = dataset_image(dataset)
    if found is not None:
        build_image(*found, settings)
