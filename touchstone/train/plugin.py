"""A Harbor job plugin: after `harbor run ... --plugin touchstone.train:TrainPlugin`, write the
distillation + RL datasets with no separate `touchstone train` command.

The plugin runs inside Harbor's own process (where `harbor` is importable and touchstone is added
with `--with`), so it imports `BaseJobPlugin` lazily and falls back to a plain base when Harbor is
absent — that keeps this module importable in the touchstone dev env and under test.

One completed job is both teacher and student: its passing trials distill, and its per-task pass
rate routes each task. Output defaults to `<dataset>/train` (the dataset holding `jobs/`).
"""

from __future__ import annotations

from pathlib import Path

try:  # inside Harbor's env
    from harbor.models.job.plugin import BaseJobPlugin
except ImportError:  # touchstone dev env / tests
    class BaseJobPlugin:  # minimal stand-in with the same constructor contract
        def __init__(self, **kwargs: object) -> None:
            pass


class TrainPlugin(BaseJobPlugin):
    """Write Touchstone training datasets when a Harbor job ends.

    `--pk out=<dir>` overrides the output directory; `--pk threshold=<float>` the distill cutoff.
    """

    def __init__(self, out: str | None = None, threshold: float = 1.0, **_: object) -> None:
        super().__init__()
        self._out = out
        self._threshold = float(threshold)
        self._job_dir: Path | None = None

    async def on_job_start(self, job: object) -> None:
        # Capture the directory now; JobResult carries no path at on_job_end.
        self._job_dir = Path(job.job_dir)

    async def on_job_end(self, job_result: object) -> None:
        if self._job_dir is None:
            return
        self.write(self._job_dir)

    def write(self, job_dir: Path) -> object:
        """Read the finished job and write the datasets. Split out so a test drives it directly."""
        from ..harbor.jobs import Job
        from .write import write_datasets

        job = Job.read(job_dir)
        out = self._out or job_dir.parent.parent / "train"
        return write_datasets(out, [job], [job], threshold=self._threshold)
