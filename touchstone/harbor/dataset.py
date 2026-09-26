"""The `touchstone/` dataset: a plain Harbor dataset the customer owns and can run without us.

Layout (all under the dataset root):

    dataset.toml     the Harbor dataset manifest
    tasks/           one directory per Harbor task
    environment/     the customer's system, copied to run in a sandbox
    agent/           the agent under test, packaged as a Harbor custom agent
    simulators/      a small local service per network boundary
    report.md        what was mapped, simulated, and left open

Output is stable and sorted so re-running is diff-friendly.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

MANIFEST = "dataset.toml"
SUBDIRS = ("tasks", "environment", "agent", "simulators")


@dataclass
class Dataset:
    name: str  # "org/name"
    version: str = "1.0.0"
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    survey_id: str = ""  # a short id minted per survey; scopes the remote build/jobs dir
    root: Path | None = None

    @classmethod
    def read(cls, root: str | Path) -> Dataset:
        root = Path(root)
        data = tomllib.loads((root / MANIFEST).read_text(encoding="utf-8"))
        info = data.get("dataset", {})
        survey_id = data.get("metadata", {}).get("touchstone", {}).get("survey_id", "")
        return cls(name=info.get("name", root.name), version=info.get("version", "1.0.0"),
                   description=info.get("description", ""),
                   keywords=list(info.get("keywords", [])), survey_id=survey_id, root=root)

    def write(self, root: str | Path) -> Path:
        root = Path(root)
        for sub in SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        (root / MANIFEST).write_text(self._manifest(), encoding="utf-8")
        report = root / "report.md"
        if not report.exists():
            report.write_text(f"# {self.name}\n", encoding="utf-8")
        self.root = root
        return root

    def task_dirs(self) -> list[Path]:
        """Every task directory (sorted) — a `tasks/` child that carries a `task.toml`."""
        tasks = (self.root or Path()) / "tasks"
        if not tasks.is_dir():
            return []
        return sorted(d for d in tasks.iterdir() if (d / "task.toml").is_file())

    def _manifest(self) -> str:
        # `tasks` stays empty: Harbor's manifest pins tasks by published digest, so local runs use
        # the implicit dataset (`harbor run -p <root>/tasks`). This file is metadata only.
        # Top-level keys are written before the [dataset] table so they don't fall inside it.
        doc = {"schema_version": "1.0", "tasks": [],
               "dataset": {"name": self.name, "version": self.version,
                           "description": self.description, "keywords": sorted(self.keywords)}}
        if self.survey_id:  # scopes this survey's remote directory so a re-survey never bleeds jobs
            doc["metadata"] = {"touchstone": {"survey_id": self.survey_id}}
        return tomli_w.dumps(doc)
