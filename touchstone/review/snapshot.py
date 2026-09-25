"""Put a task's tests/ back when a change broke its verifier (every touched trial refused to
regrade). The review agent snapshots before applying and restores on that outcome."""

from __future__ import annotations

from pathlib import Path


def snapshot_tests(task_dir: Path) -> dict[Path, str]:
    """Every file under tests/, so a change that breaks the verifier can be put back."""
    tests = task_dir / "tests"
    return {p.relative_to(tests): p.read_text(encoding="utf-8")
            for p in tests.rglob("*") if p.is_file()}


def restore_tests(task_dir: Path, files: dict[Path, str]) -> None:
    for rel, text in files.items():
        (task_dir / "tests" / rel).write_text(text, encoding="utf-8")


