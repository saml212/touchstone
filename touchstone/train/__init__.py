"""Training hook: prepare real datasets today; plug ART/TRL in when GPU infra exists."""

from __future__ import annotations

from .art import ArtTrainer
from .datasets import DatasetBundle, default_out_dir, prepare
from .trainer import InfraRequired, JobHandle, NullTrainer, TrainConfig, Trainer
from .trl import TrlTrainer

BACKENDS = {"null": NullTrainer, "art": ArtTrainer, "trl": TrlTrainer}


def trainer_for(backend: str) -> Trainer:
    """Return a Trainer for a backend name, or raise ValueError naming the valid backends."""
    cls = BACKENDS.get(backend)
    if cls is None:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    return cls()


__all__ = [
    "prepare",
    "default_out_dir",
    "DatasetBundle",
    "Trainer",
    "NullTrainer",
    "ArtTrainer",
    "TrlTrainer",
    "TrainConfig",
    "JobHandle",
    "InfraRequired",
    "BACKENDS",
    "trainer_for",
]
