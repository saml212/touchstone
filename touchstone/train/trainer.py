"""The training hook: a Protocol, a null implementation, and the shared value types.

Touchstone never trains a model itself — it produces the datasets and the exact runnable config, and
makes it obvious where GPU infra plugs in. `Trainer.prepare` writes the datasets via
`datasets.prepare`. `Trainer.submit` either writes a plan (`NullTrainer`) or writes a real backend
config and raises `InfraRequired` naming the file and command to run on GPUs (`art.py`/`trl.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .datasets import DatasetBundle, prepare


class InfraRequired(Exception):
    """Raised by a real backend after it writes its config: the run needs GPU infra we don't have.

    The message is one sentence naming the written file and the command to run where GPUs exist.
    """


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    steps: int = 200
    extra: dict = field(default_factory=dict)


@dataclass
class JobHandle:
    backend: str
    status: str  # "planned" | "config_written"
    detail: str = ""
    artifacts: list[str] = field(default_factory=list)


class Trainer(Protocol):
    def prepare(self, conn, benchmark_id: str, out_dir: str | Path) -> DatasetBundle: ...

    def submit(self, bundle: DatasetBundle, config: TrainConfig) -> JobHandle: ...


PLAN_TEMPLATE = """\
# Training plan — {benchmark_name}

Benchmark: `{benchmark_id}`
Datasets: `{out_dir}`

## What Touchstone prepared

| dataset | file | rows | purpose |
| --- | --- | --- | --- |
| SFT | `sft.jsonl` | {sft} | fine-tune on references that pass every attached hard check |
| Preference | `preference.jsonl` | {preference} | DPO pairs: a passing reply over a failing one |
| RL | `rl_tasks.jsonl` | {rl_tasks} | context + tools + serialized checks as the reward verifier |

## What would run (backend `null` — nothing executes here)

1. Load `sft.jsonl` and run supervised fine-tuning of `{base_model}` for a warm start.
2. Load `preference.jsonl` and run preference optimization (DPO) on the SFT checkpoint.
3. Load `rl_tasks.jsonl` and run RL, scoring each rollout with the task's checks as the reward.

To emit a runnable backend config instead of this plan, submit with `--backend art` or
`--backend trl`; each writes its config and prints the one command to run on a GPU host.
"""


class NullTrainer:
    """Prepares datasets and writes a `train_plan.md`; runs nothing. The zero-infra default."""

    backend = "null"

    def prepare(self, conn, benchmark_id: str, out_dir: str | Path) -> DatasetBundle:
        return prepare(conn, benchmark_id, out_dir)

    def submit(self, bundle: DatasetBundle, config: TrainConfig) -> JobHandle:
        from .datasets import atomic_write

        plan_path = bundle.out_dir / "train_plan.md"
        atomic_write(plan_path, PLAN_TEMPLATE.format(
            benchmark_name=bundle.benchmark_name,
            benchmark_id=bundle.benchmark_id,
            out_dir=bundle.out_dir,
            base_model=config.base_model,
            sft=bundle.counts["sft"],
            preference=bundle.counts["preference"],
            rl_tasks=bundle.counts["rl_tasks"],
        ))
        return JobHandle(
            backend="null",
            status="planned",
            detail=f"wrote {plan_path}",
            artifacts=[str(plan_path)],
        )
