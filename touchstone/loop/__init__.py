"""The Sample → Distill loop: find where a candidate fails, then close the gap.

`frontier` measures difficulty and finds the learnability band; `sample` runs the student and the
teacher's variants; `teach` produces gated demonstrations; `distill` packages the frontier into
training data. All run off the store and the task files, deterministically with the `scripted`
provider.
"""

from .frontier import (
    check_stop,
    frontier,
    frontier_split,
    mean_pass_rate,
    read_loop_state,
    record_run,
    write_loop_state,
)
from .sample import sample

__all__ = [
    "record_run",
    "frontier",
    "frontier_split",
    "mean_pass_rate",
    "check_stop",
    "read_loop_state",
    "write_loop_state",
    "sample",
]
