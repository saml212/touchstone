"""Standalone verifier, copied verbatim into every task's `tests/verify.py`.

Reads the `[[metadata.touchstone.check]]` blocks straight out of `task.toml` (the single source of
truth — no separate checks file), grades `/app/output.json`, and writes `/logs/verifier/reward.txt`
(1.0/0.0) plus `rewards.json` (per-check). Depends only on the vendored `touchstone_checks` package
sitting beside it, so it needs neither the network nor the `touchstone` install. Paths are
env-overridable so the same file runs in a Harbor container (defaults) and in a local subprocess.
"""

import json
import os
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from touchstone_checks.dsl import Check, Target  # noqa: E402
from touchstone_checks.run import evaluate, passes  # noqa: E402


def _load_reference(task_dir: Path, ts: dict):
    path = task_dir / ts.get("reference_file", "reference.json")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    task_toml = Path(os.environ.get("TOUCHSTONE_TASK", HERE / "task.toml"))
    output_path = Path(os.environ.get("TOUCHSTONE_OUTPUT", "/app/output.json"))
    reward_dir = Path(os.environ.get("TOUCHSTONE_REWARD_DIR", "/logs/verifier"))
    reward_dir.mkdir(parents=True, exist_ok=True)

    doc = tomllib.loads(task_toml.read_text())
    ts = doc.get("metadata", {}).get("touchstone", {})
    checks = [Check.from_toml(b) for b in ts.get("check", [])]
    reference = _load_reference(task_toml.parent, ts)

    try:
        out = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        out = {}
    if not isinstance(out, dict):
        out = {}

    target = Target(output_text=out.get("content") or "",
                    tool_calls=out.get("tool_calls") or [], reference=reference)
    results = evaluate(checks, target, judge_provider=None)
    ok = passes(results, checks)
    reward = 1.0 if ok else 0.0

    (reward_dir / "reward.txt").write_text(f"{reward}\n")
    per_check = {r.check_id: r.passed for r in results}
    (reward_dir / "rewards.json").write_text(json.dumps({"reward": reward, "checks": per_check}))
    for r in results:
        print(f"{r.check_id}: {r.passed} — {r.evidence}")
    print(f"REWARD {reward}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
