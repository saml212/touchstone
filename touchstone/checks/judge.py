"""LLM-judge check: ask a provider to grade an output against a rubric.

One judge response is a noisy reward that a policy can learn to exploit, so the judge is sampled
`samples` times (default 3, from the check's `samples` key), combined by majority vote (a split
counts as no, so a tie never awards credit), and the agreement — the majority's share of the valid
votes — is reported alongside. A check whose agreement falls below `min_agreement` (default 0.67)
yields `passed=None` ("low agreement"): a criterion the judge is unsure about never gates on its
own. Ported from the founder's RewardKit `samples = N` design.

`judge` returns `(passed, evidence, agreement)`. It never raises: a missing provider is reported as
skipped, a malformed reply is a `None` sample, and a run where every sample is `None` is `None`.
"""

from __future__ import annotations

import json

from ..llm.prompt import extract_json

_SYSTEM = (
    "You are a strict evaluator. Judge whether the OUTPUT satisfies the RUBRIC. "
    'Reply with ONLY a JSON object: {"pass": true|false, "reason": "<short reason>"}.'
)
_DEFAULT_SAMPLES = 3
_DEFAULT_MIN_AGREEMENT = 0.67


def _prompt(rubric: str, output_text: str, reference) -> str:
    parts = [f"RUBRIC:\n{rubric}", f"OUTPUT:\n{output_text}"]
    if reference:
        ref = reference if isinstance(reference, str) else json.dumps(reference, ensure_ascii=False)
        parts.append(f"REFERENCE (ideal answer):\n{ref}")
    return "\n\n".join(parts)


def _judge_once(rubric: str, target, provider) -> tuple[bool | None, str]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _prompt(rubric, target.output_text, target.reference)},
    ]
    try:
        reply = provider.chat(messages, json=True)
    except Exception as exc:  # provider failures must not crash evaluation
        return None, f"judge provider error: {exc}"
    raw = reply.content or ""
    obj_str = extract_json(raw)
    try:
        obj = json.loads(obj_str) if obj_str is not None else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict) or "pass" not in obj:
        return None, f"judge reply was not JSON with a 'pass' field: {raw[:200]}"
    return bool(obj["pass"]), str(obj.get("reason", "")).strip() or "no reason given"


def _pos_int(value, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def judge(check, target, provider) -> tuple[bool | None, str, float | None]:
    if provider is None:
        return None, "skipped: no judge provider configured", None
    rubric = check.params.get("rubric", "")
    n = _pos_int(check.params.get("samples", _DEFAULT_SAMPLES), _DEFAULT_SAMPLES)
    min_agreement = check.params.get("min_agreement", _DEFAULT_MIN_AGREEMENT)

    samples = [_judge_once(rubric, target, provider) for _ in range(n)]
    verdicts = [(passed, reason) for passed, reason in samples if passed is not None]
    if not verdicts:
        return None, samples[-1][1], None

    trues = sum(1 for passed, _ in verdicts if passed)
    total = len(verdicts)
    majority = trues > total - trues  # a tie counts as no — a split never awards credit
    agreement = max(trues, total - trues) / total
    if agreement < float(min_agreement):
        return None, (f"low agreement ({agreement:.2f} < {float(min_agreement):.2f}) "
                      f"over {total} sample(s)"), agreement
    reason = next((r for passed, r in verdicts if passed is majority), verdicts[0][1])
    return majority, reason, agreement
