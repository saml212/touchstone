"""LLM-judge check: ask a provider to grade an output against a rubric.

Returns `(passed, evidence)`. It never raises: a missing provider is reported as skipped, and a
malformed reply becomes `passed=None` with the raw reply head in the evidence.
"""

from __future__ import annotations

import json

from ..llm.prompt import extract_json

_SYSTEM = (
    "You are a strict evaluator. Judge whether the OUTPUT satisfies the RUBRIC. "
    'Reply with ONLY a JSON object: {"pass": true|false, "reason": "<short reason>"}.'
)


def _prompt(rubric: str, output_text: str, reference) -> str:
    parts = [f"RUBRIC:\n{rubric}", f"OUTPUT:\n{output_text}"]
    if reference:
        ref = reference if isinstance(reference, str) else json.dumps(reference, ensure_ascii=False)
        parts.append(f"REFERENCE (ideal answer):\n{ref}")
    return "\n\n".join(parts)


def judge(check, target, provider) -> tuple[bool | None, str]:
    if provider is None:
        return None, "skipped: no judge provider configured"
    rubric = check.params.get("rubric", "")
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
    if obj_str is None:
        return None, f"judge reply was not JSON: {raw[:200]}"
    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError:
        return None, f"judge reply was not JSON: {raw[:200]}"
    if not isinstance(obj, dict) or "pass" not in obj:
        return None, f"judge reply missing 'pass': {raw[:200]}"
    return bool(obj["pass"]), str(obj.get("reason", "")).strip() or "no reason given"
