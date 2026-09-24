"""Evaluate checks against a Target and decide pass/fail.

`evaluate` returns one `CheckResult` per check; `passes` applies hard/soft gating: a Target passes
when every hard check is True (a hard check that is False or errored/None fails it), and soft checks
are reported but never gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import jsonschema
from simpleeval import EvalWithCompoundTypes

from .dsl import Target, is_catastrophic_regex


@dataclass
class CheckResult:
    check_id: str
    passed: bool | None  # None = errored / skipped / low agreement
    evidence: str = ""
    agreement: float | None = None  # judge-sampling agreement, when the check was sampled


def evaluate(checks, target: Target, judge_provider=None) -> list[CheckResult]:
    results = []
    for check in checks:
        agreement = None
        try:
            passed, evidence, agreement = _run_one(check, target, judge_provider)
        except Exception as exc:  # one check must never crash a whole benchmark run
            passed, evidence = None, f"check error: {type(exc).__name__}: {exc}"
        results.append(CheckResult(check_id=getattr(check, "id", "") or "", passed=passed,
                                   evidence=evidence, agreement=agreement))
    return results


def passes(results: list[CheckResult], checks) -> bool:
    severity = {getattr(c, "id", ""): getattr(c, "severity", "hard") for c in checks}
    for r in results:
        if severity.get(r.check_id, "hard") == "hard" and r.passed is not True:
            return False
    return True


# ---- dispatch --------------------------------------------------------------


def _run_one(check, target: Target, judge_provider) -> tuple[bool | None, str, float | None]:
    kind = check.kind
    if kind == "judge":
        from . import judge as judge_mod  # keeps vendored run.py free of the llm subpackage

        return judge_mod.judge(check, target, judge_provider)
    fn = _EVALUATORS.get(kind)
    if fn is None:
        return None, f"no evaluator for kind {kind!r}", None
    passed, evidence = fn(check.params or {}, target)
    return passed, evidence, None


def _contains(params, target):
    values = params["values"]
    case_sensitive = params.get("case_sensitive", False)
    hay = target.output_text if case_sensitive else target.output_text.casefold()
    needles = values if case_sensitive else [v.casefold() for v in values]
    hits = [v for v, n in zip(values, needles, strict=True) if n in hay]
    found = len(hits) == len(values) if params.get("mode", "any") == "all" else bool(hits)
    return found, f"matched {hits}" if hits else "no values found"


def _not_contains(params, target):
    found, _ = _contains(params, target)
    return not found, "clean" if not found else "forbidden value present"


def _regex(params, target):
    if is_catastrophic_regex(params["pattern"]):
        return None, "regex rejected: nested quantifier risks catastrophic backtracking"
    m = re.search(params["pattern"], target.output_text)
    return bool(m), f"matched {m.group()!r}" if m else "pattern not found"


def _not_regex(params, target):
    if is_catastrophic_regex(params["pattern"]):
        return None, "regex rejected: nested quantifier risks catastrophic backtracking"
    m = re.search(params["pattern"], target.output_text)
    return m is None, "clean" if m is None else f"matched {m.group()!r}"


def _json_schema(params, target):
    try:
        instance = json.loads(target.output_text)
    except (json.JSONDecodeError, TypeError):
        return False, "output is not valid JSON"
    try:
        validator = jsonschema.Draft202012Validator(params["schema"])
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    except Exception as exc:  # invalid schema or unresolvable $ref must not crash the run
        return None, f"invalid JSON schema: {exc}"
    if errors:
        return False, errors[0].message
    return True, "valid against schema"


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _key_matches(key, expected, parsed, raw_str: str) -> bool:
    """Whether one field matcher is satisfied against the parsed args (or the raw arg string)."""
    is_regex = isinstance(expected, dict) and "regex" in expected
    if is_regex and is_catastrophic_regex(str(expected["regex"])):
        return False
    if parsed is not None and key in parsed:
        actual = parsed[key]
        return bool(re.search(expected["regex"], str(actual))) if is_regex else actual == expected
    # unparseable args or missing key: match against the raw string
    if is_regex:
        return bool(re.search(expected["regex"], raw_str))
    return str(expected) in raw_str


def _arg_matches(raw, matcher) -> bool:
    parsed = _parse_args(raw)
    raw_str = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    return all(_key_matches(key, expected, parsed, raw_str) for key, expected in matcher.items())


def _tool_called(params, target):
    name = params["name"]
    matcher = params.get("arguments_match")
    matches = [tc for tc in target.tool_calls if tc.get("name") == name]
    if not matches:
        return False, f"tool {name!r} was not called"
    if not matcher:
        return True, f"tool {name!r} called"
    if any(_arg_matches(tc.get("arguments"), matcher) for tc in matches):
        return True, f"tool {name!r} called with matching arguments"
    return False, f"tool {name!r} called but arguments did not match"


def _tool_not_called(params, target):
    name = params["name"]
    called = any(tc.get("name") == name for tc in target.tool_calls)
    return not called, f"tool {name!r} not called" if not called else f"tool {name!r} was called"


def _tool_order(params, target):
    order = params["order"]
    names = [tc.get("name") for tc in target.tool_calls]
    i = 0
    for n in names:
        if i < len(order) and n == order[i]:
            i += 1
    ok = i == len(order)
    return ok, f"order satisfied by {names}" if ok else f"expected {order} within {names}"


def _max_length(params, target):
    n = len(target.output_text)
    return n <= params["max"], f"length {n} (max {params['max']})"


def _min_length(params, target):
    n = len(target.output_text)
    return n >= params["min"], f"length {n} (min {params['min']})"


def _no_pii(params, target):
    wanted = params.get("kinds") or ["email", "phone", "card"]
    found = {k: v for k, v in find_pii(target.output_text).items() if k in wanted and v}
    if not found:
        return True, "no PII detected"
    summary = ", ".join(f"{k}({len(v)})" for k, v in found.items())
    return False, f"PII detected: {summary}"


def _expr(params, target):
    names = {
        "output": target.output_text,
        "tools": target.tool_calls,
        "reference": target.reference,
    }
    # EvalWithCompoundTypes allows comprehensions so a state assertion can iterate `tools`
    # (e.g. any(t.get('name') == 'refund' ... for t in tools)); it keeps the same dunder/import
    # sandbox as SimpleEval. `any`/`all` join the whitelisted functions.
    evaluator = EvalWithCompoundTypes(names=names, functions={"len": len, "any": any, "all": all})
    try:
        value = evaluator.eval(params["expr"])
    except Exception as exc:  # sandbox refusal, timeout guards, bad expression
        return None, f"expr error: {type(exc).__name__}: {exc}"
    return bool(value), f"expr -> {value!r}"


_EVALUATORS = {
    "contains": _contains,
    "not_contains": _not_contains,
    "regex": _regex,
    "not_regex": _not_regex,
    "json_schema": _json_schema,
    "tool_called": _tool_called,
    "tool_not_called": _tool_not_called,
    "tool_order": _tool_order,
    "max_length": _max_length,
    "min_length": _min_length,
    "no_pii": _no_pii,
    "expr": _expr,
    # "judge" is handled in _run_one before this table (it needs a provider).
}


# ---- PII detection ---------------------------------------------------------

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_E164 = re.compile(r"\+[1-9]\d{7,14}\b")
_US_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CARD_CANDIDATE = re.compile(r"\b\d[\d -]{11,17}\d\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_pii(text: str) -> dict[str, list[str]]:
    """Return detected PII keyed by kind. Card numbers must pass Luhn to count."""
    cards = []
    for m in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            cards.append(m.group())
    phones = set(_E164.findall(text)) | set(_US_PHONE.findall(text))
    # A valid card is not also a phone (avoid double counting long digit runs).
    card_digits = {re.sub(r"\D", "", c) for c in cards}
    phones = {p for p in phones if re.sub(r"\D", "", p) not in card_digits}
    return {
        "email": _EMAIL.findall(text),
        "phone": sorted(phones),
        "card": cards,
    }
