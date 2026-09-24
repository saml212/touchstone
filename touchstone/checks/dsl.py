"""Checks DSL: the `Check` model, its evaluation `Target`, the kind enum, and per-kind validation.

A `Check` is `(kind, params)` plus routing metadata. `from_dict` / `to_dict` round-trip through the
shapes `store.Check` uses, so a stored row can be evaluated and an authored check can be stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Kind(StrEnum):
    contains = "contains"
    not_contains = "not_contains"
    regex = "regex"
    not_regex = "not_regex"
    json_schema = "json_schema"
    tool_called = "tool_called"
    tool_not_called = "tool_not_called"
    tool_order = "tool_order"
    max_length = "max_length"
    min_length = "min_length"
    no_pii = "no_pii"
    expr = "expr"
    judge = "judge"


KINDS = frozenset(k.value for k in Kind)
APPLIES_TO = frozenset({"final", "any_turn", "tool_calls"})
SEVERITIES = frozenset({"hard", "soft"})


def is_catastrophic_regex(pattern: str) -> bool:
    """True when `pattern` has a quantified group whose body itself repeats unboundedly
    (e.g. `(a+)+`, `([a-z]+)*`), the signature of exponential backtracking that can hang the
    engine. Only `*`, `+`, and open-ended `{m,}` count as unbounded; `{m,n}` and `{k}` do not."""
    stack: list[bool] = []
    i, n = 0, len(pattern)

    def open_ended_brace(k: int) -> tuple[bool, int]:
        j = k + 1
        while j < n and pattern[j] != "}":
            j += 1
        parts = pattern[k + 1 : j].split(",")
        return len(parts) == 2 and parts[1].strip() == "", j + 1

    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
        elif c == "[":
            i += 1
            i += 1 if i < n and pattern[i] == "^" else 0
            i += 1 if i < n and pattern[i] == "]" else 0
            while i < n and pattern[i] != "]":
                i += 2 if pattern[i] == "\\" else 1
            i += 1
        elif c == "(":
            stack.append(False)
            i += 1
        elif c == ")":
            body_has = stack.pop() if stack else False
            j = i + 1
            quantified = j < n and (
                pattern[j] in "*+" or (pattern[j] == "{" and open_ended_brace(j)[0])
            )
            if quantified and body_has:
                return True
            if stack and (body_has or quantified):
                stack[-1] = True
            i += 1
        elif c in "*+":
            if stack:
                stack[-1] = True
            i += 1
        elif c == "{":
            open_ended, after = open_ended_brace(i)
            if open_ended and stack:
                stack[-1] = True
            i = after
        else:
            i += 1
    return False


def _require_safe_regex(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    if is_catastrophic_regex(pattern):
        raise ValueError(
            f"regex {pattern!r} has a nested quantifier prone to catastrophic backtracking; "
            "rewrite without a repeated group that itself repeats (e.g. avoid '(a+)+')"
        )


@dataclass
class Target:
    """What a check is evaluated against."""

    output_text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name","arguments"}]
    messages: list[dict] = field(default_factory=list)
    reference: dict | None = None


def coerce_tool_calls(value) -> list[dict]:
    """Validate an untrusted tool_calls payload (CLI/API boundary) into a list of dict calls."""
    if value in (None, ""):
        return []
    if not isinstance(value, list) or not all(isinstance(tc, dict) for tc in value):
        raise ValueError("tool_calls must be a JSON list of {name, arguments} objects")
    return value


@dataclass
class Check:
    kind: str
    params: dict = field(default_factory=dict)
    id: str = ""
    name: str = ""
    applies_to: str = "final"
    severity: str = "hard"

    _FIELDS = ("kind", "params", "id", "name", "applies_to", "severity")

    @classmethod
    def from_dict(cls, data: dict) -> Check:
        kwargs = {k: data[k] for k in cls._FIELDS if k in data and data[k] is not None}
        if "kind" not in kwargs:
            raise ValueError("check dict is missing 'kind'")
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "params": self.params,
            "id": self.id,
            "name": self.name,
            "applies_to": self.applies_to,
            "severity": self.severity,
        }

    def validate(self) -> None:
        validate_params(self.kind, self.params)
        if self.applies_to not in APPLIES_TO:
            raise ValueError(f"applies_to must be one of {sorted(APPLIES_TO)}")
        if self.severity not in SEVERITIES:
            raise ValueError("severity must be 'hard' or 'soft'")


# ---- per-kind parameter validation -----------------------------------------


def _require_str_list(params: dict, key: str) -> None:
    value = params.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ValueError(f"'{key}' must be a non-empty list of strings")


def _require_str(params: dict, key: str) -> None:
    if not isinstance(params.get(key), str) or not params[key]:
        raise ValueError(f"'{key}' must be a non-empty string")


def _require_int(params: dict, key: str) -> None:
    if not isinstance(params.get(key), int) or isinstance(params.get(key), bool):
        raise ValueError(f"'{key}' must be an integer")


PARAM_SPEC = {
    "contains": '{"values": [str, ...], "mode": "any|all"}',
    "not_contains": '{"values": [str, ...], "mode": "any|all"}',
    "regex": '{"pattern": str}',
    "not_regex": '{"pattern": str}',
    "json_schema": '{"schema": {json schema object}}',
    "tool_called": '{"name": str, "arguments_match": {field: value | {"regex": str}}?}',
    "tool_not_called": '{"name": str}',
    "tool_order": '{"order": [str, ...]}',
    "max_length": '{"max": int}',
    "min_length": '{"min": int}',
    "no_pii": '{"kinds": ["email"|"phone"|"card", ...]?}',
    "expr": '{"expr": str}  # simpleeval over output, tools, reference',
    "judge": '{"rubric": str}',
}


def validate_params(kind: str, params: dict) -> None:
    """Raise ValueError with a clear message if `params` is wrong for `kind`."""
    if kind not in KINDS:
        raise ValueError(f"unknown check kind {kind!r}; valid kinds: {sorted(KINDS)}")
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")

    if kind in ("contains", "not_contains"):
        _require_str_list(params, "values")
        mode = params.get("mode", "any")
        if mode not in ("any", "all"):
            raise ValueError("'mode' must be 'any' or 'all'")
    elif kind in ("regex", "not_regex"):
        _require_str(params, "pattern")
        _require_safe_regex(params["pattern"])
    elif kind == "json_schema":
        if not isinstance(params.get("schema"), dict):
            raise ValueError("'schema' must be a JSON object")
        import jsonschema

        try:
            jsonschema.Draft202012Validator.check_schema(params["schema"])
        except jsonschema.exceptions.SchemaError as exc:
            raise ValueError(f"invalid JSON schema: {exc.message}") from exc
    elif kind == "tool_called":
        _require_str(params, "name")
        matcher = params.get("arguments_match")
        if matcher is not None:
            if not isinstance(matcher, dict):
                raise ValueError("'arguments_match' must be an object of field matchers")
            for expected in matcher.values():
                if isinstance(expected, dict) and "regex" in expected:
                    _require_safe_regex(str(expected["regex"]))
    elif kind == "tool_not_called":
        _require_str(params, "name")
    elif kind == "tool_order":
        _require_str_list(params, "order")
    elif kind == "max_length":
        _require_int(params, "max")
    elif kind == "min_length":
        _require_int(params, "min")
    elif kind == "no_pii":
        kinds = params.get("kinds")
        if kinds is not None and not (
            isinstance(kinds, list) and all(k in ("email", "phone", "card") for k in kinds)
        ):
            raise ValueError("'kinds' must be a list drawn from email/phone/card")
    elif kind == "expr":
        _require_str(params, "expr")
    elif kind == "judge":
        _require_str(params, "rubric")
