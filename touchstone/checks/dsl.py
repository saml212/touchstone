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


@dataclass
class Target:
    """What a check is evaluated against."""

    output_text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name","arguments"}]
    messages: list[dict] = field(default_factory=list)
    reference: dict | None = None


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
        try:
            re.compile(params["pattern"])
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
    elif kind == "json_schema":
        if not isinstance(params.get("schema"), dict):
            raise ValueError("'schema' must be a JSON object")
    elif kind == "tool_called":
        _require_str(params, "name")
        matcher = params.get("arguments_match")
        if matcher is not None and not isinstance(matcher, dict):
            raise ValueError("'arguments_match' must be an object of field matchers")
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
