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
SOURCES = frozenset({"mined", "interview", "policy", "manual"})

# Safety kinds mean "avoid this": they attach even when the recorded reference was bad.
SAFETY_KINDS = frozenset({"no_pii", "not_contains", "not_regex", "tool_not_called"})


def _open_ended_brace(pattern: str, k: int, n: int) -> tuple[bool, int]:
    """Is the `{...}` at k an open-ended `{m,}` quantifier, and the index just past it."""
    j = k + 1
    while j < n and pattern[j] != "}":
        j += 1
    parts = pattern[k + 1 : j].split(",")
    return len(parts) == 2 and parts[1].strip() == "", j + 1


def _skip_char_class(pattern: str, i: int, n: int) -> int:
    """Index just past the `[...]` character class starting at i."""
    i += 1
    i += 1 if i < n and pattern[i] == "^" else 0
    i += 1 if i < n and pattern[i] == "]" else 0
    while i < n and pattern[i] != "]":
        i += 2 if pattern[i] == "\\" else 1
    return i + 1


def _is_quantified(pattern: str, j: int, n: int) -> bool:
    """Is index j the start of an unbounded quantifier (`*`, `+`, or open-ended `{m,}`)?"""
    return j < n and (
        pattern[j] in "*+" or (pattern[j] == "{" and _open_ended_brace(pattern, j, n)[0])
    )


def _mark_enclosing(stack: list[bool]) -> None:
    if stack:
        stack[-1] = True


# Each handler consumes the token at i and returns (is_catastrophic, next_index); it may mark the
# group on top of `stack` as "has an unbounded repeat inside". Chars with no handler advance by one.
def _h_escape(pattern, i, n, stack):
    return False, i + 2


def _h_class(pattern, i, n, stack):
    return False, _skip_char_class(pattern, i, n)


def _h_open(pattern, i, n, stack):
    stack.append(False)
    return False, i + 1


def _h_close(pattern, i, n, stack):
    body_has = stack.pop() if stack else False
    quantified = _is_quantified(pattern, i + 1, n)
    if quantified and body_has:
        return True, i + 1
    if body_has or quantified:
        _mark_enclosing(stack)
    return False, i + 1


def _h_repeat(pattern, i, n, stack):
    _mark_enclosing(stack)
    return False, i + 1


def _h_brace(pattern, i, n, stack):
    open_ended, nxt = _open_ended_brace(pattern, i, n)
    if open_ended:
        _mark_enclosing(stack)
    return False, nxt


_REGEX_HANDLERS = {
    "\\": _h_escape, "[": _h_class, "(": _h_open, ")": _h_close,
    "*": _h_repeat, "+": _h_repeat, "{": _h_brace,
}


def is_catastrophic_regex(pattern: str) -> bool:
    """True when `pattern` has a quantified group whose body itself repeats unboundedly
    (e.g. `(a+)+`, `([a-z]+)*`), the signature of exponential backtracking that can hang the
    engine. Only `*`, `+`, and open-ended `{m,}` count as unbounded; `{m,n}` and `{k}` do not."""
    stack: list[bool] = []
    i, n = 0, len(pattern)
    while i < n:
        handler = _REGEX_HANDLERS.get(pattern[i])
        if handler is None:
            i += 1
            continue
        catastrophic, i = handler(pattern, i, n, stack)
        if catastrophic:
            return True
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


# Prose/routing keys of a check block; everything else in a block is a flat param.
_RESERVED = frozenset({
    "name", "rule", "kind", "severity", "source", "because", "confidence",
    "applies_to", "id", "enabled",
})


@dataclass(frozen=True)
class _Param:
    """One parameter of a check kind: its evaluator key, the value shape shown in the prompt, an
    optional file-facing alias (the flat TOML surface), and whether it may be omitted."""

    key: str
    shape: str
    flat: str = ""  # file-facing key when it reads better than `key` (e.g. name -> tool)
    optional: bool = False


@dataclass(frozen=True)
class _KindSpec:
    params: tuple[_Param, ...]
    note: str = ""  # a trailing prompt note, e.g. what an expr evaluates over


# The one source of truth for every kind's parameters. The flat TOML surface (_PARAM_TO_FLAT),
# the prompt/placeholder text (PARAM_SPEC) and the per-kind validators below all read this table —
# so a param cannot be described one way and validated another.
_KIND_PARAMS: dict[str, _KindSpec] = {
    "contains": _KindSpec((_Param("values", "[str, ...]"),
                           _Param("mode", '"any|all"', optional=True))),
    "not_contains": _KindSpec((_Param("values", "[str, ...]"),
                               _Param("mode", '"any|all"', optional=True))),
    "regex": _KindSpec((_Param("pattern", "str"),)),
    "not_regex": _KindSpec((_Param("pattern", "str"),)),
    "json_schema": _KindSpec((_Param("schema", "{json schema object}"),)),
    "tool_called": _KindSpec((_Param("name", "str", flat="tool"),
                              _Param("arguments_match", '{field: value | {"regex": str}}',
                                     optional=True))),
    "tool_not_called": _KindSpec((_Param("name", "str", flat="tool"),)),
    "tool_order": _KindSpec((_Param("order", "[str, ...]"),)),
    "max_length": _KindSpec((_Param("max", "int"),)),
    "min_length": _KindSpec((_Param("min", "int"),)),
    "no_pii": _KindSpec((_Param("kinds", '["email"|"phone"|"card", ...]',
                                flat="pii", optional=True),)),
    "expr": _KindSpec((_Param("expr", "str"),), note="simpleeval over output, tools, reference"),
    "judge": _KindSpec((_Param("rubric", "str"),
                        _Param("samples", "int", optional=True),
                        _Param("min_agreement", "float", optional=True))),
}

# Params whose file-facing key differs from the evaluator key — derived from the one table.
_PARAM_TO_FLAT = {p.key: p.flat
                  for spec in _KIND_PARAMS.values() for p in spec.params if p.flat}
_FLAT_TO_PARAM = {v: k for k, v in _PARAM_TO_FLAT.items()}


def _spec_str(spec: _KindSpec) -> str:
    body = ", ".join(f'"{p.key}": {p.shape}' + ("?" if p.optional else "") for p in spec.params)
    return "{" + body + "}" + (f"  # {spec.note}" if spec.note else "")


# The prompt/placeholder text for each kind (LLM proposals, the interviewer, the new-check form),
# derived from the same table the validators read.
PARAM_SPEC = {kind: _spec_str(spec) for kind, spec in _KIND_PARAMS.items()}


@dataclass
class Check:
    kind: str
    params: dict = field(default_factory=dict)
    id: str = ""
    name: str = ""
    applies_to: str = "final"
    severity: str = "hard"
    rule: str = ""
    because: str = ""
    source: str = "manual"
    confidence: float | None = None

    _FIELDS = ("kind", "params", "id", "name", "applies_to", "severity",
               "rule", "because", "source", "confidence")

    @classmethod
    def from_dict(cls, data: dict) -> Check:
        kwargs = {k: data[k] for k in cls._FIELDS if k in data and data[k] is not None}
        if "because" not in kwargs and data.get("rationale"):  # v1 called it rationale
            kwargs["because"] = data["rationale"]
        if "kind" not in kwargs:
            raise ValueError("check dict is missing 'kind'")
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def from_toml(cls, block: dict) -> Check:
        """Read one flat `[[...check]]` block: prose keys plus per-kind param keys.

        The `name` is the check's stable identity; it becomes `id` so the evaluator keys
        results by name.
        """
        params = {_FLAT_TO_PARAM.get(k, k): v for k, v in block.items() if k not in _RESERVED}
        prose = {k: block[k] for k in cls._FIELDS if k in block}
        check = cls.from_dict({**prose, "params": params})
        check.id = check.id or check.name
        return check

    def to_toml(self) -> dict:
        """A flat block: name, rule, kind, params, then severity/source/because/confidence."""
        block: dict = {"name": self.name or self.kind}
        if self.rule:
            block["rule"] = self.rule
        block["kind"] = self.kind
        for pk, pv in (self.params or {}).items():
            block[_PARAM_TO_FLAT.get(pk, pk)] = pv
        block["severity"] = self.severity
        block["source"] = self.source
        if self.because:
            block["because"] = self.because
        if self.confidence is not None:
            block["confidence"] = self.confidence
        if self.applies_to != "final":
            block["applies_to"] = self.applies_to
        return block

    def validate(self) -> None:
        validate_params(self.kind, self.params)
        if self.applies_to not in APPLIES_TO:
            raise ValueError(f"applies_to must be one of {sorted(APPLIES_TO)}")
        if self.severity not in SEVERITIES:
            raise ValueError("severity must be 'hard' or 'soft'")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


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


def _v_contains(params: dict) -> None:
    _require_str_list(params, "values")
    if params.get("mode", "any") not in ("any", "all"):
        raise ValueError("'mode' must be 'any' or 'all'")


def _v_regex(params: dict) -> None:
    _require_str(params, "pattern")
    _require_safe_regex(params["pattern"])


def _v_json_schema(params: dict) -> None:
    if not isinstance(params.get("schema"), dict):
        raise ValueError("'schema' must be a JSON object")
    import jsonschema

    try:
        jsonschema.Draft202012Validator.check_schema(params["schema"])
    except jsonschema.exceptions.SchemaError as exc:
        raise ValueError(f"invalid JSON schema: {exc.message}") from exc


def _v_tool_called(params: dict) -> None:
    _require_str(params, "name")
    matcher = params.get("arguments_match")
    if matcher is None:
        return
    if not isinstance(matcher, dict):
        raise ValueError("'arguments_match' must be an object of field matchers")
    for expected in matcher.values():
        if isinstance(expected, dict) and "regex" in expected:
            _require_safe_regex(str(expected["regex"]))


def _v_judge(params: dict) -> None:
    _require_str(params, "rubric")
    if "samples" in params:
        s = params["samples"]
        if not isinstance(s, int) or isinstance(s, bool) or s < 1:
            raise ValueError("'samples' must be a positive integer")
    if "min_agreement" in params:
        ma = params["min_agreement"]
        if isinstance(ma, bool) or not isinstance(ma, (int, float)) or not 0.0 <= ma <= 1.0:
            raise ValueError("'min_agreement' must be a number between 0 and 1")


def _v_no_pii(params: dict) -> None:
    kinds = params.get("kinds")
    if kinds is not None and not (
        isinstance(kinds, list) and all(k in ("email", "phone", "card") for k in kinds)
    ):
        raise ValueError("'kinds' must be a list drawn from email/phone/card")


_VALIDATORS = {
    "contains": _v_contains,
    "not_contains": _v_contains,
    "regex": _v_regex,
    "not_regex": _v_regex,
    "json_schema": _v_json_schema,
    "tool_called": _v_tool_called,
    "tool_not_called": lambda p: _require_str(p, "name"),
    "tool_order": lambda p: _require_str_list(p, "order"),
    "max_length": lambda p: _require_int(p, "max"),
    "min_length": lambda p: _require_int(p, "min"),
    "no_pii": _v_no_pii,
    "expr": lambda p: _require_str(p, "expr"),
    "judge": _v_judge,
}

# One kind, one row in each table: the enum, its parameters, and its validator stay in lockstep.
assert set(_KIND_PARAMS) == set(_VALIDATORS) == KINDS, "check-kind tables are out of sync"


def validate_params(kind: str, params: dict) -> None:
    """Raise ValueError with a clear message if `params` is wrong for `kind`."""
    if kind not in KINDS:
        raise ValueError(f"unknown check kind {kind!r}; valid kinds: {sorted(KINDS)}")
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    _VALIDATORS[kind](params)
