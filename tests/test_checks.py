import pytest

from touchstone.checks import (
    Check,
    CheckResult,
    Target,
    evaluate,
    find_pii,
    passes,
    validate_params,
)
from touchstone.checks.run import _luhn


def _one(kind, params, output="", tool_calls=None, reference=None, provider=None):
    check = Check(kind=kind, params=params, id="x")
    target = Target(output_text=output, tool_calls=tool_calls or [], reference=reference)
    return evaluate([check], target, judge_provider=provider)[0]


# ---- contains / not_contains -----------------------------------------------


def test_contains_any_and_all():
    assert _one("contains", {"values": ["cat", "dog"]}, "a cat").passed is True
    assert _one("contains", {"values": ["cat", "dog"], "mode": "all"}, "a cat").passed is False
    assert _one("contains", {"values": ["cat", "dog"], "mode": "all"}, "cat dog").passed is True


def test_contains_case_insensitive_by_default():
    assert _one("contains", {"values": ["HELLO"]}, "hello world").passed is True
    assert _one("contains", {"values": ["HELLO"], "case_sensitive": True}, "hello").passed is False


def test_contains_unicode():
    assert _one("contains", {"values": ["café"]}, "un café au lait").passed is True


def test_not_contains():
    assert _one("not_contains", {"values": ["error"]}, "all good").passed is True
    assert _one("not_contains", {"values": ["error"]}, "an error").passed is False


def test_contains_empty_output():
    assert _one("contains", {"values": ["x"]}, "").passed is False
    assert _one("not_contains", {"values": ["x"]}, "").passed is True


# ---- regex -----------------------------------------------------------------


def test_regex_and_not_regex():
    assert _one("regex", {"pattern": r"\d{3}"}, "order 123").passed is True
    assert _one("regex", {"pattern": r"\d{3}"}, "no digits").passed is False
    assert _one("not_regex", {"pattern": r"\d{3}"}, "no digits").passed is True


# ---- json_schema -----------------------------------------------------------


def test_json_schema_valid_invalid_and_nonjson():
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "number"}}}
    assert _one("json_schema", {"schema": schema}, '{"a": 1}').passed is True
    assert _one("json_schema", {"schema": schema}, '{"a": "x"}').passed is False
    r = _one("json_schema", {"schema": schema}, "not json")
    assert r.passed is False and "not valid JSON" in r.evidence


# ---- tool checks -----------------------------------------------------------


def test_tool_called_plain():
    calls = [{"name": "refund", "arguments": "{}"}]
    assert _one("tool_called", {"name": "refund"}, tool_calls=calls).passed is True
    assert _one("tool_called", {"name": "escalate"}, tool_calls=calls).passed is False


def test_tool_called_arguments_match_exact_and_regex():
    calls = [{"name": "refund", "arguments": '{"amount": 5, "note": "ORD-42"}'}]
    assert _one("tool_called", {"name": "refund", "arguments_match": {"amount": 5}},
                tool_calls=calls).passed is True
    assert _one("tool_called", {"name": "refund", "arguments_match": {"amount": 9}},
                tool_calls=calls).passed is False
    assert _one("tool_called",
                {"name": "refund", "arguments_match": {"note": {"regex": r"ORD-\d+"}}},
                tool_calls=calls).passed is True


def test_tool_called_non_json_arguments_matched_against_raw():
    calls = [{"name": "search", "arguments": "query=widgets&limit=5"}]
    assert _one("tool_called",
                {"name": "search", "arguments_match": {"query": {"regex": "widgets"}}},
                tool_calls=calls).passed is True
    assert _one("tool_called", {"name": "search", "arguments_match": {"limit": "5"}},
                tool_calls=calls).passed is True
    assert _one("tool_called", {"name": "search", "arguments_match": {"limit": "99"}},
                tool_calls=calls).passed is False


def test_tool_not_called():
    calls = [{"name": "refund", "arguments": "{}"}]
    assert _one("tool_not_called", {"name": "escalate"}, tool_calls=calls).passed is True
    assert _one("tool_not_called", {"name": "refund"}, tool_calls=calls).passed is False


def test_tool_order_subsequence():
    calls = [{"name": "a"}, {"name": "x"}, {"name": "b"}]
    assert _one("tool_order", {"order": ["a", "b"]}, tool_calls=calls).passed is True
    assert _one("tool_order", {"order": ["b", "a"]}, tool_calls=calls).passed is False


# ---- length ----------------------------------------------------------------


def test_length_checks():
    assert _one("max_length", {"max": 5}, "hi").passed is True
    assert _one("max_length", {"max": 5}, "way too long").passed is False
    assert _one("min_length", {"min": 3}, "hey").passed is True
    assert _one("min_length", {"min": 3}, "no").passed is False


# ---- no_pii + Luhn ---------------------------------------------------------


def test_no_pii_clean():
    assert _one("no_pii", {}, "the weather is nice").passed is True


def test_no_pii_flags_email_and_valid_card():
    assert _one("no_pii", {}, "reach me at a@b.com").passed is False
    assert _one("no_pii", {}, "card 4242 4242 4242 4242").passed is False


def test_luhn_false_positive_not_flagged():
    assert _luhn("4242424242424242") is True
    assert _luhn("1234567812345678") is False
    assert _one("no_pii", {}, "id 1234567812345678").passed is True  # fails Luhn -> not a card
    assert find_pii("id 1234567812345678")["card"] == []
    assert find_pii("pay 4242424242424242")["card"] == ["4242424242424242"]


def test_no_pii_kinds_filter():
    # only look for cards; an email present should not fail the check
    assert _one("no_pii", {"kinds": ["card"]}, "mail a@b.com").passed is True


def test_no_pii_phone():
    assert _one("no_pii", {}, "call +14155552671 now").passed is False


# ---- expr ------------------------------------------------------------------


def test_expr_true_false():
    assert _one("expr", {"expr": "len(output) > 3"}, "hello").passed is True
    assert _one("expr", {"expr": "len(output) > 30"}, "hello").passed is False


def test_expr_uses_tools_and_reference():
    calls = [{"name": "refund", "arguments": "{}"}]
    r = _one("expr", {"expr": "len(tools) == 1"}, tool_calls=calls)
    assert r.passed is True


def test_expr_sandbox_refuses_import():
    r = _one("expr", {"expr": '__import__("os").system("echo hi")'}, "x")
    assert r.passed is None
    assert "error" in r.evidence.lower()


def test_expr_refuses_attribute_access():
    r = _one("expr", {"expr": "output.__class__"}, "x")
    assert r.passed is None


def test_expr_bad_expression_is_none_not_crash():
    r = _one("expr", {"expr": "output +"}, "x")
    assert r.passed is None


# ---- judge -----------------------------------------------------------------


class _StubProvider:
    def __init__(self, content):
        self.content = content

    def chat(self, messages, tools=None, json=False, timeout=None):
        from touchstone.llm import Reply

        return Reply(content=self.content)


def test_judge_pass_and_fail():
    p = _StubProvider('{"pass": true, "reason": "polite"}')
    assert _one("judge", {"rubric": "polite?"}, "hello", provider=p).passed is True
    p2 = _StubProvider('The verdict: {"pass": false, "reason": "rude"}')
    assert _one("judge", {"rubric": "polite?"}, "shut up", provider=p2).passed is False


def test_judge_malformed_reply_is_none():
    r = _one("judge", {"rubric": "x"}, "hi", provider=_StubProvider("totally not json"))
    assert r.passed is None
    assert "not JSON" in r.evidence


def test_judge_skipped_without_provider():
    r = _one("judge", {"rubric": "x"}, "hi", provider=None)
    assert r.passed is None
    assert "skipped" in r.evidence


def test_judge_provider_exception_is_none():
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("network down")

    r = _one("judge", {"rubric": "x"}, "hi", provider=Boom())
    assert r.passed is None


# ---- gating (hard vs soft) -------------------------------------------------


def test_passes_hard_soft_gating():
    hard = Check(kind="contains", params={"values": ["ok"]}, id="h", severity="hard")
    soft = Check(kind="contains", params={"values": ["bonus"]}, id="s", severity="soft")
    target = Target(output_text="ok")
    results = evaluate([hard, soft], target)
    assert passes(results, [hard, soft]) is True  # soft failing does not gate


def test_passes_fails_on_hard_failure():
    hard = Check(kind="contains", params={"values": ["missing"]}, id="h", severity="hard")
    results = evaluate([hard], Target(output_text="nope"))
    assert passes(results, [hard]) is False


def test_passes_fails_on_errored_hard_check():
    hard = Check(kind="expr", params={"expr": "output +"}, id="h", severity="hard")
    results = evaluate([hard], Target(output_text="x"))
    assert results[0].passed is None
    assert passes(results, [hard]) is False  # a hard check that errored fails the task


def test_errored_soft_check_does_not_gate():
    soft = Check(kind="expr", params={"expr": "output +"}, id="s", severity="soft")
    results = evaluate([soft], Target(output_text="x"))
    assert passes(results, [soft]) is True


# ---- dsl round-trip + validation -------------------------------------------


def test_check_from_dict_and_to_dict():
    row = {"kind": "contains", "params": {"values": ["x"]}, "id": "c1", "name": "n",
           "applies_to": "final", "severity": "soft", "source": "mined", "created_at": "t"}
    check = Check.from_dict(row)
    assert check.kind == "contains" and check.severity == "soft" and check.id == "c1"
    assert check.to_dict()["params"] == {"values": ["x"]}


def test_from_dict_requires_kind():
    with pytest.raises(ValueError, match="missing 'kind'"):
        Check.from_dict({"params": {}})


@pytest.mark.parametrize(
    "kind,params",
    [
        ("contains", {"values": []}),
        ("contains", {"values": "x"}),
        ("contains", {"values": ["x"], "mode": "most"}),
        ("regex", {"pattern": "["}),
        ("regex", {}),
        ("json_schema", {"schema": "x"}),
        ("tool_called", {}),
        ("tool_called", {"name": "t", "arguments_match": "x"}),
        ("tool_order", {"order": []}),
        ("max_length", {"max": "5"}),
        ("min_length", {}),
        ("no_pii", {"kinds": ["ssn"]}),
        ("expr", {}),
        ("judge", {}),
        ("bogus", {}),
    ],
)
def test_validate_params_rejects_bad(kind, params):
    with pytest.raises(ValueError):
        validate_params(kind, params)


def test_validate_params_accepts_good():
    validate_params("contains", {"values": ["x"], "mode": "all"})
    validate_params("tool_called", {"name": "t", "arguments_match": {"a": {"regex": "b"}}})
    validate_params("no_pii", {"kinds": ["email", "card"]})
    validate_params("max_length", {"max": 10})


def test_check_result_shape():
    r = CheckResult(check_id="c", passed=None, evidence="e")
    assert (r.check_id, r.passed, r.evidence) == ("c", None, "e")


# ---- catastrophic regex guard ----------------------------------------------


def _eval_within(check, target, seconds=3.0):
    import threading

    out = []
    t = threading.Thread(target=lambda: out.append(evaluate([check], target)), daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        pytest.fail("regex evaluation did not terminate (catastrophic backtracking)")
    return out[0][0]


@pytest.mark.parametrize("pattern", ["(a+)+$", "(a*)*$", "([a-z]+)+$", r"(\w+)+$", "((a+))+$"])
def test_validate_rejects_catastrophic_regex(pattern):
    with pytest.raises(ValueError, match="catastrophic|nested quantifier"):
        validate_params("regex", {"pattern": pattern})
    with pytest.raises(ValueError):
        validate_params("not_regex", {"pattern": pattern})


def test_validate_allows_safe_repetition_patterns():
    validate_params("regex", {"pattern": r"(\d{3})+"})
    validate_params("regex", {"pattern": r"(ab){2,5}"})
    validate_params("regex", {"pattern": r"[a-z]+@[a-z]+\.[a-z]+"})


def test_catastrophic_regex_does_not_hang_at_eval():
    check = Check(kind="regex", params={"pattern": "(a+)+$"}, id="x")
    target = Target(output_text="a" * 60 + "!")
    result = _eval_within(check, target)
    assert result.passed is None
    assert "regex" in result.evidence.lower()


def test_validate_rejects_catastrophic_arguments_match_regex():
    with pytest.raises(ValueError):
        validate_params("tool_called", {"name": "t", "arguments_match": {"x": {"regex": "(a+)+$"}}})
