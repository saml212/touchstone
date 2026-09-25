"""Cross-call minted-id detection fed into the simulator prompt."""

from touchstone.survey.minted_ids import minted_id_bindings, minted_ids_text
from touchstone.survey.simulate import _prompt

SERVICE = {"name": "booking", "kind": "http", "base_url_env": "BOOK_URL",
           "base_url_default": None, "calls": []}
TOOLS = [{"name": "save_passenger", "import_path": "t:save_passenger", "file": "t.py"},
         {"name": "book_flight", "import_path": "t:book_flight", "file": "t.py"}]


def test_binding_for_id_returned_then_referenced():
    examples = [
        {"tool": "save_passenger", "arguments": {"name": "Ada"},
         "returned": {"passenger_id": "PAX-9", "name": "Ada"}},
        {"tool": "book_flight", "arguments": {"passenger_id": "PAX-9", "flight": "AA1"},
         "returned": {"booking_id": "BK-1"}},
    ]
    assert minted_id_bindings(examples) == [
        {"created_by": "save_passenger", "arguments": {"name": "Ada"}, "id": "PAX-9"}]


def test_id_never_referenced_later_is_not_a_binding():
    # An id returned by the LAST call (or never passed to a later call) is not cross-call.
    examples = [{"tool": "book_flight", "arguments": {"passenger_id": "PAX-9"},
                 "returned": {"booking_id": "BK-1"}}]
    assert minted_id_bindings(examples) == []


def test_bools_are_not_ids():
    examples = [{"tool": "a", "arguments": {}, "returned": {"ok": True}},
                {"tool": "b", "arguments": {"ok": True}, "returned": {}}]
    assert minted_id_bindings(examples) == []


def test_minted_ids_text_none():
    assert minted_ids_text([]) == "(no cross-call ids detected)"


def test_prompt_includes_cross_call_section(tmp_path):
    examples = [
        {"tool": "save_passenger", "arguments": {"name": "Ada"},
         "returned": {"passenger_id": "PAX-9"}},
        {"tool": "book_flight", "arguments": {"passenger_id": "PAX-9"}, "returned": {"ok": 1}},
    ]
    text = _prompt(tmp_path, SERVICE, TOOLS, examples)
    assert "Cross-call ids" in text and "PAX-9" in text
    assert "return this EXACT id" in text  # the deterministic-id instruction reached the prompt
