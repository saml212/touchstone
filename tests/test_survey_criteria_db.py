"""A document-store db table (id, doc) yields json_extract criteria per changed field, not a
check on the whole JSON blob — criteria read `SELECT json_extract(doc,'$.status') ...`."""

import json

from touchstone.survey.criteria import derive_criteria
from touchstone.survey.recordings import ToolEvent

WORLD = {"name": "world", "kind": "db", "base_url_env": None, "base_url_default": ":memory:"}
MAP = {"tools": [{"name": "exchange", "import_path": "t:exchange", "calls": ["world"]}],
       "services": [WORLD]}


def _doc_table(status):
    return {"pk": "id", "rows": [{"id": "#W1", "doc": json.dumps({"status": status, "total": 5})}]}


def test_document_store_change_becomes_json_extract_criteria():
    effect = {"initial": {"world": {"orders": _doc_table("pending")}},
              "final": {"world": {"orders": _doc_table("delivered")}}, "replayed": []}
    calls = [ToolEvent(tool="exchange", arguments={"order_id": "#W1"},
                       output={"ok": True}, episode="e1")]
    state, _tool, literals = derive_criteria(effect, [WORLD], MAP, calls)
    calls_text = [c for c, _ in state]
    assert any("json_extract(doc,'$.status')" in c and "'delivered'" in c
               for c in calls_text), state
    # the unchanged field is not graded, and the whole-blob doc column is never asserted
    assert not any("$.total" in c for c in calls_text)
    assert not any("SELECT doc FROM" in c for c in calls_text)
    assert "#W1" in literals


def test_document_store_ignores_unreferenced_row():
    # a changed doc the agent never named is not graded (coupled to the seed, not the effect)
    effect = {"initial": {"world": {"orders": _doc_table("pending")}},
              "final": {"world": {"orders": _doc_table("delivered")}}, "replayed": []}
    calls = [ToolEvent(tool="exchange", arguments={"order_id": "#W9"},
                       output={"ok": True}, episode="e1")]
    state, _tool, _literals = derive_criteria(effect, [WORLD], MAP, calls)
    assert state == []
