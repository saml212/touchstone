from touchstone.survey.report import render_report, summary

MAP = {
    "entrypoints": ["agent.py:main"],
    "model_call": {"file": "agent.py", "line": 71, "sdk": "openai", "model_setting": "arg"},
    "tools": [
        {"name": "order_status", "import_path": "agent:order_status", "calls": ["orders"]},
        {"name": "summarize", "import_path": "agent:summarize", "calls": []},
    ],
    "services": [{"name": "orders", "kind": "http", "base_url_env": "ORDERS_URL",
                  "calls": [{"method": "GET", "path_template": "/orders/{id}"}]}],
    "sort": {"unmapped": ["ghost"], "unused": ["summarize"]},
}
FIDELITY = {"orders": {"calls": 12, "reproduced": 11, "score": 0.9167, "threshold": 0.8}}


def test_report_has_all_sections():
    md = render_report(MAP, FIDELITY)
    for heading in ("# Survey report", "## Map", "## Services", "## Simulators", "## Flags"):
        assert heading in md
    assert "order_status" in md and "agent:order_status" in md
    assert "0.92" in md  # fidelity rounded in the table
    assert "ghost" in md and "summarize" in md  # flags


def test_report_without_simulators():
    md = render_report({"tools": [], "services": [], "sort": {}}, {})
    assert "No network-crossing services" in md
    assert "None." in md  # no flags


def test_summary_line():
    assert summary(MAP, FIDELITY) == (
        "Mapped 2 tools, 1 service. Simulator orders: fidelity 0.92 (11/12 calls)")


def test_summary_singular_and_no_sims():
    one = {"tools": [{"name": "a", "calls": []}], "services": []}
    assert summary(one, {}) == "Mapped 1 tool, 0 services."


def test_report_tasks_and_needs_review_sections():
    tasks = {"written": ["a-1"], "reused": ["a-2"],
             "skipped": [{"episode": "e9", "reason": "sim could not reproduce"}]}
    gate = {"gated": ["a-1"], "needs_review": [{"name": "a-2", "failed_side": "nop",
                                               "oracle": 1.0, "nop": 1.0}]}
    groups = {"no_job": ["chatter-1"]}
    md = render_report(MAP, FIDELITY, tasks, gate, groups)
    assert "## Tasks" in md and "a-1" in md and "gated" in md
    assert "Skipped episodes" in md and "e9" in md
    assert "## Needs review" in md and "nop" in md
    assert "## Unclustered conversations" in md and "chatter-1" in md


def test_summary_with_task_clause():
    stats = {"tasks": 5, "conversations": 12, "gated": 4, "needs_review": 1}
    line = summary(MAP, FIDELITY, stats)
    assert line.endswith("Built 5 tasks from 12 conversations (4 gated, 1 needs review).")


def test_summary_gate_skipped_clause():
    stats = {"tasks": 1, "conversations": 3, "gate_skipped": True}
    line = summary(MAP, FIDELITY, stats)
    assert line.endswith("Built 1 task from 3 conversations (gate skipped).")
