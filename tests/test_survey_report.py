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
