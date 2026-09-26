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


def test_summary_names_why_gate_and_baseline_were_skipped():
    # A Docker-daemon failure moved every task to needs-review; the summary must say so, not just
    # "0 gated, 7 needs review" (the stranger's false-green class).
    stats = {"tasks": 7, "conversations": 12, "gated": 0, "needs_review": 7,
             "skipped_reason": "Docker daemon not running on local. Start Docker."}
    line = summary(MAP, FIDELITY, stats)
    assert line.endswith("Gate and baseline skipped: Docker daemon not running on local. "
                         "Start Docker.")


def test_gate_infra_note_only_fires_on_harbor_failures_with_nothing_gated():
    from touchstone.survey.survey import _gate_infra_note

    down = {"gated": [], "needs_review": [{"failed_side": "harbor", "reason": "Docker down.\nx"}]}
    assert _gate_infra_note(down) == "Docker down."
    # a real gate result (oracle/nop failures, some gated) is not an infra skip
    assert _gate_infra_note({"gated": ["t1"], "needs_review": []}) == ""
    assert _gate_infra_note({"gated": [], "needs_review": [{"failed_side": "oracle"}]}) == ""


def test_summary_baseline_sentence_is_the_design_line():
    stats = {"tasks": 7, "built_names": [f"t{i}" for i in range(7)],
             "conversations": 12, "gated": 7, "needs_review": 0}
    baseline = {"pass_rates": {f"t{i}": (1.0 if i < 5 else 0.0) for i in range(7)},
                "passed": [f"t{i}" for i in range(5)], "failed": ["t5", "t6"],
                "model": "openai/gpt-4o-mini", "mode": "packaged"}
    line = summary(MAP, FIDELITY, stats, baseline)
    assert line == ("Built 7 tasks from 12 conversations. Your current setup passes 5. "
                    "2 failures — walk through them? (touchstone review)")


def test_report_baseline_shows_mean_reward_and_check():
    baseline = {"pass_rates": {"a-1": 1.0, "a-2": 0.0}, "rewards": {"a-1": 1.0, "a-2": 0.75},
                "passed": ["a-1"], "failed": ["a-2"], "model": "openai/gpt-4o-mini",
                "mode": "packaged"}
    md = render_report(MAP, FIDELITY, baseline=baseline)
    assert "## Baseline" in md and "gpt-4o-mini" in md
    assert "100% ✓" in md  # a-1 fully passed
    assert "75%" in md and "75% ✓" not in md  # a-2 partial reward, no check
    from touchstone.survey.report import baseline_section
    assert "Failed:" in baseline_section({"error": "boom\nmore"})


# ---- first-five-minutes sentence: N built, K passed, U gated-but-not-run -------------------------

from touchstone.survey.report import first_five_sentence  # noqa: E402


def test_sentence_all_pass_when_baseline_covers_every_task():
    s = first_five_sentence(["t1", "t2"], 5, {"t1": 1.0, "t2": 1.0}, ["t1", "t2"])
    assert s == ("Built 2 tasks from 5 conversations. Your current setup passes all 2 — "
                 "spot-check a few? (touchstone review)")


def test_sentence_failures_when_baseline_covers_every_task():
    s = first_five_sentence(["t1", "t2", "t3"], 5, {"t1": 1.0, "t2": 0.0, "t3": 1.0}, ["t1", "t3"])
    assert s == ("Built 3 tasks from 5 conversations. Your current setup passes 2. "
                 "1 failure — walk through them? (touchstone review)")


def test_sentence_points_to_bench_when_tasks_were_not_run():
    # 7 built, the latest baseline only ran/passed 5 -> 2 not run yet; never "everything passes".
    ran = {f"t{i}": 1.0 for i in range(1, 6)}
    passed = list(ran)
    s = first_five_sentence([f"t{i}" for i in range(1, 8)], 12, ran, passed)
    assert s == ("Built 7 tasks from 12 conversations. Your current setup passes 5 of the 5 run; "
                 "2 not run yet — run touchstone bench.")
