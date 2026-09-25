"""The survey benchmark scores a surveyed target's outputs: fidelity, tasks, baseline, agreement."""

import json

from touchstone import store
from touchstone.survey.benchmark import render_markdown, score_target, survey_bench


def _task(out, kind, name, episodes, oracle=1.0, nop=0.0):
    d = out / kind / name
    d.mkdir(parents=True)
    eps = "".join(f'    "{e}",\n' for e in episodes)
    (d / "task.toml").write_text(
        f'[metadata.touchstone]\nepisodes = [\n{eps}]\noracle = {oracle}\nnop = {nop}\n',
        encoding="utf-8")


def _surveyed_target(repo):
    (repo / ".touchstone").mkdir(parents=True)
    conn = store.connect(str(repo / ".touchstone" / "touchstone.db"))
    e1 = store.insert_episode(conn, store.Episode(name="e1", outcome_score=1.0,
                                                  outcome_label="resolved"))
    e2 = store.insert_episode(conn, store.Episode(name="e2", outcome_score=1.0,
                                                  outcome_label="resolved"))
    conn.close()
    out = repo / "touchstone"
    (out).mkdir()
    (out / "fidelity.json").write_text(json.dumps({"orders": {"score": 1.0}}), encoding="utf-8")
    _task(out, "tasks", "t1", [e1.id])
    _task(out, "tasks", "t2", [e2.id])
    _task(out, "needs-review", "t3", [], oracle=0.0)  # written but not gated
    (out / "baseline.json").write_text(
        json.dumps({"pass_rates": {"t1": 1.0, "t2": 0.0}, "model": "openai/gpt-4o-mini"}),
        encoding="utf-8")
    return repo


def test_score_target_computes_the_four_metrics(tmp_path):
    repo = _surveyed_target(tmp_path / "target")
    row = score_target("t", repo)
    assert row["services"] == {"orders": 1.0}
    assert row["fidelity_min"] == 1.0
    assert row["written"] == 3 and row["gated"] == 2
    assert row["baseline_pass"] == 0.5  # mean of 1.0 and 0.0
    # both tasks come from resolved (recorded-passed) episodes; baseline passes t1, fails t2 -> 0.5
    assert row["agreement"] == 0.5


def test_agreement_is_none_without_a_baseline(tmp_path):
    repo = _surveyed_target(tmp_path / "target")
    (repo / "touchstone" / "baseline.json").unlink()
    row = score_target("t", repo)
    assert row["baseline_pass"] is None and row["agreement"] is None


def test_survey_bench_writes_json_and_markdown(tmp_path):
    repo = _surveyed_target(tmp_path / "target")
    config = tmp_path / "targets.toml"
    config.write_text(f'[[target]]\nname = "t"\nrepo = "{repo}"\nrun = "recorded x"\n',
                      encoding="utf-8")
    doc = survey_bench(config, tmp_path)
    assert len(doc["targets"]) == 1 and doc["targets"][0]["name"] == "t"
    bench = tmp_path / "benchmarks" / "survey"
    written = sorted(p.suffix for p in bench.iterdir() if p.suffix in (".json", ".md"))
    assert written == [".json", ".md"]
    md = next(bench.glob("*.md")).read_text()
    assert "| t |" in md and "orders 100%" in md


def test_render_markdown_dashes_missing_values():
    row = {"name": "x", "services": {}, "written": 0, "gated": 0,
           "baseline_pass": None, "agreement": None}
    md = render_markdown([row], "2026-09-25T00:00:00+00:00")
    assert "| x | — | 0/0 | — | — |" in md


def test_score_target_without_needs_review_dir(tmp_path):
    # A target where every task gated leaves no needs-review/ dir; counting must not crash.
    repo = _surveyed_target(tmp_path / "target")
    import shutil
    shutil.rmtree(repo / "touchstone" / "needs-review")
    row = score_target("t", repo)
    assert row["written"] == 2 and row["gated"] == 2
