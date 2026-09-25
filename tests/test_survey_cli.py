


def test_rebaseline_forces_only_the_baseline_step(monkeypatch):
    from touchstone.survey import survey as survey_mod

    seen = {}
    monkeypatch.setattr(survey_mod, "run_gate",
                        lambda *a, **k: {"gated": ["t"], "force": seen.setdefault("gate", a[3])})
    monkeypatch.setattr(survey_mod, "run_baseline",
                        lambda repo, env, settings, force, skip: seen.setdefault("base", force))
    survey_mod._gate_and_baseline("repo", {}, None, False, False, False, rebaseline=True)
    assert seen["base"] is True and seen["gate"] is False
