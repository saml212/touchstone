from touchstone.survey.fidelity import _compare, _mask


def test_volatile_ids_are_masked_so_fresh_ids_still_match():
    ok, masked = _compare({"ticket": "T-7", "order_id": "A1", "ok": True},
                          {"ticket": "T-1", "order_id": "A9", "ok": True})
    assert ok
    assert "ticket" in masked and "order_id" in masked


def test_real_difference_is_not_masked_away():
    ok, _ = _compare({"status": "shipped"}, {"status": "processing"})
    assert not ok


def test_iso_timestamps_masked():
    m: set = set()
    out = _mask({"created_at": "2024-09-25T10:00:00Z", "note": "2024-09-25T11:00:00Z"}, m)
    assert out["created_at"] == "<masked>"  # by key
    assert out["note"] == "<ts>"  # by ISO value
    assert "<iso-timestamp>" in m


def test_nested_masking():
    ok, _ = _compare({"data": {"id": 1, "v": 5}}, {"data": {"id": 999, "v": 5}})
    assert ok


def test_constant_base_url_service_is_flagged_not_replayed(tmp_path):
    # Attack (stage-7 survey): a tool whose base URL is a hardcoded constant (no base_url_env in
    # the map) cannot be pointed at the simulator. measure_service must flag it (below threshold)
    # with an honest reason and NEVER start a replay against the real (possibly production) service.
    from touchstone.survey.fidelity import measure_service
    from touchstone.survey.recordings import ToolEvent
    from touchstone.survey.scrub import Scrubber

    class _S:
        survey_fidelity_threshold = 0.8
        survey_python = None

    calls = [ToolEvent(tool="charge", arguments={"id": "A1"}, output={"ok": True}, episode="e1")]
    ctx = {"base_url_env": None, "tools": {"charge": "mod:charge"}}
    # sim_dir has no app.py; if the guard failed to short-circuit, starting the sim would error out
    # differently — the honest no-redirect reason proves it never tried.
    result = measure_service(tmp_path, tmp_path, calls, ctx, _S(), Scrubber())
    assert result["score"] == 0.0
    assert result["score"] < result["threshold"]
    assert "constant" in result["failures"][0]["error"]


def test_scalar_list_compared_unordered():
    # A tool that sorts ids returns them in a different order once scrubbing renumbers the ids;
    # a set of the same ids is still a faithful reproduction.
    ok, masked = _compare({"exchange_items": ["b2", "a1"]}, {"exchange_items": ["a1", "b2"]})
    assert ok
    assert "<unordered-list>" in masked


def test_structured_list_keeps_order():
    # A list of objects (a sequence) is still order-sensitive.
    ok, _ = _compare({"steps": [{"n": 1}, {"n": 2}]}, {"steps": [{"n": 2}, {"n": 1}]})
    assert not ok
