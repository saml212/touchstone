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
