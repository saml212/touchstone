from touchstone.ids import new_id


def test_ids_are_26_char_and_unique_and_sorted_under_load():
    ids = [new_id() for _ in range(10_000)]
    assert all(len(i) == 26 for i in ids)
    assert len(set(ids)) == 10_000, "ids must be unique"
    assert ids == sorted(ids), "ids must be monotonically sortable in creation order"


def test_ids_use_crockford_alphabet():
    alphabet = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
    assert set(new_id()) <= alphabet
