from touchstone.survey.scrub import Scrubber


def test_email_scrubbed_and_stable():
    s = Scrubber()
    a = s.text("mail me at jane@shop.com please")
    b = s.text("again jane@shop.com")
    assert "jane@shop.com" not in a
    assert "@example.invalid" in a
    assert a.split("at ")[1].split(" ")[0] == b.split("again ")[1]  # same fake both times


def test_distinct_emails_get_distinct_fakes():
    s = Scrubber()
    out = s.text("a@x.com and b@y.com")
    fakes = [w for w in out.replace(" and ", " ").split() if "@" in w]
    assert len(set(fakes)) == 2


def test_card_and_phone_scrubbed():
    s = Scrubber()
    out = s.text("card 4111 1111 1111 1111 phone 415-555-0199")
    assert "4111" not in out
    assert "415-555-0199" not in out


def test_short_ids_and_dates_survive():
    s = Scrubber()
    out = s.text("order B6305 total 285.40 on 2024-09-25")
    assert "B6305" in out and "285.40" in out and "2024-09-25" in out


def test_names_scrubbed_case_insensitive_and_stable():
    s = Scrubber(names=["Jane Roe"])
    out = s.text("Jane Roe called; jane roe emailed")
    assert "Jane Roe" not in out and "jane roe" not in out
    # both spellings map to the same fake stem, so "Person" appears once per occurrence
    assert out.count("Person") == 2


def test_scrub_nested_structure():
    s = Scrubber()
    scrubbed = s.scrub({"to": "x@y.com", "items": ["a@b.com", 7], "n": 3})
    assert scrubbed["n"] == 3 and scrubbed["items"][1] == 7
    assert "@example.invalid" in scrubbed["to"]
    assert "@example.invalid" in scrubbed["items"][0]


def test_parenthesized_phone_is_scrubbed():
    # Attack (stage-7 survey PII): a common US format `(415) 555-0132` leaked entirely because the
    # phone pattern required the number to start with a digit and forbade `)` separators.
    s = Scrubber()
    out = s.text("call the customer at (415) 555-0132 today")
    assert "(415) 555-0132" not in out
    assert "415" not in out


def test_card_fake_is_not_rescrubbed_into_a_phone():
    # Attack: sequential passes let the phone pass re-match the card replacement
    # `4000-0000-0000-0001`, so scrubbed cards came out phone-shaped. A single pass keeps the
    # card fake intact.
    s = Scrubber()
    out = s.text("card 4111 1111 1111 1111")
    assert "4111" not in out
    assert "4000-0000-0000-0001" in out


def test_scrub_scrubs_dict_keys_consistently_with_values():
    # A store keyed by a 10-digit id: the key and the same id as a field value and as an argument
    # must all map to the SAME fake, or a replay can't resolve it against the scrubbed state.db.
    s = Scrubber()
    data = {"9523456873": {"product_id": "9523456873", "name": "Keyboard"}}
    scrubbed = s.scrub(data)
    key = next(iter(scrubbed))
    assert "9523456873" not in key
    assert scrubbed[key]["product_id"] == key  # key and field value share one fake
    arg = s.scrub({"product_id": "9523456873"})
    assert arg["product_id"] == key  # a tool argument maps to the same fake


def test_scrub_keeps_non_pii_keys():
    s = Scrubber()
    scrubbed = s.scrub({"user_id": "u1", "email": "x@y.com"})
    assert "user_id" in scrubbed and "email" in scrubbed  # ordinary field names untouched
