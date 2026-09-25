"""Group recorded episodes by job-to-be-done with a scripted provider."""

import json

from touchstone import store
from touchstone.survey.group import group_episodes, variant_episodes
from touchstone.survey.provider import ScriptedSurveyProvider
from touchstone.survey.recordings import ToolEvent, tool_events
from touchstone.survey.scrub import Scrubber


def _model_span(ep_id, user=None, tool=None, args="{}", tool_result=None, call_id="c1"):
    inp = {"messages": []}
    if user is not None:
        inp["messages"].append({"role": "user", "content": user})
    if tool_result is not None:
        inp["messages"].append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
    calls = [{"id": call_id, "name": tool, "arguments": args}] if tool else []
    return store.Span(episode_id=ep_id, kind="model", name="gpt", input=inp,
                      output={"message": {"content": "", "tool_calls": calls}})


def _episode(conn, name, kind, outcome, user, tool=None, args="{}", result=None):
    ep = store.insert_episode(conn, store.Episode(id=name, name=name, meta={"kind": kind},
                                                  outcome_label=outcome))
    store.insert_span(conn, _model_span(ep.id, user=user, tool=tool, args=args))
    if result is not None:
        store.insert_span(conn, _model_span(ep.id, tool_result=result))
    return ep


def _seed(conn):
    _episode(conn, "e1", "status", "resolved", "Where is order B1?", "order_status",
             '{"order_id": "B1"}', '{"id": "B1", "status": "shipped"}')
    _episode(conn, "e2", "status", "resolved", "Where is order B2?", "order_status",
             '{"order_id": "B2"}', '{"id": "B2", "status": "delivered"}')
    _episode(conn, "e3", "refund", "resolved", "Refund order B3", "refund",
             '{"order_id": "B3", "amount": 5.0}', '{"ok": true}')


def _groups_response():
    return json.dumps({"groups": [
        {"label": "Check status", "slug": "check-status", "episodes": ["e1", "e2"]},
        {"label": "Refund", "slug": "refund", "episodes": ["e3"]},
    ]})


def test_group_happy_path(tmp_path, conn):
    _seed(conn)
    out = tmp_path / "touchstone"
    provider = ScriptedSurveyProvider([_groups_response()])
    data = group_episodes(conn, tool_events(conn), provider, tmp_path, out, Scrubber())
    slugs = [g["slug"] for g in data["groups"]]
    assert slugs == ["check-status", "refund"]
    assert data["groups"][0]["episodes"] == ["e1", "e2"]
    assert data["no_job"] == []
    assert (out / "groups.json").exists()


def test_group_schema_retry(tmp_path, conn):
    _seed(conn)
    out = tmp_path / "touchstone"
    bad = json.dumps({"groups": [{"label": "x"}]})  # missing slug + episodes
    provider = ScriptedSurveyProvider([bad, _groups_response()])
    data = group_episodes(conn, tool_events(conn), provider, tmp_path, out, Scrubber())
    assert len(provider.calls) == 2
    assert [g["slug"] for g in data["groups"]] == ["check-status", "refund"]


def test_group_no_job_bucket(tmp_path, conn):
    _seed(conn)
    # an episode with no tools and no outcome
    ep = store.insert_episode(conn, store.Episode(name="chatter"))
    store.insert_span(conn, _model_span(ep.id, user="just saying hi"))
    out = tmp_path / "touchstone"
    provider = ScriptedSurveyProvider([_groups_response()])
    data = group_episodes(conn, tool_events(conn), provider, tmp_path, out, Scrubber())
    assert data["no_job"] == [ep.id]
    # the no_job episode was not offered to the provider for grouping
    assert ep.id not in provider.calls[0]


def test_group_dedup_slug(tmp_path, conn):
    _seed(conn)
    out = tmp_path / "touchstone"
    dupe = json.dumps({"groups": [
        {"label": "A", "slug": "same", "episodes": ["e1"]},
        {"label": "B", "slug": "same", "episodes": ["e3"]},
    ]})
    provider = ScriptedSurveyProvider([dupe])
    data = group_episodes(conn, tool_events(conn), provider, tmp_path, out, Scrubber())
    assert sorted(g["slug"] for g in data["groups"]) == ["same", "same-2"]


def test_group_cached(tmp_path, conn):
    _seed(conn)
    out = tmp_path / "touchstone"
    provider = ScriptedSurveyProvider([_groups_response()])
    group_episodes(conn, tool_events(conn), provider, tmp_path, out, Scrubber())
    group_episodes(conn, tool_events(conn), provider, tmp_path, out, Scrubber())
    assert len(provider.calls) == 1  # second call reused groups.json


def test_group_scrubs_pii_in_listing(tmp_path, conn):
    ep = _episode(conn, "e4", "refund", "resolved",
                  "email me at jane@corp.com about order B9", "order_status",
                  '{"order_id": "B9"}', '{"id": "B9"}')
    out = tmp_path / "touchstone"
    provider = ScriptedSurveyProvider([json.dumps({"groups": [
        {"label": "R", "slug": "r", "episodes": [ep.id]}]})])
    group_episodes(conn, tool_events(conn), provider, tmp_path, out, Scrubber())
    assert "jane@corp.com" not in provider.calls[0]
    assert "example.invalid" in provider.calls[0]


def test_variant_episodes_caps_at_three():
    events_by_ep = {f"e{i}": [ToolEvent("order_status", {"order_id": f"B{i}"}, {}, f"e{i}"),
                              ToolEvent("refund", {"order_id": f"B{i}"}, {}, f"e{i}")]
                    for i in range(5)}
    group = {"episodes": list(events_by_ep)}
    resolved = set(events_by_ep)
    picked = variant_episodes(group, events_by_ep, resolved)
    assert len(picked) == 3


def test_variant_episodes_one_when_signatures_collide():
    # three episodes but identical seed signature -> only one variant
    events_by_ep = {f"e{i}": [ToolEvent("order_status", {"order_id": "SAME"}, {}, f"e{i}")]
                    for i in range(3)}
    group = {"episodes": list(events_by_ep)}
    picked = variant_episodes(group, events_by_ep, set(events_by_ep))
    assert len(picked) == 1


def test_variant_episodes_skips_unresolved():
    events_by_ep = {"e1": [ToolEvent("order_status", {"order_id": "B1"}, {}, "e1")]}
    picked = variant_episodes({"episodes": ["e1"]}, events_by_ep, resolved=set())
    assert picked == []
