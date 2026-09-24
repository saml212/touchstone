"""Checks: list, kind catalog, create, patch, and one-off evaluation.

Create and patch validate through the DSL `Check` before touching the store, so an invalid kind or
params returns 422 with a one-sentence message instead of persisting garbage.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ... import store
from ...checks import Check as DslCheck
from ...checks import Target, coerce_tool_calls, evaluate
from ...checks.dsl import PARAM_SPEC
from ._deps import check_view, get_conn

router = APIRouter()


@router.get("/api/checks")
def list_checks(conn=Depends(get_conn)) -> dict:
    return {"checks": [check_view(c) for c in store.list_checks(conn)]}


@router.get("/api/checks/kinds")
def check_kinds() -> dict:
    return {"kinds": PARAM_SPEC}


@router.post("/api/checks")
def create_check(body: dict, conn=Depends(get_conn)) -> dict:
    kind = body.get("kind") or ""
    params = body.get("params") or {}
    severity = body.get("severity", "hard")
    applies_to = body.get("applies_to", "final")
    try:
        DslCheck(kind=kind, params=params, severity=severity, applies_to=applies_to).validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    check = store.insert_check(conn, store.Check(
        name=body.get("name") or kind, kind=kind, params=params,
        applies_to=applies_to, severity=severity, source="manual",
        rationale=body.get("rationale", ""), enabled=1 if body.get("enabled", True) else 0,
    ))
    return check_view(check)


@router.patch("/api/checks/{check_id}")
def patch_check(check_id: str, body: dict, conn=Depends(get_conn)) -> dict:
    check = store.get_check(conn, check_id)
    if check is None:
        raise HTTPException(404, f"no check with id {check_id}")
    fields: dict = {}
    if "name" in body:
        fields["name"] = body["name"]
    if "enabled" in body:
        fields["enabled"] = 1 if body["enabled"] else 0
    if "severity" in body:
        fields["severity"] = body["severity"]
    if "params" in body:
        fields["params"] = body["params"]
    severity = fields.get("severity", check.severity)
    params = fields.get("params", check.params)
    try:
        DslCheck(kind=check.kind, params=params, severity=severity,
                 applies_to=check.applies_to).validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store.update_check(conn, check_id, **fields)
    return check_view(store.get_check(conn, check_id))


@router.post("/api/checks/{check_id}/eval")
def eval_check(check_id: str, body: dict, request: Request, conn=Depends(get_conn)) -> dict:
    row = store.get_check(conn, check_id)
    if row is None:
        raise HTTPException(404, f"no check with id {check_id}")
    try:
        calls = coerce_tool_calls(body.get("tool_calls"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    target = Target(output_text=body.get("text", "") or "", tool_calls=calls)
    provider = _judge_provider(request) if row.kind == "judge" else None
    check = DslCheck.from_dict({
        "kind": row.kind, "params": row.params, "id": row.id, "name": row.name,
        "applies_to": row.applies_to, "severity": row.severity,
    })
    result = evaluate([check], target, judge_provider=provider)[0]
    return {"check_id": row.id, "passed": result.passed, "evidence": result.evidence}


def _judge_provider(request: Request):
    from ...llm import provider_from_spec

    settings = request.app.state.settings
    try:
        return provider_from_spec(settings.provider, settings)
    except Exception:  # a judge without a working provider degrades to 'skipped'
        return None
