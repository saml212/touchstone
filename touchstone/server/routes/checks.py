"""Checks: the policies in checks.toml — list, kind catalog, create, patch, and one-off evaluation.

Create and patch validate through the DSL `Check` before writing checks.toml, so an invalid kind or
params returns 422 with a one-sentence message instead of persisting garbage.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ... import policies as policies_mod
from ...checks import Check as DslCheck
from ...checks import Target, coerce_tool_calls, evaluate
from ...checks.dsl import PARAM_SPEC
from ...policies import Policy
from ._deps import check_view, get_root

router = APIRouter()


def _policy_view(p: Policy) -> dict:
    return {**check_view(p.check), "enabled": p.enabled}


@router.get("/api/checks")
def list_checks(root=Depends(get_root)) -> dict:
    return {"checks": [_policy_view(p) for p in policies_mod.read_policies(root)]}


@router.get("/api/checks/kinds")
def check_kinds() -> dict:
    return {"kinds": PARAM_SPEC}


@router.post("/api/checks")
def create_check(body: dict, root=Depends(get_root)) -> dict:
    kind = body.get("kind") or ""
    params = body.get("params") or {}
    name = body.get("name") or kind
    try:
        check = DslCheck(kind=kind, params=params, name=name,
                         severity=body.get("severity", "hard"),
                         applies_to=body.get("applies_to", "final"),
                         because=body.get("because", ""), source="manual")
        check.id = check.name
        check.validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if policies_mod.get_policy(root, name) is not None:
        raise HTTPException(409, f"a check named {name!r} already exists")
    policy = Policy(check=check, enabled=bool(body.get("enabled", True)))
    policies_mod.add_policy(root, policy)
    return _policy_view(policy)


@router.patch("/api/checks/{name}")
def patch_check(name: str, body: dict, root=Depends(get_root)) -> dict:
    policies = policies_mod.read_policies(root)
    hit = next((p for p in policies if p.check.name == name), None)
    if hit is None:
        raise HTTPException(404, f"no check named {name!r}")
    if "enabled" in body:
        hit.enabled = bool(body["enabled"])
    if "severity" in body:
        hit.check.severity = body["severity"]
    if "params" in body:
        hit.check.params = body["params"]
    try:
        hit.check.validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    policies_mod.write_policies(root, policies)
    return _policy_view(hit)


@router.post("/api/checks/{name}/eval")
def eval_check(name: str, body: dict, request: Request, root=Depends(get_root)) -> dict:
    policy = policies_mod.get_policy(root, name)
    if policy is None:
        raise HTTPException(404, f"no check named {name!r}")
    try:
        calls = coerce_tool_calls(body.get("tool_calls"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    target = Target(output_text=body.get("text", "") or "", tool_calls=calls)
    provider = _judge_provider(request) if policy.check.kind == "judge" else None
    result = evaluate([policy.check], target, judge_provider=provider)[0]
    return {"name": name, "passed": result.passed, "evidence": result.evidence}


def _judge_provider(request: Request):
    from ...llm import provider_from_spec

    settings = request.app.state.settings
    try:
        return provider_from_spec(settings.provider, settings)
    except Exception:  # a judge without a working provider degrades to 'skipped'
        return None
