"""POST /api/mine — propose checks and cut tasks. The mine pass blocks, so it runs in a thread."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from ... import store

router = APIRouter()


@router.post("/api/mine")
async def run_mine(body: dict, request: Request) -> dict:
    settings = request.app.state.settings
    code = body.get("code")
    no_llm = bool(body.get("no_llm"))
    limit = body.get("limit")
    provider_spec = body.get("provider") or settings.agent_provider
    return await asyncio.to_thread(_mine, settings, code, no_llm, limit, provider_spec)


def _mine(settings, code, no_llm, limit, provider_spec) -> dict:
    from ...mine import mine as run
    from ...mine import scan_codebase

    snippets = scan_codebase(code) if code else []
    provider = None
    if not no_llm:
        from ...llm import provider_from_spec

        try:
            provider = provider_from_spec(provider_spec, settings)
        except Exception:  # no key / binary: mine on statistics alone
            provider = None
    conn = store.connect(settings.db_path)
    try:
        summary = run(conn, settings.root, provider=provider, code_snippets=snippets,
                      no_llm=no_llm, limit=limit)
    except (OSError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        conn.close()
    return {
        "inserted": summary["inserted"],
        "stats": summary["stats"],
        "llm": summary["llm"],
        "tasks_cut": summary["tasks_cut"],
        "proposals": [
            {"kind": p.kind, "name": p.name, "severity": p.severity,
             "rationale": p.rationale, "support_count": p.support_count}
            for p in summary["proposals"]
        ],
    }
