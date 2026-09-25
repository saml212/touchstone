"""Rewrite a customer's hardcoded service host to a local simulator, without touching their code.

A tool whose base URL is a compile-time constant (`http://api.weatherapi.com/v1/current.json`)
cannot be repointed with an env var. Instead this monkeypatches the two common HTTP client
entrypoints — `requests.Session.request` and httpx's `Client.send` / `AsyncClient.send` — so any
outbound URL whose host matches a key in the map has its scheme+host(+port) rewritten to the
simulator's loopback URL, path and query untouched.

The map is host -> simulator base url (e.g. `{"api.weatherapi.com": "http://127.0.0.1:8712"}`), read
from the `TOUCHSTONE_SIMULATORS` env var (JSON) by `install_from_env()`. It is installed in the
fidelity-replay subprocess, in the environment image (via `_touchstone/sitecustomize.py`), and while
recording a target against a fake. Idempotent, and a no-op when the map is empty or neither library
is importable — then the constant-base-URL service stays flagged rather than risk the real host.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlsplit, urlunsplit

_MAPPING: dict[str, str] = {}
_PATCHED: set[str] = set()


def _rewrite_url(url: str, mapping: dict[str, str]) -> str | None:
    """Return `url` with its scheme+netloc swapped for the simulator base when its host is mapped,
    else None. Path, query, and fragment are preserved exactly."""
    parts = urlsplit(url)
    base = mapping.get((parts.hostname or "").lower())
    if not base:
        return None
    sim = urlsplit(base)
    return urlunsplit((sim.scheme or parts.scheme, sim.netloc, parts.path, parts.query,
                       parts.fragment))


def _patch_requests() -> bool:
    try:
        import requests
    except ImportError:
        return False
    if "requests" not in _PATCHED:
        original = requests.Session.request

        def request(self, method, url, *args, **kwargs):
            return original(self, method, _rewrite_url(url, _MAPPING) or url, *args, **kwargs)

        requests.Session.request = request
        _PATCHED.add("requests")
    return True


def _rewrite_httpx_request(request) -> None:
    import httpx

    new = _rewrite_url(str(request.url), _MAPPING)
    if new:
        request.url = httpx.URL(new)


def _wrap_httpx_sync(client_cls) -> None:
    original = client_cls.send

    def send(self, request, *args, **kwargs):
        _rewrite_httpx_request(request)
        return original(self, request, *args, **kwargs)

    client_cls.send = send


def _wrap_httpx_async(client_cls) -> None:
    original = client_cls.send

    async def send(self, request, *args, **kwargs):
        _rewrite_httpx_request(request)
        return await original(self, request, *args, **kwargs)

    client_cls.send = send


def _patch_httpx() -> bool:
    try:
        import httpx
    except ImportError:
        return False
    if "httpx" not in _PATCHED:
        _wrap_httpx_sync(httpx.Client)
        _wrap_httpx_async(httpx.AsyncClient)
        _PATCHED.add("httpx")
    return True


def install(mapping: dict[str, str]) -> list[str]:
    """Point every mapped host at its simulator. Returns the libraries patched (once a process)."""
    _MAPPING.clear()
    _MAPPING.update({str(h).lower(): u for h, u in (mapping or {}).items() if h and u})
    if not _MAPPING:
        return []
    return [name for name, patch in (("requests", _patch_requests), ("httpx", _patch_httpx))
            if patch()]


def install_from_env() -> list[str]:
    """Install the shim from `TOUCHSTONE_SIMULATORS` (a JSON host->base_url object). No-op if unset
    or malformed — a missing shim leaves the service flagged, never silently hits the real host."""
    raw = os.environ.get("TOUCHSTONE_SIMULATORS")
    if not raw:
        return []
    try:
        mapping = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    return install(mapping) if isinstance(mapping, dict) else []
