"""Shared HTTP plumbing for the httpx-based providers (openai_compat, anthropic).

Retries 429 / 5xx / transport failures up to `retries` times with exponential backoff, never
retrying other 4xx (auth/validation) errors, and raises `ProviderError` with a one-sentence
message that never contains a key.
"""

from __future__ import annotations

import time

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """A provider call failed; the message is safe to show and never contains a secret."""


def _should_retry(status: int) -> bool:
    return status in RETRY_STATUS or status >= 500


def _sleep(backoff: float, attempt: int) -> None:
    if backoff > 0:
        time.sleep(backoff * (2**attempt))


def _error_detail(resp: httpx.Response) -> str:
    """The API's own error reason (`error.message` when present), capped at 200 chars.

    Reads only the response body — never the request or a key.
    """
    try:
        data = resp.json()
    except (ValueError, UnicodeDecodeError):
        return (resp.text or "").strip()[:200]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:200]
        if isinstance(err, str) and err:
            return err[:200]
        if data.get("message"):
            return str(data["message"])[:200]
    return str(data)[:200]


def _http_error(label: str, resp: httpx.Response) -> ProviderError:
    detail = _error_detail(resp)
    suffix = f": {detail}" if detail else "."
    return ProviderError(f"{label} failed with HTTP {resp.status_code}{suffix}")


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict,
    body: dict,
    timeout: float,
    label: str,
    retries: int = 2,
    backoff: float = 0.5,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            resp = client.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.TimeoutException as exc:
            if attempt < retries:
                _sleep(backoff, attempt)
                attempt += 1
                continue
            raise ProviderError(f"{label} timed out after {timeout}s.") from exc
        except httpx.TransportError as exc:
            if attempt < retries:
                _sleep(backoff, attempt)
                attempt += 1
                continue
            raise ProviderError(f"{label} could not reach the endpoint.") from exc
        if resp.status_code >= 400:
            if _should_retry(resp.status_code) and attempt < retries:
                _sleep(backoff, attempt)
                attempt += 1
                continue
            raise _http_error(label, resp)
        return resp


async def apost_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict,
    body: dict,
    timeout: float,
    label: str,
    retries: int = 2,
    backoff: float = 0.5,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            resp = await client.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.TimeoutException as exc:
            if attempt < retries:
                _sleep(backoff, attempt)
                attempt += 1
                continue
            raise ProviderError(f"{label} timed out after {timeout}s.") from exc
        except httpx.TransportError as exc:
            if attempt < retries:
                _sleep(backoff, attempt)
                attempt += 1
                continue
            raise ProviderError(f"{label} could not reach the endpoint.") from exc
        if resp.status_code >= 400:
            if _should_retry(resp.status_code) and attempt < retries:
                _sleep(backoff, attempt)
                attempt += 1
                continue
            raise _http_error(label, resp)
        return resp
