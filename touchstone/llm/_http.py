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
            raise ProviderError(f"{label} failed with HTTP {resp.status_code}.")
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
            raise ProviderError(f"{label} failed with HTTP {resp.status_code}.")
        return resp
