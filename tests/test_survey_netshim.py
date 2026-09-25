"""The network shim rewrites a hardcoded host to a simulator, on requests and on httpx."""

import httpx
import pytest
import requests

import touchstone.survey.netshim as ns

_MAP = {"api.weatherapi.com": "http://127.0.0.1:8712"}


@pytest.fixture(autouse=True)
def _reset():
    ns._PATCHED.clear()
    ns._MAPPING.clear()
    yield
    ns._PATCHED.clear()
    ns._MAPPING.clear()


def test_rewrite_url_swaps_host_keeps_path_and_query():
    out = ns._rewrite_url("http://api.weatherapi.com/v1/current.json?q=Paris&aqi=no", _MAP)
    assert out == "http://127.0.0.1:8712/v1/current.json?q=Paris&aqi=no"


def test_rewrite_url_unknown_host_is_left_alone():
    assert ns._rewrite_url("http://example.com/x", _MAP) is None


def test_requests_request_is_rewritten(monkeypatch):
    captured = {}

    def fake_request(self, method, url, *a, **k):
        captured["url"] = url
        return "ok"

    monkeypatch.setattr(requests.Session, "request", fake_request)
    assert "requests" in ns.install(_MAP)
    requests.Session().request("GET", "http://api.weatherapi.com/v1/current.json?q=x")
    assert captured["url"] == "http://127.0.0.1:8712/v1/current.json?q=x"


def test_requests_unmapped_host_untouched(monkeypatch):
    captured = {}
    monkeypatch.setattr(requests.Session, "request",
                        lambda self, method, url, *a, **k: captured.setdefault("url", url))
    ns.install(_MAP)
    requests.Session().request("GET", "http://other.example/keep")
    assert captured["url"] == "http://other.example/keep"


def test_httpx_client_send_is_rewritten(monkeypatch):
    captured = {}

    def fake_send(self, request, *a, **k):
        captured["url"] = str(request.url)
        return "ok"

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    assert "httpx" in ns.install(_MAP)
    httpx.Client().send(httpx.Request("GET", "http://api.weatherapi.com/forecast.json?days=3"))
    assert captured["url"] == "http://127.0.0.1:8712/forecast.json?days=3"


async def test_httpx_async_client_send_is_rewritten(monkeypatch):
    captured = {}

    async def fake_send(self, request, *a, **k):
        captured["url"] = str(request.url)
        return "ok"

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    ns.install(_MAP)
    await httpx.AsyncClient().send(httpx.Request("GET", "http://api.weatherapi.com/v1/x"))
    assert captured["url"] == "http://127.0.0.1:8712/v1/x"


def test_install_from_env_reads_json(monkeypatch):
    monkeypatch.setenv("TOUCHSTONE_SIMULATORS", '{"api.weatherapi.com": "http://127.0.0.1:9001"}')
    ns.install_from_env()
    assert ns._MAPPING == {"api.weatherapi.com": "http://127.0.0.1:9001"}


def test_install_from_env_absent_is_noop(monkeypatch):
    monkeypatch.delenv("TOUCHSTONE_SIMULATORS", raising=False)
    assert ns.install_from_env() == []
    assert ns._MAPPING == {}


def test_empty_mapping_patches_nothing():
    assert ns.install({}) == []
    assert ns._PATCHED == set()
