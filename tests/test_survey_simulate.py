"""Simulate + fidelity end to end with a scripted provider and the current interpreter.

The scripted provider emits a real (tiny) FastAPI simulator; fidelity replays a real customer tool
module (in a tmp project) against it. `[survey] python` is set to the current interpreter so the
replay does not need uv.
"""

import json
import sys

from touchstone.config import Settings
from touchstone.survey.provider import ScriptedSurveyProvider
from touchstone.survey.recordings import ToolEvent
from touchstone.survey.scrub import Scrubber
from touchstone.survey.simulate import generate_simulator

CUSTOMER_TOOL = '''\
import os
import httpx

BASE = os.environ.get("WIDGET_URL", "http://127.0.0.1:9")
_client = httpx.Client(base_url=BASE, timeout=5)


def get_widget(widget_id: str) -> dict:
    return _client.get(f"/widgets/{widget_id}").json()
'''

SIM_SRC = '''\
import json
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI()
HERE = Path(__file__).parent
DB = HERE / "state.db"


def load():
    conn = sqlite3.connect(DB)
    conn.execute("DROP TABLE IF EXISTS widgets")
    conn.execute("CREATE TABLE widgets (id TEXT PRIMARY KEY, color TEXT)")
    seed = json.loads((HERE / "seed.json").read_text())
    for w in seed.get("widgets", []):
        conn.execute("INSERT OR REPLACE INTO widgets VALUES (?, ?)", (w["id"], w["color"]))
    conn.commit()
    conn.close()


@app.get("/__health")
def health():
    return {"ok": True}


@app.post("/__reset")
def reset():
    load()
    return {"ok": True}


@app.get("/widgets/{widget_id}")
def get_widget(widget_id: str):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM widgets WHERE id=?", (widget_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "no widget")
    return {"id": row["id"], "color": row["color"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
'''

SERVICE = {
    "name": "widget", "kind": "http", "base_url_env": "WIDGET_URL",
    "base_url_default": "http://127.0.0.1:9",
    "calls": [{"method": "GET", "path_template": "/widgets/{id}", "from_tool": "get_widget"}],
}
TOOLS = [{"name": "get_widget", "import_path": "customer_tools:get_widget",
          "file": "customer_tools.py", "line": 1, "calls": ["widget"]}]


def _project(tmp_path):
    (tmp_path / "customer_tools.py").write_text(CUSTOMER_TOOL, encoding="utf-8")
    return tmp_path


def _settings(threshold=0.8):
    return Settings(survey_python=sys.executable, survey_fidelity_threshold=threshold)


def _good_answer():
    return json.dumps({"app.py": SIM_SRC,
                       "seed.json": {"widgets": [{"id": "w1", "color": "red"}]},
                       "README.md": "widget simulator"})


def test_generate_and_measure_reaches_full_fidelity(tmp_path):
    repo = _project(tmp_path)
    events = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]
    provider = ScriptedSurveyProvider([_good_answer()])
    result = generate_simulator(repo, provider, SERVICE, TOOLS, events,
                                repo / "touchstone" / "simulators", Scrubber(), _settings())
    assert result == {"calls": 1, "reproduced": 1, "score": 1.0, "threshold": 0.8,
                      "masked_keys": ["id"], "failures": []}
    sim = repo / "touchstone" / "simulators" / "widget"
    assert (sim / "app.py").exists() and (sim / "seed.json").exists()


def test_cached_simulator_not_regenerated(tmp_path):
    repo = _project(tmp_path)
    events = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]
    provider = ScriptedSurveyProvider([_good_answer()])
    root = repo / "touchstone" / "simulators"
    generate_simulator(repo, provider, SERVICE, TOOLS, events, root, Scrubber(), _settings())
    generate_simulator(repo, provider, SERVICE, TOOLS, events, root, Scrubber(), _settings())
    assert len(provider.calls) == 1  # second run reused files


def test_syntax_error_simulator_scores_zero_and_survives(tmp_path):
    repo = _project(tmp_path)
    events = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]
    broken = json.dumps({"app.py": "def (:\n  syntax error", "seed.json": {}, "README.md": "x"})
    provider = ScriptedSurveyProvider([broken])  # retry reuses the same broken answer
    result = generate_simulator(repo, provider, SERVICE, TOOLS, events,
                                repo / "touchstone" / "simulators", Scrubber(), _settings())
    assert result["score"] == 0.0
    assert result["failures"] and "error" in result["failures"][0]


def test_regenerate_keeps_better_of_two(tmp_path):
    repo = _project(tmp_path)
    events = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]
    broken = json.dumps({"app.py": "raise SystemExit(1)", "seed.json": {}, "README.md": "x"})
    provider = ScriptedSurveyProvider([broken, _good_answer()])  # first fails, retry fixes it
    result = generate_simulator(repo, provider, SERVICE, TOOLS, events,
                                repo / "touchstone" / "simulators", Scrubber(), _settings())
    assert result["score"] == 1.0
    assert len(provider.calls) == 2


CONST_HOST_TOOL = '''\
import requests


def get_widget(widget_id: str) -> dict:
    return requests.get(f"http://api.fake.test/widgets/{widget_id}", timeout=5).json()
'''

SERVICE_CONST = {
    "name": "widget", "kind": "http", "base_url_env": None,
    "base_url_default": "http://api.fake.test/",
    "calls": [{"method": "GET", "path_template": "/widgets/{id}", "from_tool": "get_widget"}],
}


def test_constant_host_service_is_simulated_via_net_shim(tmp_path):
    # A tool that hard-codes its host (no base_url_env to override) must still be measurable: the
    # net shim rewrites api.fake.test to the loopback simulator. Before the shim this was flagged.
    (tmp_path / "customer_tools.py").write_text(CONST_HOST_TOOL, encoding="utf-8")
    events = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]
    result = generate_simulator(tmp_path, ScriptedSurveyProvider([_good_answer()]), SERVICE_CONST,
                                TOOLS, events, tmp_path / "touchstone" / "simulators", Scrubber(),
                                _settings())
    assert result["score"] == 1.0 and result["reproduced"] == 1 and result["failures"] == []


def test_two_services_same_env_collapse_to_one():
    from touchstone.survey.simulate import crossing_services, service_tools
    map_data = {
        "services": [
            {"name": "orders", "base_url_env": "SHARED_URL", "calls": []},
            {"name": "billing", "base_url_env": "SHARED_URL", "calls": []},
        ],
        "tools": [{"name": "t1", "calls": ["orders"]}, {"name": "t2", "calls": ["billing"]}],
    }
    services = crossing_services(map_data)
    assert [s["name"] for s in services] == ["orders"]  # one per base_url_env
    assert [t["name"] for t in service_tools(map_data, services[0])] == ["t1"]


def test_generated_simulator_is_forced_to_bind_loopback(tmp_path):
    # Attack (stage-7 security): the simulator's bind host depends on the LLM following the prompt.
    # A generated `host="0.0.0.0"` would expose the seeded service on every interface when it runs
    # on the host during fidelity. The writer forces loopback regardless of what the model emitted.
    repo = _project(tmp_path)
    exposed = SIM_SRC.replace('host="127.0.0.1"', 'host="0.0.0.0"')
    assert 'host="0.0.0.0"' in exposed  # the model tried to bind all interfaces
    answer = json.dumps({"app.py": exposed,
                         "seed.json": {"widgets": [{"id": "w1", "color": "red"}]},
                         "README.md": "widget simulator"})
    events = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]
    generate_simulator(repo, ScriptedSurveyProvider([answer]), SERVICE, TOOLS, events,
                       repo / "touchstone" / "simulators", Scrubber(), _settings())
    written = (repo / "touchstone" / "simulators" / "widget" / "app.py").read_text()
    assert 'host="127.0.0.1"' in written
    assert "0.0.0.0" not in written


def test_crossing_services_skips_a_service_no_tool_calls():
    # The model SDK the agent thinks with often shows up as a service with no tool referencing it;
    # it is swapped by setting, not simulated, so it must be dropped (not given an empty simulator).
    from touchstone.survey.simulate import crossing_services
    map_data = {
        "tools": [{"name": "get_today_weather", "calls": ["WeatherAPI"]}],
        "services": [{"name": "OpenAI", "kind": "sdk", "base_url_env": None, "calls": []},
                     {"name": "WeatherAPI", "kind": "http", "base_url_env": None,
                      "base_url_default": "http://api.weatherapi.com/v1", "calls": []}],
    }
    assert [s["name"] for s in crossing_services(map_data)] == ["WeatherAPI"]


def test_external_host_is_shielded_by_the_shim_even_with_a_base_url_env():
    # If the map wrongly records an API-key var as base_url_env, the env override would not repoint
    # the hard-coded external host. The shim shield rewrites that host so replay never hits it.
    from touchstone.survey.fidelity import _replay_spec
    from touchstone.survey.recordings import ToolEvent
    ctx = {"base_url_env": "WEATHER_API_KEY", "host": "api.weatherapi.com", "kind": "http",
           "tools": {"t": "m:t"}}
    spec = _replay_spec([ToolEvent("t", {}, {}, "e")], "http://127.0.0.1:9999", ctx)
    assert spec["base_url_env"] == "WEATHER_API_KEY"
    assert spec["simulators"] == {"api.weatherapi.com": "http://127.0.0.1:9999"}


def test_loopback_default_host_is_not_shimmed_when_env_overrides():
    # A dev-default loopback host with a real base_url_env is repointed by the env var alone; the
    # shim must not broadly rewrite 127.0.0.1.
    from touchstone.survey.fidelity import _replay_spec
    from touchstone.survey.recordings import ToolEvent
    ctx = {"base_url_env": "ORDERS_URL", "host": "127.0.0.1", "kind": "http",
           "tools": {"t": "m:t"}}
    spec = _replay_spec([ToolEvent("t", {}, {}, "e")], "http://127.0.0.1:8000", ctx)
    assert "simulators" not in spec


# A tool that TRANSFORMS the response (extracts nested fields) — the recorded output is the parsed
# value, so the simulator must emit the raw wire body the tool parses, not the parsed value itself.
TRANSFORM_TOOL = '''\
import os
import httpx

BASE = os.environ.get("WIDGET_URL", "http://127.0.0.1:9")


def get_widget(widget_id: str) -> dict:
    data = httpx.get(f"{BASE}/widgets/{widget_id}", timeout=5).json()
    return {"id": widget_id, "color": data["result"]["color"]}
'''

TRANSFORM_SIM = '''\
import sys

from fastapi import FastAPI

app = FastAPI()


@app.get("/__health")
def health():
    return {"ok": True}


@app.get("/widgets/{widget_id}")
def get_widget(widget_id: str):
    return {"result": {"color": "red"}, "meta": {"served": True}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
'''


def test_transforming_tool_needs_raw_wire_body(tmp_path):
    (tmp_path / "customer_tools.py").write_text(TRANSFORM_TOOL, encoding="utf-8")
    answer = json.dumps({"app.py": TRANSFORM_SIM, "seed.json": {}, "README.md": "x"})
    # recorded output is the tool's PARSED return; the sim emits the nested wire body it parses.
    events = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]
    result = generate_simulator(tmp_path, ScriptedSurveyProvider([answer]), SERVICE, TOOLS, events,
                                tmp_path / "touchstone" / "simulators", Scrubber(), _settings())
    assert result["score"] == 1.0 and result["failures"] == []


def test_response_key_paths_finds_nested_reads():
    from touchstone.survey.simulate import response_key_paths
    src = (
        "def get(r):\n"
        "    d = r.json()\n"
        "    return {'t': d['current']['temp_c'], 'c': d['current']['condition']['text'],\n"
        "            'e': d['error']['message']}\n"
    )
    paths = response_key_paths(src)
    assert "current.temp_c" in paths
    assert "current.condition.text" in paths
    assert "error.message" in paths
    assert "current" not in paths  # pure prefixes dropped


def test_failure_hint_names_missing_keyerror_keys():
    from touchstone.survey.simulate import _failure_hint
    result = {"failures": [{"tool": "t", "got": {"__error__": "KeyError: 'current'"}},
                           {"tool": "t", "got": {"__error__": "KeyError: 'forecast'"}}]}
    hint = _failure_hint(result)
    assert "current" in hint and "forecast" in hint and "MUST include them" in hint


def test_routes_include_base_path_only_for_constant_base_url_service():
    from touchstone.survey.simulate import _routes_text
    const = {"base_url_env": None, "base_url_default": "http://api.weatherapi.com/v1",
             "calls": [{"method": "GET", "path_template": "/current.json", "from_tool": "t"}]}
    env = {"base_url_env": "ORDERS_URL", "base_url_default": "http://127.0.0.1:8710",
           "calls": [{"method": "GET", "path_template": "/orders/{id}", "from_tool": "t"}]}
    assert "GET /v1/current.json" in _routes_text(const)  # shim keeps the /v1 base path
    assert "GET /orders/{id}" in _routes_text(env)  # env override replaces the whole base
    assert "/v1" not in _routes_text(env)


def test_auth_env_names_finds_key_token_secret():
    from touchstone.survey.tool_reads import auth_env_names
    src = ('_require_env("FLIGHT_API_KEY")\nos.getenv("OPENAI_API_KEY")\nx = "STRIPE_SECRET_KEY"\n'
           'print("GITHUB_TOKEN", "DB_PASSWORD")\ny = os.environ["BASE_URL"]')
    names = auth_env_names(src)
    assert names == {"FLIGHT_API_KEY", "OPENAI_API_KEY", "STRIPE_SECRET_KEY", "GITHUB_TOKEN",
                     "DB_PASSWORD"}
    assert "BASE_URL" not in names  # not auth-shaped, never clobbered


def test_replay_sets_auth_env_placeholder_from_repo(tmp_path, monkeypatch):
    # A tool that must read an API key to build its client works under replay even with no real key:
    # replay sets a placeholder for the auth-shaped env it finds in the repo.
    import os

    from touchstone.survey import replay as replay_mod
    (tmp_path / "svc.py").write_text(
        'import os\n\n\ndef ping(x):\n    return {"key": os.environ["SVC_API_KEY"], "x": x}\n',
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("SVC_API_KEY", raising=False)
    out = replay_mod.replay({"tools": {"ping": "svc:ping"}, "calls": [{"tool": "ping",
                                                                       "arguments": {"x": 1}}]})
    assert out[0]["got"] == {"key": "touchstone-placeholder", "x": 1}
    assert os.environ.get("SVC_API_KEY") == "touchstone-placeholder"


# ---- cross-call minted ids: a create returns an id a later call uses ----------------------------

BOOKING_TOOL = '''\
import os

import httpx

BASE = os.environ.get("BOOK_URL", "http://127.0.0.1:9")
_client = httpx.Client(base_url=BASE, timeout=5)


def save_passenger(name: str) -> dict:
    return _client.post("/passengers", json={"name": name}).json()


def book_flight(passenger_id: str, flight: str) -> dict:
    r = _client.post("/bookings", json={"passenger_id": passenger_id, "flight": flight})
    if r.status_code == 404:
        return {"error": "Passenger not found"}
    return r.json()
'''

BOOKING_SIM = '''\
import json
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
HERE = Path(__file__).parent
DB = HERE / "state.db"


def load():
    conn = sqlite3.connect(DB)
    conn.execute("DROP TABLE IF EXISTS passengers")
    conn.execute("CREATE TABLE passengers (passenger_id TEXT PRIMARY KEY, name TEXT)")
    seed = json.loads((HERE / "seed.json").read_text())
    for p in seed.get("passengers", []):
        conn.execute("INSERT OR REPLACE INTO passengers VALUES (?, ?)",
                     (p["passenger_id"], p["name"]))
    conn.commit()
    conn.close()


class Passenger(BaseModel):
    name: str


class Booking(BaseModel):
    passenger_id: str
    flight: str


@app.get("/__health")
def health():
    return {"ok": True}


@app.post("/__reset")
def reset():
    load()
    return {"ok": True}


@app.post("/passengers")
def create(p: Passenger):
    conn = sqlite3.connect(DB)
    pid = "PAX-NEW"  # a freshly minted id, NOT the recorded one
    conn.execute("INSERT OR REPLACE INTO passengers VALUES (?, ?)", (pid, p.name))
    conn.commit()
    conn.close()
    return {"passenger_id": pid, "name": p.name}


@app.post("/bookings")
def book(b: Booking):
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT name FROM passengers WHERE passenger_id = ?",
                       (b.passenger_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Passenger not found")
    return {"booking_id": "BK-NEW", "passenger_id": b.passenger_id, "flight": b.flight}


if __name__ == "__main__":
    load()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
'''

BOOKING_SERVICE = {
    "name": "booking", "kind": "http", "base_url_env": "BOOK_URL", "base_url_default": None,
    "calls": [{"method": "POST", "path_template": "/passengers", "from_tool": "save_passenger"},
              {"method": "POST", "path_template": "/bookings", "from_tool": "book_flight"}],
}
BOOKING_TOOLS = [
    {"name": "save_passenger", "import_path": "customer_tools:save_passenger",
     "file": "customer_tools.py", "line": 1, "calls": ["booking"]},
    {"name": "book_flight", "import_path": "customer_tools:book_flight",
     "file": "customer_tools.py", "line": 1, "calls": ["booking"]},
]


def test_two_step_create_then_use_reproduces_via_seeded_id(tmp_path):
    # save_passenger mints PAX-9 in the recording; book_flight later passes PAX-9. The simulator
    # mints its OWN id on save, so book only reproduces because seed.json carries the recorded id
    # (the cross-call-id fix) AND the calls replay in recorded order within the episode.
    (tmp_path / "customer_tools.py").write_text(BOOKING_TOOL, encoding="utf-8")
    events = [
        ToolEvent("save_passenger", {"name": "Ada"},
                  {"passenger_id": "PAX-9", "name": "Ada"}, "e1"),
        ToolEvent("book_flight", {"passenger_id": "PAX-9", "flight": "AA1"},
                  {"booking_id": "BK-1", "passenger_id": "PAX-9", "flight": "AA1"}, "e1"),
    ]
    answer = json.dumps({"app.py": BOOKING_SIM,
                         "seed.json": {"passengers": [{"passenger_id": "PAX-9", "name": "Ada"}]},
                         "README.md": "booking simulator"})
    result = generate_simulator(tmp_path, ScriptedSurveyProvider([answer]), BOOKING_SERVICE,
                                BOOKING_TOOLS, events, tmp_path / "touchstone" / "simulators",
                                Scrubber(), _settings())
    assert result["score"] == 1.0 and result["reproduced"] == 2 and result["failures"] == []
