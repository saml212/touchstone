"""agent/invoke.py lets the survey drive tools that aren't free functions (methods on a constructed
client, or central dispatch). Adapter-checked; fidelity routes through it when it passes."""

import json
import sys

from tests.test_survey_simulate import SIM_SRC
from touchstone.config import Settings
from touchstone.survey import fidelity
from touchstone.survey.invoke import build_invoke
from touchstone.survey.provider import ScriptedSurveyProvider
from touchstone.survey.recordings import ToolEvent
from touchstone.survey.scrub import Scrubber
from touchstone.survey.simulate import _replay_ctx

# The tool the model calls is dispatched through `dispatch(client, name, args)` and the client is
# constructed from an env var — a bare import_path(**args) cannot drive it.
DISPATCH_TOOL = '''\
import os

import httpx


class Client:
    def __init__(self):
        self.base = os.environ["SVC_URL"]

    def get_widget(self, widget_id):
        return httpx.get(f"{self.base}/widgets/{widget_id}", timeout=5).json()


def dispatch(client, name, arguments):
    return getattr(client, name)(**arguments)
'''

GOOD_INVOKE = '''\
import customer_dispatch


def invoke(name, arguments):
    return customer_dispatch.dispatch(customer_dispatch.Client(), name, arguments)
'''

BROKEN_INVOKE = '''\
def invoke(name, arguments):
    raise RuntimeError("cannot dispatch")
'''

MAP = {
    "tools": [{"name": "get_widget", "import_path": "customer_dispatch:dispatch",
               "file": "customer_dispatch.py", "line": 1, "calls": ["widget"]}],
    "services": [{"name": "widget", "kind": "http", "base_url_env": "SVC_URL",
                  "base_url_default": "http://127.0.0.1:9",
                  "calls": [{"method": "GET", "path_template": "/widgets/{id}",
                             "from_tool": "get_widget"}]}],
}
EVENTS = [ToolEvent("get_widget", {"widget_id": "w1"}, {"id": "w1", "color": "red"}, "e1")]


def _repo(tmp_path):
    (tmp_path / "customer_dispatch.py").write_text(DISPATCH_TOOL, encoding="utf-8")
    sim = tmp_path / "touchstone" / "simulators" / "widget"
    sim.mkdir(parents=True)
    (sim / "app.py").write_text(SIM_SRC, encoding="utf-8")
    (sim / "seed.json").write_text(json.dumps({"widgets": [{"id": "w1", "color": "red"}]}),
                                   encoding="utf-8")
    return tmp_path


def _settings():
    return Settings(survey_python=sys.executable, survey_fidelity_threshold=0.8)


def test_invoke_py_reproduces_a_dispatcher_tool(tmp_path):
    repo = _repo(tmp_path)
    out = repo / "touchstone"
    result = build_invoke(repo, MAP, ScriptedSurveyProvider([GOOD_INVOKE]), EVENTS, out,
                          _settings())
    assert result["ok"] and result["path"]

    ctx = _replay_ctx(MAP["services"][0], MAP["tools"])
    scrub = Scrubber()
    # Without invoke.py the free-function replay calls dispatch(**args) -> TypeError -> fidelity 0.
    bare = fidelity.measure_service(out / "simulators" / "widget", repo, EVENTS, ctx, _settings(),
                                    scrub)
    assert bare["score"] == 0.0
    # Through invoke.py the client is constructed and the tool dispatched -> full fidelity.
    via = fidelity.measure_service(out / "simulators" / "widget", repo, EVENTS, ctx, _settings(),
                                   scrub, invoke=result["path"])
    assert via["score"] == 1.0 and via["failures"] == []


def test_broken_invoke_py_fails_the_adapter_check_and_is_removed(tmp_path):
    repo = _repo(tmp_path)
    out = repo / "touchstone"
    result = build_invoke(repo, MAP, ScriptedSurveyProvider([BROKEN_INVOKE]), EVENTS, out,
                          _settings())
    assert result["ok"] is False and result["path"] is None
    assert "adapter check failed" in result["flag"]
    assert not (out / "agent" / "invoke.py").exists()  # removed so callers fall back
