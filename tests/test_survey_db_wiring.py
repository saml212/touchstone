"""A kind=="db" service is wired without a port: a synthesized env var (or the mapped one) whose
value is the state.db path, no host, and a start.sh branch that re-materializes state.db."""

from touchstone.survey import package, task_files
from touchstone.survey.environment import START_SH, _ports
from touchstone.survey.recordings import ToolEvent

DB_MAP = {
    "tools": [{"name": "get_order", "import_path": "t:get_order", "calls": ["world"]},
              {"name": "hit", "import_path": "t:hit", "calls": ["api"]}],
    "services": [
        {"name": "world", "kind": "db", "base_url_env": None, "base_url_default": ":memory:",
         "calls": [{"method": "GET", "path_template": "/x", "from_tool": "get_order"}]},
        {"name": "api", "kind": "http", "base_url_env": "API_URL", "base_url_default": None,
         "calls": [{"method": "GET", "path_template": "/y", "from_tool": "hit"}]},
    ],
}


def test_ports_synthesize_db_env_and_skip_port():
    wiring = _ports(DB_MAP)
    assert wiring["base_url_envs"]["world"] == "TOUCHSTONE_DB_WORLD"  # synthesized
    assert "world" not in wiring["ports"]  # a db service gets no port
    assert wiring["ports"]["api"] == 8000  # http numbering unaffected by the db service
    assert wiring["kinds"]["world"] == "db"
    assert wiring["db_urls"]["world"] is False  # :memory: default -> a bare path, not a URL
    assert "world" in wiring["services"] and "api" in wiring["services"]


def test_start_sh_has_a_db_branch():
    assert "db_sim" in START_SH
    assert 'if [ ! -f "$sim/app.py" ]; then' in START_SH


def _env_result():
    w = _ports(DB_MAP)
    w["image_tag"] = "img:1"
    return w


def test_package_base_urls_and_manifest_for_db():
    env_result = _env_result()
    base_urls = package._base_urls(env_result)
    assert base_urls["TOUCHSTONE_DB_WORLD"] == "/app/simulators/world/state.db"  # container path
    assert base_urls["API_URL"] == "http://127.0.0.1:8000"
    manifest = {m["name"]: m for m in package._sim_manifest(env_result)}
    assert manifest["world"] == {"name": "world", "kind": "db",
                                 "base_url_env": "TOUCHSTONE_DB_WORLD", "db_url": False}
    assert manifest["api"] == {"name": "api", "port": 8000, "base_url_env": "API_URL"}


def test_solve_lines_and_replay_spec_for_db():
    services = DB_MAP["services"]
    envs = {"world": "TOUCHSTONE_DB_WORLD", "api": "API_URL"}
    ports = {"api": 8000}
    lines = task_files._solve_lines(services, ports, envs)
    assert "bash /app/simulators/start.sh world" in lines  # no port for the db sim
    assert 'export TOUCHSTONE_DB_WORLD="/app/simulators/world/state.db"' in lines
    assert "bash /app/simulators/start.sh api 8000" in lines
    calls = [ToolEvent(tool="get_order", arguments={"id": "1"}, output={}, episode="e")]
    spec = task_files._replay_spec(calls, {"get_order": "t:get_order"}, services, ports, envs)
    assert spec["base_urls"]["TOUCHSTONE_DB_WORLD"] == "/app/simulators/world/state.db"
