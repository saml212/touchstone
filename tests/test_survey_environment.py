"""Snapshot the customer repo into a Harbor environment image directory."""

import re
import subprocess

from touchstone.survey.environment import build_environment

MAP = {
    "tools": [{"name": "get_widget", "calls": ["widget"]}],
    "services": [{"name": "widget", "kind": "http", "base_url_env": "WIDGET_URL", "calls": []},
                 {"name": "paint", "kind": "http", "base_url_env": "PAINT_URL", "calls": []}],
}


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "agent.py").write_text("import httpx\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname="c"\nversion="0.1"\ndependencies=['
        '"httpx>=0.27", "openai>=1.40", "c @ git+ssh://git@github.com/x/c"]\n', encoding="utf-8")
    (repo / ".gitignore").write_text(".venv/\n.touchstone/\ntouchstone/\nsecrets.env\n",
                                     encoding="utf-8")
    (repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
    # ignored + untracked things that must NOT be snapshotted
    (repo / ".venv").mkdir()
    (repo / ".venv" / "junk.py").write_text("x=1\n", encoding="utf-8")
    (repo / "touchstone").mkdir()
    (repo / "touchstone" / "map.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "agent.py", "pyproject.toml", ".gitignore"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b.c", "-c", "user.name=a",
                    "commit", "-qm", "init"], check=True)
    return repo


def _simulators(out):
    sim = out / "simulators" / "widget"
    sim.mkdir(parents=True)
    (sim / "app.py").write_text("# sim\n", encoding="utf-8")
    (sim / "seed.json").write_text("{}", encoding="utf-8")
    (sim / "state.db").write_text("BINARY", encoding="utf-8")  # must be excluded
    (sim / ".sim.log").write_text("log", encoding="utf-8")  # must be excluded


def test_environment_snapshot_and_deps(tmp_path):
    repo = _git_repo(tmp_path)
    out = tmp_path / "touchstone"
    _simulators(out)
    result = build_environment(repo, MAP, out)
    env = out / "environment"

    # repo snapshot honours .gitignore + drops secrets
    assert (env / "repo" / "agent.py").exists()
    assert not (env / "repo" / ".venv").exists()
    assert not (env / "repo" / "touchstone").exists()
    assert not (env / "repo" / ".env").exists()

    # simulators copied without runtime files
    assert (env / "simulators" / "widget" / "app.py").exists()
    assert not (env / "simulators" / "widget" / "state.db").exists()
    assert not (env / "simulators" / "widget" / ".sim.log").exists()

    # the one shared simulator-start script is baked into the image
    start = (env / "simulators" / "start.sh").read_text()
    assert 'python "/app/simulators/$name/app.py" "$port"' in start
    assert "/__health" in start and "/__reset" in start

    # touchstone vendored for offline import
    assert (env / "_touchstone" / "touchstone" / "__init__.py").exists()

    # deps: git dep dropped, sim runtime added
    reqs = (env / "requirements.txt").read_text()
    assert "httpx>=0.27" in reqs and "openai>=1.40" in reqs
    assert "git+ssh" not in reqs
    assert "fastapi" in reqs and "uvicorn" in reqs
    # touchstone's own runtime deps must be present (it is vendored, not pip-installed)
    from touchstone.survey.environment import _touchstone_deps
    pkgs = {re.split(r"[<>=!~ ]", d, maxsplit=1)[0] for d in _touchstone_deps()}
    assert {"typer", "jsonschema", "websockets"} <= pkgs
    for pkg in pkgs:
        assert pkg in reqs

    assert (env / "Dockerfile").read_text().startswith("FROM python:3.12-slim")
    assert result["deps_ok"] is True
    assert result["ports"] == {"widget": 8000, "paint": 8001}
    assert result["base_url_envs"] == {"widget": "WIDGET_URL", "paint": "PAINT_URL"}
    assert result["image_tag"].startswith("touchstone-env-repo:")


CONST_MAP = {
    "tools": [{"name": "get_today_weather", "calls": ["weather"]}],
    "services": [{"name": "weather", "kind": "http", "base_url_env": None,
                  "base_url_default": "http://api.weatherapi.com/v1", "calls": []}],
}


def test_environment_writes_sitecustomize_and_service_hosts(tmp_path):
    repo = _git_repo(tmp_path)
    out = tmp_path / "touchstone"
    result = build_environment(repo, CONST_MAP, out)

    # the net-shim loader is written where PYTHONPATH picks it up, and is self-contained
    site = (out / "environment" / "_touchstone" / "sitecustomize.py").read_text()
    assert "TOUCHSTONE_SIMULATORS" in site and "netshim.py" in site
    assert "import touchstone" not in site  # loads netshim by path, not the whole package

    # a constant-base-URL service exposes its host so the shim can rewrite it
    assert result["hosts"] == {"weather": "api.weatherapi.com"}
    assert result["base_url_envs"] == {"weather": None}


def test_packaged_manifest_and_simulators_map_use_host_for_constant_service():
    from touchstone.survey.package import _sim_manifest, _simulators_map

    env_result = {"services": ["weather"], "ports": {"weather": 8000},
                  "base_url_envs": {"weather": None}, "hosts": {"weather": "api.weatherapi.com"}}
    assert _sim_manifest(env_result) == [{"name": "weather", "port": 8000,
                                          "host": "api.weatherapi.com"}]
    assert _simulators_map(env_result) == {"api.weatherapi.com": "http://127.0.0.1:8000"}


def test_trace_installs_net_shim_from_env(monkeypatch, tmp_path):
    import touchstone
    import touchstone.survey.netshim as netshim

    netshim._PATCHED.clear()
    netshim._MAPPING.clear()
    monkeypatch.setenv("TOUCHSTONE_SIMULATORS", '{"api.weatherapi.com": "http://127.0.0.1:8000"}')
    monkeypatch.chdir(tmp_path)
    touchstone.trace(db=str(tmp_path / "t.db"))
    assert netshim._MAPPING == {"api.weatherapi.com": "http://127.0.0.1:8000"}
    netshim._PATCHED.clear()
    netshim._MAPPING.clear()


def test_environment_flags_missing_deps(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "agent.py").write_text("x=1\n", encoding="utf-8")
    out = tmp_path / "touchstone"
    result = build_environment(repo, MAP, out)
    assert result["deps_ok"] is False
    assert "no pyproject" in result["deps_reason"]
    # sims still runnable — the runtime is always installed
    assert "fastapi" in (out / "environment" / "requirements.txt").read_text()


def test_environment_requirements_txt_fallback(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("httpx==0.27.0  # http client\n\n-e .\n",
                                           encoding="utf-8")
    out = tmp_path / "touchstone"
    result = build_environment(repo, MAP, out)
    reqs = (out / "environment" / "requirements.txt").read_text()
    assert "httpx==0.27.0" in reqs
    assert "-e ." not in reqs  # editable/local dropped
    assert result["deps_ok"] is True


def test_environment_idempotent_then_force(tmp_path):
    repo = _git_repo(tmp_path)
    out = tmp_path / "touchstone"
    _simulators(out)
    build_environment(repo, MAP, out)
    marker = out / "environment" / "repo" / "MARKER"
    marker.write_text("kept", encoding="utf-8")
    build_environment(repo, MAP, out)  # reuse: marker survives
    assert marker.exists()
    build_environment(repo, MAP, out, force=True)  # rebuild: marker gone
    assert not marker.exists()
