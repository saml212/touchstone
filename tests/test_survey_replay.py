import sys

from touchstone.survey import replay

MODULE = '''\
def echo(**kw):
    return {"kw": kw}


def shout(text):
    return text.upper()


def boom(**kw):
    raise ValueError("nope")
'''


def _install(tmp_path, monkeypatch):
    (tmp_path / "replaymod.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("replaymod", None)


def test_dict_arguments_passed_as_kwargs(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    out = replay.replay({"base_url_env": None, "base_url": "",
                         "tools": {"echo": "replaymod:echo"},
                         "calls": [{"tool": "echo", "arguments": {"a": 1}}]})
    assert out == [{"tool": "echo", "got": {"kw": {"a": 1}}}]


def test_non_object_arguments_passed_raw(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    out = replay.replay({"base_url_env": None, "base_url": "",
                         "tools": {"shout": "replaymod:shout"},
                         "calls": [{"tool": "shout", "arguments": "hi"}]})
    assert out == [{"tool": "shout", "got": "HI"}]


def test_tool_error_captured_as_data(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    out = replay.replay({"base_url_env": None, "base_url": "",
                         "tools": {"boom": "replaymod:boom"},
                         "calls": [{"tool": "boom", "arguments": {}}]})
    assert "__error__" in out[0]["got"]
    assert "ValueError" in out[0]["got"]["__error__"]


def test_base_url_env_set_before_use(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    monkeypatch.delenv("WIDGET_URL", raising=False)
    replay.replay({"base_url_env": "WIDGET_URL", "base_url": "http://x", "tools": {}, "calls": []})
    import os
    assert os.environ["WIDGET_URL"] == "http://x"
