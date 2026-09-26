import pytest

import touchstone
from touchstone import store
from touchstone.capture import context
from touchstone.demo import run_demo


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / ".touchstone" / "touchstone.db")


@pytest.fixture
def root(tmp_path):
    """Project root that holds the `touchstone/` dataset (db lives under .touchstone/)."""
    return str(tmp_path)


@pytest.fixture
def seed_task():
    """Write a minimal task under <dataset>/tasks so the CLI empty-tasks guard passes."""
    from pathlib import Path

    def make(dataset_root, name="t1"):
        d = Path(dataset_root) / "tasks" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "task.toml").write_text("[task]\nname = 'd/t1'\n", encoding="utf-8")
        return d

    return make


@pytest.fixture
def conn(db):
    c = store.connect(db)
    yield c
    c.close()


@pytest.fixture
def traced(db):
    """Configure capture against a fresh db and reset process-global capture state."""
    context._untracked_ids.clear()
    context._local.__dict__.pop("conn", None)
    touchstone.trace(db)
    yield db
    context._untracked_ids.clear()
    context._local.__dict__.pop("conn", None)


@pytest.fixture
def demo_db(traced):
    """Factory for a demo db: `conn = demo_db(16)` runs `n` scripted episodes and returns a
    connection to the traced db. Every connection handed out is closed at teardown."""
    conns = []

    def make(n: int = 30):
        run_demo(n=n)
        conn = store.connect(traced)
        conns.append(conn)
        return conn

    yield make
    for conn in conns:
        conn.close()
