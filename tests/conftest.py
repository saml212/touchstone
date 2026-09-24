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
    """Project root that holds tasks/, checks.toml, benchmarks/ (db lives under .touchstone/)."""
    return str(tmp_path)


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


@pytest.fixture
def project(traced, tmp_path):
    """A ready project: demo episodes traced, mined (policies enabled), tasks synced.

    `conn, root = project(12)` — returns a connection and the project root with task/checks files.
    """
    from touchstone import policies
    from touchstone.mine import mine, sync

    conns = []
    root = str(tmp_path)

    def make(n: int = 12, enable: bool = True):
        run_demo(n=n)
        conn = store.connect(traced)
        conns.append(conn)
        mine(conn, root, no_llm=True)
        if enable:
            policies.enable_all_mined(root)
            sync(conn, root)
        return conn, root

    yield make
    for conn in conns:
        conn.close()
