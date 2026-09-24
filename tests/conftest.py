import pytest

import touchstone
from touchstone import store
from touchstone.capture import context


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "touchstone.db")


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
