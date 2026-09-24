import pytest

from touchstone import store
from touchstone.interview import rooms


def test_open_post_close_lifecycle(conn):
    room = rooms.open(conn, task_id=None, topic="refunds")
    assert store.get_room(conn, room.id) is not None

    rooms.post(conn, room.id, "sam", "user", "hello")
    msgs = store.list_room_messages(conn, room.id)
    assert len(msgs) == 1 and msgs[0].text == "hello"

    rooms.close(conn, room.id)
    assert store.get_room(conn, room.id).closed_at is not None


@pytest.mark.asyncio
async def test_hub_fans_out_to_two_subscribers():
    hub = rooms.Hub()
    a = hub.subscribe("r1")
    b = hub.subscribe("r1")
    assert hub.subscriber_count("r1") == 2

    delivered = hub.publish("r1", rooms.Event("message", {"text": "hi"}))
    assert delivered == 2
    assert (await a.get()).data["text"] == "hi"
    assert (await b.get()).data["text"] == "hi"


def test_slow_subscriber_drops_without_blocking_others():
    hub = rooms.Hub(maxsize=1)
    slow = hub.subscribe("r1")
    fast = hub.subscribe("r1")

    # Fill the slow subscriber's queue; it now drops, but the fast one keeps receiving.
    slow.put_nowait(rooms.Event("message", {"n": 0}))
    delivered = hub.publish("r1", rooms.Event("message", {"n": 1}))
    assert delivered == 1  # only the fast subscriber accepted it
    assert fast.get_nowait().data["n"] == 1
    assert slow.qsize() == 1  # still holding the stale event, never blocked the broadcast


def test_unsubscribe_removes_the_room_when_empty():
    hub = rooms.Hub()
    q = hub.subscribe("r1")
    hub.unsubscribe("r1", q)
    assert hub.subscriber_count("r1") == 0
    # publishing to an empty room is a no-op, never an error
    assert hub.publish("r1", rooms.Event("closed", {})) == 0


def test_publish_isolated_per_room():
    hub = rooms.Hub()
    a = hub.subscribe("r1")
    hub.subscribe("r2")
    assert hub.publish("r1", rooms.Event("message", {})) == 1
    assert a.qsize() == 1
