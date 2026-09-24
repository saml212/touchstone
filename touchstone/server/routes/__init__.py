"""JSON API routers, one module per area, mounted by the app."""

from . import bench, checks, episodes, mine, overview, rooms, tasks, train

ROUTERS = (
    overview.router,
    episodes.router,
    checks.router,
    tasks.router,
    mine.router,
    bench.router,
    rooms.router,
    train.router,
)

__all__ = ["ROUTERS"]
