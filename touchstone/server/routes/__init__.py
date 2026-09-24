"""JSON API routers, one module per area, mounted by the app."""

from . import bench, checks, episodes, mine, overview, rooms, tasks

ROUTERS = (
    overview.router,
    episodes.router,
    checks.router,
    tasks.router,
    mine.router,
    bench.router,
    rooms.router,
)

__all__ = ["ROUTERS"]
