"""JSON API routers, one module per area, mounted by the app."""

from . import episodes, overview, rooms

ROUTERS = (
    overview.router,
    episodes.router,
    rooms.router,
)

__all__ = ["ROUTERS"]
