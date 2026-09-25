"""JSON API routers, one module per area, mounted by the app."""

from . import episodes, overview, pages, rooms

ROUTERS = (
    overview.router,
    episodes.router,
    pages.router,
    rooms.router,
)

__all__ = ["ROUTERS"]
