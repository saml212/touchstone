"""FastAPI app for Touchstone: wires the JSON routers, serves the static UI, and holds shared state.

Every request gets its own SQLite connection (the `get_conn` dependency); the room routes and their
agent turn run in a threadpool with fresh connections of their own. See `routes/rooms.py` for the
interview request handling and `routes/_deps.py` for the connection dependency and view helpers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, load_settings
from ..interview.realtime import Bridges
from ..interview.rooms import Hub
from ..interview.speech import Speech
from .routes import ROUTERS

STATIC = Path(__file__).parent / "static"
_NO_STORE = {"Cache-Control": "no-store"}


def _default_provider_factory(settings: Settings):
    def make():
        from ..llm import provider_from_spec

        try:
            return provider_from_spec(settings.agent_provider, settings)
        except Exception:  # no key/binary: the interviewer degrades to plain questions
            return None

    return make


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    await app.state.bridges.close_all()  # tear down any open realtime sessions


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Touchstone interview", lifespan=_lifespan)
    app.state.settings = settings
    app.state.hub = Hub()
    app.state.speech = Speech.from_settings(settings)
    app.state.bridges = Bridges(settings, app.state.hub)
    app.state.provider_factory = _default_provider_factory(settings)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers=_NO_STORE)

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript", headers=_NO_STORE)

    @app.get("/app.css")
    def app_css() -> FileResponse:
        return FileResponse(STATIC / "app.css", media_type="text/css", headers=_NO_STORE)

    @app.get("/rooms/{room_id}")
    def room_page(room_id: str) -> FileResponse:
        return FileResponse(STATIC / "room.html")

    for router in ROUTERS:
        app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


__all__ = ["create_app"]
