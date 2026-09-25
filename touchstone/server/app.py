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
from ..llm.keychain import secret
from .routes import ROUTERS
from .routes.rooms import ingest_turn

STATIC = Path(__file__).parent / "static"
_NO_STORE = {"Cache-Control": "no-store"}


def _default_provider_factory(settings: Settings):
    def make():
        from ..llm import provider_from_spec

        try:
            return provider_from_spec(review_provider(settings), settings)
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
    app.state.provider_factory = _default_provider_factory(settings)

    async def _respond(room_id: str, speaker: str, text: str) -> str | None:
        """Run a transcribed voice turn through the same ReviewAgent the text path uses, and return
        the agent's reply for the realtime bridge to voice. A closed/missing room yields None."""
        try:
            result = await ingest_turn(app, room_id, speaker, text)
        except Exception:  # a closed/missing room must not crash the voice bridge
            return None
        return (result.get("agent") or {}).get("text")

    app.state.bridges = Bridges(settings, app.state.hub, respond=_respond)

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


_CLI = ("claude-cli", "codex-cli")


def review_provider(settings: Settings) -> str:
    """The chat provider the review room drives. A CLI provider emulates tool calls through a
    JSON convention and its harness sometimes balks; when the customer already has an OpenAI key
    the room uses real function calling instead. `review_provider` in touchstone.toml pins it."""
    if settings.review_provider:
        return settings.review_provider
    head = settings.agent_provider.split(":", 1)[0]
    service = settings.keychain_service(settings.keychain_openai)
    if head in _CLI and secret("OPENAI_API_KEY", service):
        return "openai:gpt-4o-mini"
    return settings.agent_provider
