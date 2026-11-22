"""FastAPI application factory."""

from fastapi import FastAPI

from app.api import health
from app.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Email Assistant")
    app.state.settings = settings or get_settings()
    app.include_router(health.router)
    return app
