"""FastAPI application factory.

`create_app` wires the routers, the error handlers and per-process state on `app.state`:
settings, the DB sessionmaker, the shared `MockFaults` and the login throttle. Without an
injected sessionmaker the lifespan builds the engine on startup and disposes it on shutdown;
tests inject one (ASGI test transports do not run the lifespan).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api import auth, decisions, dev, events, health, preferences, reviews, system, workflows
from app.api.auth import LoginThrottle
from app.api.errors import install_error_handlers
from app.config import Settings, get_settings
from app.providers.mock_mail import MockFaults


def create_app(
    settings: Settings | None = None,
    *,
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if sessionmaker is not None:
            yield
            return
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        app.state.sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            yield
        finally:
            await engine.dispose()

    production = settings.app_env == "production"
    app = FastAPI(
        title="Email Assistant",
        lifespan=lifespan,
        # No public API map in production: /docs, /redoc and /openapi.json answer 404.
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    app.state.settings = settings
    app.state.sessionmaker = sessionmaker
    app.state.faults = MockFaults()
    app.state.login_throttle = LoginThrottle()
    install_error_handlers(app)
    for module in (health, auth, reviews, workflows, decisions, preferences, system, events):
        app.include_router(module.router)
    if not production:
        app.include_router(dev.router)
    return app
