"""Health endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import health
from app.config import Settings
from app.main import create_app


async def test_live(client: AsyncClient) -> None:
    r = await client.get("/api/v1/health/live")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


async def test_ready_checks_database(anon: AsyncClient) -> None:
    r = await anon.get("/api/v1/health/ready")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


async def test_ready_503_when_database_unreachable(
    base_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "READY_TIMEOUT_SECONDS", 0.5)
    engine = create_async_engine("postgresql+psycopg://u:p@127.0.0.1:1/none")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    app = create_app(settings, sessionmaker=async_sessionmaker(engine))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/health/ready")
    await engine.dispose()
    assert r.status_code == 503 and r.json() == {"status": "unavailable"}
