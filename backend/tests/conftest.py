"""Shared pytest fixtures: a clean, valid environment and an in-process HTTP client."""

import asyncio
import sys
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings, get_settings
from app.main import create_app

# psycopg async needs a selector event loop on Windows. pytest-asyncio 1.x removed the
# `event_loop_policy` fixture, so the policy is installed once at import time instead.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_ENV_NAMES = [name.upper() for name in Settings.model_fields]


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ambient settings env vars and cached settings so tests never see a real config."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal valid environment for constructing Settings."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
    monkeypatch.setenv("CHECKPOINT_DATABASE_URL", "postgresql://u:p@localhost:5432/test")
    monkeypatch.setenv("OWNER_EMAIL", "owner@example.com")
    monkeypatch.setenv("OWNER_PASSPHRASE", "a-real-passphrase")
    monkeypatch.setenv("SESSION_SECRET", "a-real-session-secret")


@pytest.fixture
async def client(base_env: None) -> AsyncIterator[AsyncClient]:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
