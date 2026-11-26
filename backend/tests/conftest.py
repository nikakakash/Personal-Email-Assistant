"""Shared pytest fixtures: a clean, valid environment and an in-process HTTP client."""

import asyncio
import os
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings
from app.db.models import (
    Base,
    EmailMessage,
    EmailThread,
    Owner,
    Proposal,
    ProposalRevision,
    WorkflowRun,
)
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


_BACKEND_DIR = Path(__file__).resolve().parent.parent
_TEST_DB_NAME = "email_assistant_test"


def _pg_param(name: str, default: str) -> str:
    return os.environ.get(name, default)


@pytest.fixture(scope="session")
def db_url() -> str:
    """SQLAlchemy URL of `email_assistant_test`, created if missing and migrated to head.

    Connection details come from POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_HOST /
    POSTGRES_PORT (defaults match compose.yaml). Skips when Postgres is unreachable, except
    under CI (env `CI` set), where it fails.
    """
    user = _pg_param("POSTGRES_USER", "app")
    password = _pg_param("POSTGRES_PASSWORD", "app")
    host = _pg_param("POSTGRES_HOST", "127.0.0.1")
    port = _pg_param("POSTGRES_PORT", "5433")
    admin_db = _pg_param("POSTGRES_DB", "email_assistant")
    try:
        with psycopg.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            dbname=admin_db,
            connect_timeout=3,
            autocommit=True,
        ) as conn:
            exists = conn.execute(
                "select 1 from pg_database where datname = %s", (_TEST_DB_NAME,)
            ).fetchone()
            if not exists:
                conn.execute(f'create database "{_TEST_DB_NAME}"')
    except psycopg.OperationalError as exc:
        message = f"Postgres unreachable at {host}:{port}; run `docker compose up -d db` ({exc})"
        if os.environ.get("CI"):
            pytest.fail(message)  # in CI a missing database must not turn into green skips
        pytest.skip(message)

    creds = f"{quote(user, safe='')}:{quote(password, safe='')}"
    url = f"postgresql+psycopg://{creds}@{host}:{port}/{_TEST_DB_NAME}"
    cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(cfg, "head")
    return url


@pytest.fixture
async def sessionmaker(db_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Sessionmaker on the test DB; all app tables are truncated after each test."""
    engine = create_async_engine(db_url, poolclass=NullPool)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    await engine.dispose()


@pytest.fixture
async def session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as s:
        yield s


@pytest.fixture
async def owner(session: AsyncSession) -> Owner:
    row = Owner(email="owner@example.com", time_zone="Europe/Berlin")
    session.add(row)
    await session.commit()
    return row


@pytest.fixture
def workflow_factory(session: AsyncSession) -> Callable[[], Awaitable[WorkflowRun]]:
    """Create a thread, message and WorkflowRun (with unique ids per call)."""

    async def make() -> WorkflowRun:
        n = uuid.uuid4().hex[:8]
        thread = EmailThread(account_id="mock-account", provider_thread_id=f"t-{n}")
        session.add(thread)
        await session.flush()
        message = EmailMessage(
            thread_id=thread.id,
            account_id="mock-account",
            provider_message_id=f"m-{n}",
            sender="alice@example.com",
            received_at=datetime.now(UTC),
            content_hash=f"hash-{n}",
        )
        session.add(message)
        await session.flush()
        run = WorkflowRun(
            graph_thread_id=f"mock-account:m-{n}:wf1",
            message_id=message.id,
            provider_thread_id=thread.provider_thread_id,
            graph_version="test",
        )
        session.add(run)
        await session.commit()
        return run

    return make


@pytest.fixture
def revision_factory(
    session: AsyncSession, workflow_factory: Callable[[], Awaitable[WorkflowRun]]
) -> Callable[[], Awaitable[ProposalRevision]]:
    """Create a WorkflowRun, its Proposal and a ProposalRevision; return the revision."""

    async def make() -> ProposalRevision:
        run = await workflow_factory()
        proposal = Proposal(workflow_id=run.id, kind="send_reply")
        session.add(proposal)
        await session.flush()
        revision = ProposalRevision(
            proposal_id=proposal.id,
            revision=1,
            payload={"body": "Hello"},
            payload_hash=uuid.uuid4().hex,
            author="model",
        )
        session.add(revision)
        await session.commit()
        return revision

    return make


@pytest.fixture
async def client(base_env: None) -> AsyncIterator[AsyncClient]:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
