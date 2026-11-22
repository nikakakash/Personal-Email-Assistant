"""Fixtures for queue and worker tests: the real `AsyncPostgresSaver` on the test DB.

`checkpoint_url` runs `AsyncPostgresSaver.setup()` once per session; `checkpoints` empties the
checkpoint tables around each test (`checkpoint_migrations` is kept, it records the schema
version). Each "worker" in a test opens its own saver with `open_graph`, so a restart is a new
saver and a new compiled graph, exactly as in a new process.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.graphs.deps import GraphDeps
from app.graphs.parent import build_parent_graph
from app.jobs.worker import Graph
from app.llm.client import LLMClient, Purpose
from app.llm.scripted import ScriptedLLMClient
from app.providers.interfaces import MailProvider
from app.providers.mock_calendar import MockCalendarProvider
from app.providers.mock_mail import MockFaults, MockMailProvider

CHECKPOINT_TABLES = ("checkpoints", "checkpoint_writes", "checkpoint_blobs")
# The scheduling fixture's message arrived Thursday 19 Nov 2026, 07:30 UTC.
NOW = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)

Scripts = Mapping[str, Mapping[Purpose, Sequence[dict[str, Any]]]]


@pytest.fixture(scope="session")
def checkpoint_url(db_url: str) -> str:
    """Plain `postgresql://` URL of the test DB, with LangGraph's checkpoint tables set up."""
    url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    async def setup() -> None:
        async with AsyncPostgresSaver.from_conn_string(url) as saver:
            await saver.setup()

    asyncio.run(setup())
    return url


def _truncate_checkpoints(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f"TRUNCATE {', '.join(CHECKPOINT_TABLES)}")


@pytest.fixture
def checkpoints(checkpoint_url: str) -> Iterator[str]:
    _truncate_checkpoints(checkpoint_url)
    yield checkpoint_url
    _truncate_checkpoints(checkpoint_url)


@pytest.fixture
def worker_settings(base_env: None, db_url: str, checkpoints: str) -> Settings:
    return Settings(
        _env_file=None,
        database_url=db_url,
        checkpoint_database_url=checkpoints,
        outbound_execution_enabled=True,
    )


@asynccontextmanager
async def open_graph(url: str) -> AsyncIterator[Graph]:
    """One worker's checkpointer and compiled graph; closing it is the worker stopping."""
    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        yield build_parent_graph(saver)


@pytest.fixture
def faults() -> MockFaults:
    return MockFaults()


@pytest.fixture
def mail(sessionmaker: async_sessionmaker[AsyncSession], faults: MockFaults) -> MockMailProvider:
    return MockMailProvider(sessionmaker, faults)


class DepsBuilder:
    """Fresh `GraphDeps` per worker: a scripted model, the mock providers and a fixed clock."""

    def __init__(
        self,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        mail: MockMailProvider,
        faults: MockFaults,
    ) -> None:
        self.settings = settings
        self.sessionmaker = sessionmaker
        self.mail = mail
        self.faults = faults

    def __call__(
        self,
        scripts: Scripts | None = None,
        *,
        llm: LLMClient | None = None,
        mail: MailProvider | None = None,
    ) -> GraphDeps:
        return GraphDeps(
            llm=llm or ScriptedLLMClient(scripts or {}),
            mail=mail or self.mail,
            calendar=MockCalendarProvider(self.sessionmaker, None, "UTC", self.faults),
            sessionmaker=self.sessionmaker,
            settings=self.settings,
            now=lambda: NOW,
        )


@pytest.fixture
def make_deps(
    worker_settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    mail: MockMailProvider,
    faults: MockFaults,
) -> DepsBuilder:
    return DepsBuilder(worker_settings, sessionmaker, mail, faults)
