"""The worker runs `index_message` jobs: success, RAG disabled, transient errors, budget."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import EmailMessage, Job, MessageChunk, WorkflowRun
from app.ingestion.ingest import ingest_message
from app.jobs.worker import run_once
from app.llm.client import BudgetExceeded
from app.providers.interfaces import NormalizedMessage
from app.retrieval.embeddings import (
    EmbeddingTransientError,
    EmbedKind,
    EmbedResult,
    HashEmbeddingsClient,
)
from tests.jobs.conftest import DepsBuilder, open_graph

LEASE = timedelta(seconds=30)


def _msg(mid: str = "m1") -> NormalizedMessage:
    return NormalizedMessage(
        provider_message_id=mid,
        provider_thread_id="t1",
        sender="owner@example.com",  # owner mail: indexed, no workflow started
        to=["alex@example.com"],
        subject="Quote",
        body_text="We agreed on 4500.",
        received_at=datetime(2026, 11, 19, 9, 0, tzinfo=UTC),
    )


async def _seed(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    async with sessionmaker() as session, session.begin():
        await ingest_message(session, _msg(), ["owner@example.com"], index=True)


async def _job(session: AsyncSession) -> Job:
    stmt = select(Job).where(Job.kind == "index_message")
    return (await session.execute(stmt.execution_options(populate_existing=True))).scalar_one()


async def _chunks(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(MessageChunk)) or 0)


class Failing(HashEmbeddingsClient):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> EmbedResult:
        raise self.exc


async def test_worker_indexes_message(
    sessionmaker: async_sessionmaker[AsyncSession],
    session: AsyncSession,
    make_deps: DepsBuilder,
    checkpoints: str,
) -> None:
    await _seed(sessionmaker)
    async with open_graph(checkpoints) as graph:
        deps = make_deps(embeddings=HashEmbeddingsClient())
        assert await run_once(graph, deps, worker_id="w", lease=LEASE)
    assert (await _job(session)).state == "done"
    assert await _chunks(session) == 1
    assert await session.scalar(select(func.count()).select_from(WorkflowRun)) == 0


async def test_rag_disabled_completes_without_indexing(
    sessionmaker: async_sessionmaker[AsyncSession],
    session: AsyncSession,
    make_deps: DepsBuilder,
    checkpoints: str,
) -> None:
    await _seed(sessionmaker)
    async with open_graph(checkpoints) as graph:
        await run_once(graph, make_deps(), worker_id="w", lease=LEASE)
    assert (await _job(session)).state == "done"
    assert await _chunks(session) == 0


async def test_transient_embedding_error_requeues_with_backoff(
    sessionmaker: async_sessionmaker[AsyncSession],
    session: AsyncSession,
    make_deps: DepsBuilder,
    checkpoints: str,
) -> None:
    await _seed(sessionmaker)
    async with open_graph(checkpoints) as graph:
        deps = make_deps(embeddings=Failing(EmbeddingTransientError("x")))
        await run_once(graph, deps, worker_id="w", lease=LEASE)
    job = await _job(session)
    assert (job.state, job.error, job.attempts) == ("queued", "index_transient", 1)
    assert job.scheduled_at > datetime.now(UTC)


async def test_budget_exhausted_waits_without_using_an_attempt(
    sessionmaker: async_sessionmaker[AsyncSession],
    session: AsyncSession,
    make_deps: DepsBuilder,
    checkpoints: str,
) -> None:
    await _seed(sessionmaker)
    async with open_graph(checkpoints) as graph:
        deps = make_deps(embeddings=Failing(BudgetExceeded("x")))
        await run_once(graph, deps, worker_id="w", lease=LEASE)
    job = await _job(session)
    assert (job.state, job.attempts) == ("queued", 0)
    assert job.scheduled_at > datetime.now(UTC) + timedelta(minutes=30)


async def test_definite_failure_fails_the_job_but_no_run(
    sessionmaker: async_sessionmaker[AsyncSession],
    session: AsyncSession,
    make_deps: DepsBuilder,
    checkpoints: str,
) -> None:
    await _seed(sessionmaker)
    async with open_graph(checkpoints) as graph:
        deps = make_deps(embeddings=Failing(ValueError("boom")))
        await run_once(graph, deps, worker_id="w", lease=LEASE)
    job = await _job(session)
    assert (job.state, job.error) == ("failed", "internal_error")
    assert await session.scalar(select(func.count()).select_from(EmailMessage)) == 1
