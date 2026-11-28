"""index_message / reindex_all: chunks + embeddings in the database, incremental and idempotent."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ChunkEmbedding, EmailMessage, EmailThread, MessageChunk
from app.retrieval.embeddings import EmbedKind, EmbedResult, HashEmbeddingsClient
from app.retrieval.indexer import index_message, reindex_all


class CountingEmbedder(HashEmbeddingsClient):
    """The hash fake, recording every text it embeds."""

    def __init__(self, model: str = "hash-v1") -> None:
        super().__init__(model=model)
        self.calls: list[tuple[EmbedKind, list[str]]] = []

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> EmbedResult:
        self.calls.append((kind, list(texts)))
        return await super().embed(texts, kind=kind)

    @property
    def embedded(self) -> int:
        return sum(len(t) for _, t in self.calls)


async def _message(
    session: AsyncSession, body: str = "The agreed price is 4500.", account: str = "mock-account"
) -> EmailMessage:
    n = uuid.uuid4().hex[:8]
    thread = EmailThread(account_id=account, provider_thread_id=f"t-{n}")
    session.add(thread)
    await session.flush()
    msg = EmailMessage(
        thread_id=thread.id,
        account_id=account,
        provider_message_id=f"m-{n}",
        sender="alex@example.com",
        to_addrs=["owner@example.com"],
        subject="Quote",
        body_text=body,
        received_at=datetime(2026, 3, 4, tzinfo=UTC),
        content_hash=n,
    )
    session.add(msg)
    await session.commit()
    return msg


async def _counts(session: AsyncSession) -> tuple[int, int]:
    chunks = await session.scalar(select(func.count()).select_from(MessageChunk))
    embeddings = await session.scalar(select(func.count()).select_from(ChunkEmbedding))
    return int(chunks or 0), int(embeddings or 0)


LONG = "\n\n".join(" ".join(f"p{p}w{i}" for i in range(200)) for p in range(4))


async def test_indexes_chunks_and_embeddings(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(session)
    embedder = CountingEmbedder()
    stats = await index_message(sessionmaker, msg.id, embedder)

    assert (stats.chunks, stats.embedded) == (1, 1)
    chunk = (await session.execute(select(MessageChunk))).scalar_one()
    assert chunk.account_id == "mock-account" and chunk.thread_id == msg.thread_id
    assert chunk.sender == "alex@example.com" and "4500" in chunk.text
    emb = (await session.execute(select(ChunkEmbedding))).scalar_one()
    assert emb.model == "hash-v1" and len(emb.embedding) == 384
    assert embedder.calls[0][0] == "document"


async def test_second_run_embeds_nothing(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(session, LONG)
    embedder = CountingEmbedder()
    first = await index_message(sessionmaker, msg.id, embedder)
    again = await index_message(sessionmaker, msg.id, embedder)
    assert first.chunks > 1 and first.embedded == first.chunks
    assert again.embedded == 0 and again.chunks == first.chunks
    assert await _counts(session) == (first.chunks, first.chunks)


async def test_changed_body_reembeds_only_changed_chunks_and_drops_extras(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(session, LONG)
    embedder = CountingEmbedder()
    first = await index_message(sessionmaker, msg.id, embedder)
    await session.execute(
        update(EmailMessage).where(EmailMessage.id == msg.id).values(body_text="Short now.")
    )
    await session.commit()

    second = await index_message(sessionmaker, msg.id, embedder)
    assert first.chunks > 1 and second.chunks == 1 and second.embedded == 1
    assert await _counts(session) == (1, 1)
    chunk = (await session.execute(select(MessageChunk))).scalar_one()
    assert chunk.text.endswith("Short now.")


async def test_another_model_adds_embeddings_and_keeps_existing(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(session)
    await index_message(sessionmaker, msg.id, CountingEmbedder("hash-v1"))
    other = CountingEmbedder("hash-other")
    stats = await index_message(sessionmaker, msg.id, other)
    assert stats.embedded == 1
    models = set((await session.execute(select(ChunkEmbedding.model))).scalars())
    assert models == {"hash-v1", "hash-other"}


async def test_missing_message_is_a_noop(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    stats = await index_message(sessionmaker, uuid.uuid4(), CountingEmbedder())
    assert stats.missing and stats.chunks == 0


async def test_message_deleted_while_embedding_writes_nothing(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(session)

    class DeletingEmbedder(CountingEmbedder):
        async def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> EmbedResult:
            async with sessionmaker() as s, s.begin():
                await s.delete(await s.get(EmailMessage, msg.id))
            return await super().embed(texts, kind=kind)

    stats = await index_message(sessionmaker, msg.id, DeletingEmbedder())
    assert stats.missing
    assert await _counts(session) == (0, 0)


async def test_reindex_all_filters_by_account_and_force_reembeds(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    await _message(session, account="a1")
    await _message(session, account="a1")
    await _message(session, account="a2")
    embedder = CountingEmbedder()

    total = await reindex_all(sessionmaker, embedder, account_id="a1")
    assert (total.messages, total.embedded) == (2, 2)
    assert (await reindex_all(sessionmaker, embedder, account_id="a1")).embedded == 0
    assert (await reindex_all(sessionmaker, embedder, account_id="a1", force=True)).embedded == 2
    assert (await reindex_all(sessionmaker, embedder)).embedded == 1  # only a2 was new


async def test_empty_message_indexes_nothing(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(session, body="")
    await session.execute(update(EmailMessage).where(EmailMessage.id == msg.id).values(subject=""))
    await session.commit()
    stats = await index_message(sessionmaker, msg.id, CountingEmbedder())
    assert stats.chunks == 0
    assert await _counts(session) == (0, 0)


@pytest.mark.parametrize("prefix", [None, "Thread with Alex about the March quote."])
async def test_embedded_text_includes_context_prefix(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession, prefix: str | None
) -> None:
    msg = await _message(session)
    embedder = CountingEmbedder()
    await index_message(sessionmaker, msg.id, embedder)
    if prefix is None:
        assert embedder.calls[0][1][0].startswith("From: alex@example.com")
        return
    await session.execute(update(MessageChunk).values(context_prefix=prefix))
    await session.execute(ChunkEmbedding.__table__.delete())
    await session.commit()
    await index_message(sessionmaker, msg.id, embedder)
    assert embedder.calls[-1][1][0].startswith(prefix + "\n")
