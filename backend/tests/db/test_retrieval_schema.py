"""Retrieval schema (migration 0004): pgvector, chunks, per-model embeddings, cascade delete."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    EMBEDDING_DIMENSIONS,
    ChunkEmbedding,
    EmailMessage,
    EmailThread,
    MessageChunk,
    embedding_index_name,
)


async def _message(session: AsyncSession) -> EmailMessage:
    n = uuid.uuid4().hex[:8]
    thread = EmailThread(account_id="mock-account", provider_thread_id=f"t-{n}")
    session.add(thread)
    await session.flush()
    msg = EmailMessage(
        thread_id=thread.id,
        account_id="mock-account",
        provider_message_id=f"m-{n}",
        sender="alice@example.com",
        received_at=datetime.now(UTC),
        content_hash=f"hash-{n}",
    )
    session.add(msg)
    await session.flush()
    return msg


def _chunk(
    msg: EmailMessage, index: int = 0, body: str = "We agreed on 4500 dollars"
) -> MessageChunk:
    return MessageChunk(
        account_id=msg.account_id,
        message_id=msg.id,
        thread_id=msg.thread_id,
        chunk_index=index,
        text=body,
        sender=msg.sender,
        received_at=msg.received_at,
        content_hash=f"c-{index}",
    )


async def test_vector_extension_installed(session: AsyncSession) -> None:
    ext = await session.scalar(text("select extname from pg_extension where extname='vector'"))
    assert ext == "vector"


async def test_tsv_is_generated_from_text(session: AsyncSession) -> None:
    msg = await _message(session)
    chunk = _chunk(msg, body="The invoice deadline is Friday")
    session.add(chunk)
    await session.commit()
    hit = await session.scalar(
        select(func.count())
        .select_from(MessageChunk)
        .where(MessageChunk.tsv.op("@@")(func.websearch_to_tsquery("english", "invoices")))
    )
    assert hit == 1


async def test_chunk_index_unique_per_message(session: AsyncSession) -> None:
    msg = await _message(session)
    session.add(_chunk(msg, 0))
    await session.commit()
    session.add(_chunk(msg, 0))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_embeddings_of_different_dimensions_coexist(session: AsyncSession) -> None:
    msg = await _message(session)
    chunk = _chunk(msg)
    session.add(chunk)
    await session.flush()
    session.add_all(
        [
            ChunkEmbedding(chunk_id=chunk.id, model="bge-small-en-v1.5", embedding=[0.1] * 384),
            ChunkEmbedding(chunk_id=chunk.id, model="voyage-3.5-lite", embedding=[0.1] * 512),
        ]
    )
    await session.commit()
    count = await session.scalar(select(func.count()).select_from(ChunkEmbedding))
    assert count == 2


async def test_deleting_a_message_removes_its_chunks_and_embeddings(
    session: AsyncSession,
) -> None:
    msg = await _message(session)
    chunk = _chunk(msg)
    session.add(chunk)
    await session.flush()
    session.add(ChunkEmbedding(chunk_id=chunk.id, model="bge-small-en-v1.5", embedding=[0.0] * 384))
    await session.commit()

    await session.execute(text("delete from email_messages where id = :id"), {"id": msg.id})
    await session.commit()
    assert await session.scalar(select(func.count()).select_from(MessageChunk)) == 0
    assert await session.scalar(select(func.count()).select_from(ChunkEmbedding)) == 0


async def test_hnsw_index_per_embedding_model(session: AsyncSession) -> None:
    """Every model in EMBEDDING_DIMENSIONS has its partial HNSW index at the right dimension.

    Alembic cannot compare expression indexes, so this is the drift check for them."""
    rows = (
        await session.execute(
            text("select indexname, indexdef from pg_indexes where tablename='chunk_embeddings'")
        )
    ).all()
    hnsw = {name: d for name, d in rows if "hnsw" in d}
    assert set(hnsw) == {embedding_index_name(m) for m in EMBEDDING_DIMENSIONS}
    for model, dim in EMBEDDING_DIMENSIONS.items():
        definition = hnsw[embedding_index_name(model)]
        assert f"vector({dim})" in definition
        assert f"'{model}'" in definition
        assert "vector_cosine_ops" in definition
