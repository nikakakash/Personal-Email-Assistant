"""Write one email's chunks and embeddings to the retrieval index.

`index_message` is incremental and safe to repeat (index jobs may run twice):

1. Read the message and its current chunks in a short session.
2. Chunk the message; only chunks whose text changed, or that lack an embedding for this
   model, are embedded. The embedding call runs outside any transaction (it may take seconds).
3. Write in one transaction that first locks the message row: if the message was deleted
   meanwhile nothing is written; a concurrent indexer of the same message waits its turn.
   Chunks beyond the new count are removed, and a chunk whose text changed loses all its old
   embeddings (any model) because they describe text that no longer exists.
"""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ChunkEmbedding, EmailMessage, MessageChunk
from app.llm.client import BudgetExceeded, LLMError
from app.mail.content import model_text
from app.retrieval.chunking import Chunk, chunk_message
from app.retrieval.contextualize import Contextualizer, MessageInfo
from app.retrieval.embeddings import EmbeddingsClient

log = logging.getLogger(__name__)


@dataclass
class IndexStats:
    chunks: int = 0
    embedded: int = 0
    missing: bool = False


@dataclass
class ReindexStats:
    messages: int = 0
    chunks: int = 0
    embedded: int = 0


def embedding_text(chunk_text: str, context_prefix: str | None) -> str:
    """What is embedded: the optional situating prefix, then the chunk (header + body)."""
    return f"{context_prefix}\n{chunk_text}" if context_prefix else chunk_text


async def index_message(
    sessionmaker: async_sessionmaker[AsyncSession],
    message_id: uuid.UUID,
    embedder: EmbeddingsClient,
    contextualizer: Contextualizer | None = None,
) -> IndexStats:
    """With `contextualizer`, chunks that are new or changed (and so have no prefix yet) get
    a model-written situating sentence before embedding. A failure there only skips the prefix."""
    async with sessionmaker() as session:
        msg = await session.get(EmailMessage, message_id)
        if msg is None:
            return IndexStats(missing=True)
        chunks = chunk_message(
            sender=msg.sender,
            to=msg.to_addrs,
            subject=msg.subject,
            received_at=msg.received_at,
            body_text=msg.body_text,
            body_html=msg.body_html,
        )
        info = MessageInfo(
            provider_message_id=msg.provider_message_id,
            sender=msg.sender,
            to=list(msg.to_addrs),
            subject=msg.subject,
            date=msg.received_at.date().isoformat(),
            body=model_text(msg.body_text, msg.body_html),
        )
        snapshot = {
            "account_id": msg.account_id,
            "thread_id": msg.thread_id,
            "sender": msg.sender,
            "received_at": msg.received_at,
        }
        existing = {
            c.chunk_index: c
            for c in (
                await session.execute(
                    select(MessageChunk).where(MessageChunk.message_id == message_id)
                )
            ).scalars()
        }
        embedded_ids = set(
            (
                await session.execute(
                    select(ChunkEmbedding.chunk_id).where(
                        ChunkEmbedding.chunk_id.in_([c.id for c in existing.values()]),
                        ChunkEmbedding.model == embedder.model,
                    )
                )
            ).scalars()
        )

    def unchanged(chunk: Chunk) -> bool:
        old = existing.get(chunk.index)
        return old is not None and old.content_hash == chunk.content_hash

    prefixes = {c.index: existing[c.index].context_prefix for c in chunks if unchanged(c)}

    def prefix(chunk: Chunk) -> str | None:
        return prefixes.get(chunk.index)

    todo = [c for c in chunks if not (unchanged(c) and existing[c.index].id in embedded_ids)]
    if contextualizer is not None:
        for chunk in todo:
            if prefix(chunk) is None:
                prefixes[chunk.index] = await _contextualize(contextualizer, info, chunk)
    vectors: list[list[float]] = []
    if todo:
        texts = [embedding_text(c.text, prefix(c)) for c in todo]
        vectors = (await embedder.embed(texts, kind="document")).vectors

    async with sessionmaker() as session, session.begin():
        locked = await session.scalar(
            select(EmailMessage.id).where(EmailMessage.id == message_id).with_for_update()
        )
        if locked is None:
            return IndexStats(missing=True)
        await session.execute(
            delete(MessageChunk).where(
                MessageChunk.message_id == message_id, MessageChunk.chunk_index >= len(chunks)
            )
        )
        chunk_ids: dict[int, uuid.UUID] = {}
        for chunk in chunks:
            if not unchanged(chunk) and chunk.index in existing:
                await session.execute(
                    delete(ChunkEmbedding).where(
                        ChunkEmbedding.chunk_id == existing[chunk.index].id
                    )
                )
            values = {
                **snapshot,
                "message_id": message_id,
                "chunk_index": chunk.index,
                "text": chunk.text,
                "content_hash": chunk.content_hash,
                "context_prefix": prefix(chunk),
            }
            ins = insert(MessageChunk).values(**values)
            updatable = [k for k in values if k not in ("message_id", "chunk_index")]
            upsert = ins.on_conflict_do_update(
                index_elements=["message_id", "chunk_index"],
                set_={k: ins.excluded[k] for k in updatable},
            ).returning(MessageChunk.id)
            chunk_ids[chunk.index] = (await session.execute(upsert)).scalar_one()
        for chunk, vector in zip(todo, vectors, strict=True):
            emb = insert(ChunkEmbedding).values(
                chunk_id=chunk_ids[chunk.index], model=embedder.model, embedding=vector
            )
            await session.execute(
                emb.on_conflict_do_update(
                    index_elements=["chunk_id", "model"],
                    set_={"embedding": emb.excluded.embedding},
                )
            )
    return IndexStats(chunks=len(chunks), embedded=len(todo))


async def _contextualize(
    contextualizer: Contextualizer, info: MessageInfo, chunk: Chunk
) -> str | None:
    try:
        return await contextualizer(info, chunk.text) or None
    except (BudgetExceeded, LLMError) as exc:
        log.warning("contextualize skipped (%s); indexing without a prefix", type(exc).__name__)
        return None


async def reindex_all(
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: EmbeddingsClient,
    *,
    account_id: str | None = None,
    force: bool = False,
    contextualizer: Contextualizer | None = None,
) -> ReindexStats:
    """Index every stored message (optionally of one account). `force` first drops this
    model's embeddings so everything is embedded again (after a chunking or model change);
    with a `contextualizer` it also drops the prefixes so they are written again."""
    query = select(EmailMessage.id).order_by(EmailMessage.received_at, EmailMessage.id)
    if account_id is not None:
        query = query.where(EmailMessage.account_id == account_id)
    async with sessionmaker() as session, session.begin():
        ids = list((await session.execute(query)).scalars())
        if force:
            scope = select(MessageChunk.id).where(MessageChunk.message_id.in_(query))
            await session.execute(
                delete(ChunkEmbedding).where(
                    ChunkEmbedding.model == embedder.model, ChunkEmbedding.chunk_id.in_(scope)
                )
            )
            if contextualizer is not None:
                await session.execute(
                    update(MessageChunk)
                    .where(MessageChunk.message_id.in_(query))
                    .values(context_prefix=None)
                )
    total = ReindexStats()
    for n, message_id in enumerate(ids, 1):
        stats = await index_message(sessionmaker, message_id, embedder, contextualizer)
        if not stats.missing:
            total.messages += 1
            total.chunks += stats.chunks
            total.embedded += stats.embedded
        if n % 500 == 0:
            log.info("reindex: %d/%d messages", n, len(ids))
    return total
