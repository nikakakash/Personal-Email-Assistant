"""search(): keyword (Postgres full text), vector (pgvector), hybrid (RRF), filters, isolation."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import EmailMessage
from app.retrieval.embeddings import HashEmbeddingsClient
from app.retrieval.indexer import index_message
from app.retrieval.search import Hit, rrf_fuse, search

EMB = HashEmbeddingsClient()


async def _add(
    sessionmaker: async_sessionmaker[AsyncSession],
    body: str,
    *,
    account: str = "acct",
    thread: str | None = None,
    sender: str = "alex@example.com",
    subject: str = "Hello",
    day: int = 1,
) -> EmailMessage:
    async with sessionmaker() as session, session.begin():
        provider_thread = thread or f"t-{uuid.uuid4().hex[:6]}"
        row = await session.scalar(
            text(
                "insert into email_threads (account_id, provider_thread_id) values (:a, :t) "
                "on conflict (account_id, provider_thread_id) do update set account_id = :a "
                "returning id"
            ),
            {"a": account, "t": provider_thread},
        )
        msg = EmailMessage(
            thread_id=row,
            account_id=account,
            provider_message_id=f"m-{uuid.uuid4().hex[:8]}",
            sender=sender,
            to_addrs=["owner@example.com"],
            subject=subject,
            body_text=body,
            received_at=datetime(2026, 3, day, tzinfo=UTC),
            content_hash=uuid.uuid4().hex,
        )
        session.add(msg)
    await index_message(sessionmaker, msg.id, EMB)
    return msg


def _ids(hits: list[Hit]) -> list[str]:
    return [h.provider_message_id for h in hits]


@pytest.fixture
async def corpus(sessionmaker: async_sessionmaker[AsyncSession]) -> dict[str, EmailMessage]:
    return {
        "price": await _add(sessionmaker, "We agreed the price is 4500 dollars for the website."),
        "invoice": await _add(sessionmaker, "Invoice INV-7731 is attached, due on May 1."),
        "weather": await _add(sessionmaker, "The weather in Tbilisi was lovely this weekend."),
        "lunch": await _add(sessionmaker, "Shall we have lunch on Friday near the office?"),
    }


def test_rrf_fuse_rewards_agreement() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert [key for key, _ in fused][:2] == ["b", "c"]
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf_fuse([], k=60) == []


async def test_keyword_exact_identifier(
    session: AsyncSession, corpus: dict[str, EmailMessage]
) -> None:
    hits = await search(session, EMB, account_id="acct", query="INV-7731", mode="keyword")
    assert _ids(hits)[0] == corpus["invoice"].provider_message_id


async def test_keyword_matches_any_word_of_a_question(
    session: AsyncSession, corpus: dict[str, EmailMessage]
) -> None:
    hits = await search(
        session, EMB, account_id="acct", query="what price did we agree on?", mode="keyword"
    )
    assert _ids(hits)[0] == corpus["price"].provider_message_id
    assert hits[0].keyword_rank == 1 and hits[0].vector_rank is None


async def test_vector_mode(session: AsyncSession, corpus: dict[str, EmailMessage]) -> None:
    hits = await search(session, EMB, account_id="acct", query="lunch Friday", mode="vector")
    assert _ids(hits)[0] == corpus["lunch"].provider_message_id
    assert hits[0].vector_rank == 1


async def test_hybrid_combines_both(session: AsyncSession, corpus: dict[str, EmailMessage]) -> None:
    hits = await search(session, EMB, account_id="acct", query="price agreed 4500", k=2)
    assert _ids(hits)[0] == corpus["price"].provider_message_id
    assert hits[0].keyword_rank == 1 and hits[0].vector_rank == 1
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


async def test_other_accounts_never_leak(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    await _add(sessionmaker, "secret merger price 9999", account="other")
    for mode in ("keyword", "vector", "hybrid"):
        hits = await search(
            session, EMB, account_id="acct", query="secret merger price 9999", mode=mode
        )
        assert hits == [], mode


async def test_account_is_mandatory(session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="account_id"):
        await search(session, EMB, account_id="", query="x")


async def test_current_thread_is_excluded(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    await _add(sessionmaker, "price 4500 confirmed", thread="current")
    older = await _add(sessionmaker, "price 4500 quoted earlier", thread="old")
    hits = await search(
        session, EMB, account_id="acct", query="price 4500", exclude_provider_thread_id="current"
    )
    assert _ids(hits) == [older.provider_message_id]
    assert hits[0].provider_thread_id == "old"


async def test_sender_and_date_filters(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    await _add(sessionmaker, "contract draft v1", sender="bob@example.com", day=1)
    late = await _add(sessionmaker, "contract draft v2", sender="alex@example.com", day=20)
    by_sender = await search(
        session, EMB, account_id="acct", query="contract draft", sender="ALEX@example.com"
    )
    assert _ids(by_sender) == [late.provider_message_id]
    since = await search(
        session,
        EMB,
        account_id="acct",
        query="contract draft",
        since=datetime(2026, 3, 10, tzinfo=UTC),
    )
    assert _ids(since) == [late.provider_message_id]


async def test_one_hit_per_message_with_its_best_chunk(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    filler = "\n\n".join(" ".join(f"filler{p}x{i}" for i in range(200)) for p in range(3))
    long = await _add(sessionmaker, filler + "\n\nThe deposit is 1200 euros.")
    hits = await search(session, EMB, account_id="acct", query="deposit 1200 euros", k=5)
    assert _ids(hits) == [long.provider_message_id]
    assert "deposit is 1200" in hits[0].text
    assert hits[0].subject == "Hello" and hits[0].sender == "alex@example.com"


async def test_hostile_query_text_is_just_text(
    session: AsyncSession, corpus: dict[str, EmailMessage]
) -> None:
    for query in ["'); drop table email_messages; --", "a & | ! <-> :*", "", "   ", "the of and"]:
        hits = await search(session, EMB, account_id="acct", query=query)
        assert isinstance(hits, list)
    assert await session.scalar(text("select count(*) from email_messages")) == 4


async def test_hybrid_rerank_needs_a_reranker(session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="reranker"):
        await search(session, EMB, account_id="acct", query="x", mode="hybrid_rerank")


async def test_vector_query_uses_the_model_hnsw_index_at_realistic_size(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    """With a few thousand vectors (and statistics), the planner walks the partial HNSW index;
    this proves the query's expression and predicate match the index definition."""
    from app.retrieval.search import vector_candidates_sql

    msg = await _add(sessionmaker, "seed")
    await session.execute(
        text(
            "insert into message_chunks (account_id, message_id, thread_id, chunk_index, text, "
            "sender, received_at, content_hash) "
            "select 'acct', :m, :t, g, 'chunk ' || g, 'a@x.com', now(), md5(g::text) "
            "from generate_series(1, 3000) g"
        ),
        {"m": msg.id, "t": msg.thread_id},
    )
    await session.execute(
        text(
            "insert into chunk_embeddings (chunk_id, model, embedding) "
            "select c.id, 'bge-small-en-v1.5', "
            "(select array_agg(random())::real[]::vector from generate_series(1, 384) "
            " where c.id is not null) "
            "from message_chunks c where c.chunk_index > 0"
        )
    )
    await session.commit()
    await session.execute(text("analyze message_chunks"))
    await session.execute(text("analyze chunk_embeddings"))
    sql, params = vector_candidates_sql(
        model="bge-small-en-v1.5", dim=384, vector=[0.1] * 384, account_id="acct", limit=10
    )
    plan = "\n".join((await session.execute(text("explain " + sql), params)).scalars().all())
    assert "ix_chunk_embeddings_hnsw_bge_small_en_v1_5" in plan, plan


async def test_hybrid_rerank_reorders_by_reranker_score(
    session: AsyncSession, corpus: dict[str, EmailMessage]
) -> None:
    class PreferLunch:
        model = "fake-reranker"

        async def score(self, query: str, passages: Sequence[str]) -> list[float]:
            return [10.0 if "lunch" in p else 0.0 for p in passages]

    hits = await search(
        session,
        EMB,
        account_id="acct",
        query="price lunch",
        k=2,
        mode="hybrid_rerank",
        reranker=PreferLunch(),
    )
    assert _ids(hits)[0] == corpus["lunch"].provider_message_id
    assert hits[0].score == 10.0 and len(hits) == 2


async def test_participant_scope_only_returns_their_mail(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    alex = await _add(sessionmaker, "price agreed at 4500", sender="alex@example.com")
    await _add(sessionmaker, "price agreed at 7000", sender="bob@example.com")
    # The owner wrote to Alex: Alex is a recipient, given with a display name.
    async with sessionmaker() as s, s.begin():
        await s.execute(
            text(
                "update email_messages set to_addrs = array['Alex A <ALEX@example.com>'] "
                "where id = :id"
            ),
            {"id": alex.id},
        )
    to_alex = await _add(sessionmaker, "confirming the price of 4500", sender="owner@example.com")
    async with sessionmaker() as s, s.begin():
        await s.execute(
            text(
                "update email_messages set to_addrs = array['Alex <alex@example.com>'] "
                "where id = :id"
            ),
            {"id": to_alex.id},
        )
    for mode in ("keyword", "vector", "hybrid"):
        hits = await search(
            session,
            EMB,
            account_id="acct",
            query="price agreed",
            mode=mode,
            participants=["Alex@Example.com"],
        )
        assert set(_ids(hits)) == {alex.provider_message_id, to_alex.provider_message_id}, mode


async def test_empty_participant_scope_returns_nothing(
    session: AsyncSession, corpus: dict[str, EmailMessage]
) -> None:
    assert await search(session, EMB, account_id="acct", query="price", participants=[]) == []
