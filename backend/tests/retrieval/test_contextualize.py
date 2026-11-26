"""Contextual retrieval and reranking: prefixes are written, indexed, never shown, and optional."""

import os
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import EmailMessage, EmailThread, MessageChunk
from app.llm.client import BudgetExceeded, LLMCall, LLMResult, LLMTransientError, T
from app.llm.scripted import ScriptedLLMClient
from app.retrieval.contextualize import LLMContextualizer, MessageInfo
from app.retrieval.embeddings import HashEmbeddingsClient
from app.retrieval.indexer import index_message, reindex_all
from app.retrieval.search import search

EMB = HashEmbeddingsClient()
INFO = MessageInfo("m1", "alex@example.com", ["owner@example.com"], "Quote", "2026-06-05", "x")


class Recording(ScriptedLLMClient):
    def __init__(self, contexts: dict[str, list[str]]) -> None:
        super().__init__(
            {k: {"contextualize": [{"context": c} for c in v]} for k, v in contexts.items()}
        )
        self.calls: list[LLMCall[object]] = []  # type: ignore[type-var]

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        self.calls.append(call)  # type: ignore[arg-type]
        return await super().structured(call)


class Failing:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def __call__(self, message: MessageInfo, chunk_text: str) -> str:
        self.calls += 1
        raise self.exc


async def _message(sessionmaker: async_sessionmaker[AsyncSession], body: str) -> EmailMessage:
    n = uuid.uuid4().hex[:8]
    async with sessionmaker() as s, s.begin():
        thread = EmailThread(account_id="acct", provider_thread_id=f"t-{n}")
        s.add(thread)
        await s.flush()
        msg = EmailMessage(
            thread_id=thread.id,
            account_id="acct",
            provider_message_id=f"m-{n}",
            sender="alex@example.com",
            to_addrs=["owner@example.com"],
            subject="Re: quote",
            body_text=body,
            received_at=datetime(2026, 6, 5, tzinfo=UTC),
            content_hash=n,
        )
        s.add(msg)
    return msg


async def test_llm_contextualizer_call_shape() -> None:
    llm = Recording({"m1": ["  Alex  revises the\nwebsite quote to 5,200.  "]})
    out = await LLMContextualizer(llm)(INFO, "Yes, 5,200 works.")
    assert out == "Alex revises the website quote to 5,200."
    (call,) = llm.calls
    assert call.purpose == "contextualize" and call.max_tokens == 200
    assert "From: alex@example.com" in call.cached_context  # the whole email, cacheable
    assert call.user_content == "<chunk>\nYes, 5,200 works.\n</chunk>"


async def test_prefix_is_indexed_but_never_returned(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(sessionmaker, "Yes, that works for us.")
    llm = Recording({msg.provider_message_id: ["Brightweb website redesign quote accepted"]})
    await index_message(sessionmaker, msg.id, EMB, LLMContextualizer(llm))

    chunk = (await session.execute(select(MessageChunk))).scalar_one()
    assert chunk.context_prefix == "Brightweb website redesign quote accepted"
    hits = await search(session, EMB, account_id="acct", query="Brightweb redesign", mode="keyword")
    assert [h.provider_message_id for h in hits] == [msg.provider_message_id]  # found via prefix
    assert "Brightweb" not in hits[0].text  # ...but the prefix is not handed to the drafter


@pytest.mark.parametrize("exc", [LLMTransientError("down"), BudgetExceeded("spent")])
async def test_contextualize_failure_still_indexes(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession, exc: Exception
) -> None:
    msg = await _message(sessionmaker, "The deposit is 1200.")
    stats = await index_message(sessionmaker, msg.id, EMB, Failing(exc))
    assert stats.embedded == 1
    chunk = (await session.execute(select(MessageChunk))).scalar_one()
    assert chunk.context_prefix is None


async def test_unchanged_chunks_are_not_recontextualized_unless_forced(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    msg = await _message(sessionmaker, "The deposit is 1200.")
    llm = Recording({msg.provider_message_id: ["first context", "second context"]})
    ctx = LLMContextualizer(llm)
    await index_message(sessionmaker, msg.id, EMB, ctx)
    await index_message(sessionmaker, msg.id, EMB, ctx)
    assert len(llm.calls) == 1

    await reindex_all(sessionmaker, EMB, account_id="acct", force=True, contextualizer=ctx)
    assert len(llm.calls) == 2
    chunk = (
        await session.execute(select(MessageChunk).execution_options(populate_existing=True))
    ).scalar_one()
    assert chunk.context_prefix == "second context"


def test_contextualize_uses_its_own_cheap_model(base_env: None) -> None:
    from anthropic import AsyncAnthropic

    from app.llm.anthropic_client import AnthropicLLMClient

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    client = AnthropicLLMClient(AsyncAnthropic(api_key="x"), settings, None)  # type: ignore[arg-type]
    assert client._model_and_effort("contextualize") == ("claude-haiku-5-5", "low")
    assert client._model_and_effort("draft_reply") == ("claude-opus-5-5", "medium")
    assert client._model_and_effort("judge") == ("claude-sonnet-5-5", "low")


def test_worker_deps_follow_settings(
    base_env: None, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    from app.jobs.worker import make_deps
    from app.providers.mock_mail import MockFaults
    from app.retrieval.rerank import FastEmbedReranker

    def deps(**kw: object):  # type: ignore[no-untyped-def]
        settings = Settings(_env_file=None, llm_provider="scripted", **kw)  # type: ignore[arg-type]
        return make_deps(settings, sessionmaker, MockFaults())

    off = deps()
    assert (off.embeddings, off.reranker, off.contextualizer) == (None, None, None)
    on = deps(
        rag_enabled=True,
        embeddings_provider="hash",
        retrieval_mode="hybrid_rerank",
        rag_contextualize=True,
    )
    assert on.embeddings is not None
    assert isinstance(on.reranker, FastEmbedReranker)
    assert isinstance(on.contextualizer, LLMContextualizer)
    assert deps(rag_enabled=True, embeddings_provider="hash").reranker is None


@pytest.mark.skipif(
    not os.environ.get("EMBED_MODEL_TESTS"), reason="set EMBED_MODEL_TESTS=1 to run"
)
async def test_fastembed_reranker_prefers_the_answer() -> None:
    from app.retrieval.rerank import FastEmbedReranker

    scores = await FastEmbedReranker().score(
        "what did the kitchen renovation cost?",
        ["My cat sleeps on the sofa.", "The kitchen renovation estimate is $18,400."],
    )
    assert scores[1] > scores[0]
