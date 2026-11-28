"""Embedding clients: the deterministic hash fake, the Voyage adapter (HTTP mocked) and the
local fastembed model (opt-in: set EMBED_MODEL_TESTS=1, it downloads ~130 MB once)."""

import json
import math
import os
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import LlmUsage
from app.llm.budget import BudgetGuard
from app.llm.client import BudgetExceeded
from app.retrieval.embeddings import (
    EmbeddingError,
    EmbeddingTransientError,
    HashEmbeddingsClient,
    VoyageEmbeddingsClient,
)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# --- hash fake ------------------------------------------------------------------------------


async def test_hash_client_is_deterministic_normalized_and_sized() -> None:
    client = HashEmbeddingsClient()
    a = await client.embed(["Invoice for March", "Invoice for March"], kind="document")
    assert a.model == client.model and client.dim == 384
    assert a.vectors[0] == a.vectors[1]
    assert len(a.vectors[0]) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in a.vectors[0])), 1.0, rel_tol=1e-9)
    assert a.cost_usd == Decimal("0")


async def test_hash_client_puts_word_overlap_closer() -> None:
    client = HashEmbeddingsClient()
    q = (await client.embed(["price agreed with Alex"], kind="query")).vectors[0]
    docs = await client.embed(
        ["Alex and I agreed the price is 4500", "The weather in Tbilisi is sunny"],
        kind="document",
    )
    assert _cos(q, docs.vectors[0]) > _cos(q, docs.vectors[1])


async def test_hash_client_empty_input() -> None:
    result = await HashEmbeddingsClient().embed([], kind="document")
    assert result.vectors == []


# --- Voyage ---------------------------------------------------------------------------------


def _voyage(
    sessionmaker: async_sessionmaker[AsyncSession], limit: str = "1.00", batch_size: int = 128
) -> VoyageEmbeddingsClient:
    return VoyageEmbeddingsClient(
        api_key="test-key",
        budget=BudgetGuard(sessionmaker, Decimal(limit)),
        batch_size=batch_size,
    )


def _reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    n = len(body["input"])
    # Out of order on purpose: the client must sort by index.
    data = [{"index": i, "embedding": [float(i)] * 512} for i in reversed(range(n))]
    return httpx.Response(
        200, json={"data": data, "model": body["model"], "usage": {"total_tokens": 10 * n}}
    )


@respx.mock
async def test_voyage_request_shape_and_ordering(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    route = respx.post(VOYAGE_URL).mock(side_effect=_reply)
    result = await _voyage(sessionmaker).embed(["a", "b", "c"], kind="query")

    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "input": ["a", "b", "c"],
        "model": "voyage-3.5-lite",
        "input_type": "query",
        "output_dimension": 512,
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
    assert [v[0] for v in result.vectors] == [0.0, 1.0, 2.0]
    assert result.tokens == 30


@respx.mock
async def test_voyage_batches_and_settles_actual_usage(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    route = respx.post(VOYAGE_URL).mock(side_effect=_reply)
    texts = [f"t{i}" for i in range(300)]
    result = await _voyage(sessionmaker, batch_size=128).embed(texts, kind="document")

    assert route.call_count == 3
    assert len(result.vectors) == 300
    assert [v[0] for v in result.vectors[126:130]] == [126.0, 127.0, 0.0, 1.0]
    rows = (await session.execute(select(LlmUsage))).scalars().all()
    assert {r.purpose for r in rows} == {"embed"}
    assert all(r.status == "settled" for r in rows)
    assert sum(r.input_tokens for r in rows) == 3000
    assert result.cost_usd == sum((r.cost_usd for r in rows), Decimal("0"))


async def test_voyage_refuses_when_budget_exhausted(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(VOYAGE_URL).mock(side_effect=_reply)
        with pytest.raises(BudgetExceeded):
            await _voyage(sessionmaker, limit="0").embed(["x" * 4000], kind="document")
        assert route.call_count == 0


@respx.mock
async def test_voyage_rate_limit_is_transient_and_not_billed(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    respx.post(VOYAGE_URL).mock(return_value=httpx.Response(429, json={"detail": "slow down"}))
    with pytest.raises(EmbeddingTransientError):
        await _voyage(sessionmaker).embed(["hi"], kind="document")
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert row.status == "settled" and row.cost_usd == 0


@respx.mock
async def test_voyage_client_error_is_definite(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    respx.post(VOYAGE_URL).mock(return_value=httpx.Response(400, json={"detail": "bad"}))
    with pytest.raises(EmbeddingError) as info:
        await _voyage(sessionmaker).embed(["hi"], kind="document")
    assert not isinstance(info.value, EmbeddingTransientError)
    assert "test-key" not in str(info.value)


@respx.mock
async def test_voyage_timeout_keeps_the_estimate(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    respx.post(VOYAGE_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(EmbeddingTransientError):
        await _voyage(sessionmaker).embed(["hello world"], kind="document")
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert row.status == "settled" and row.cost_usd > 0  # may have been billed: fail closed


@respx.mock
async def test_voyage_malformed_body_is_an_error_and_keeps_the_estimate(
    sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    respx.post(VOYAGE_URL).mock(return_value=httpx.Response(200, text="not json"))
    with pytest.raises(EmbeddingError, match="malformed"):
        await _voyage(sessionmaker).embed(["hi"], kind="document")
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert row.status == "settled" and row.cost_usd > 0


@respx.mock
async def test_voyage_rejects_wrong_dimension(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    respx.post(VOYAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [0.1] * 10}],
                "model": "voyage-3.5-lite",
                "usage": {"total_tokens": 1},
            },
        )
    )
    with pytest.raises(EmbeddingError, match="dimension"):
        await _voyage(sessionmaker).embed(["hi"], kind="document")


# --- fastembed (real local model, opt-in) ----------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("EMBED_MODEL_TESTS"), reason="set EMBED_MODEL_TESTS=1 to run"
)
async def test_fastembed_semantic_similarity() -> None:
    from app.retrieval.embeddings import FastEmbedClient

    client = FastEmbedClient("bge-small-en-v1.5")
    q = (await client.embed(["how much will the project cost?"], kind="query")).vectors[0]
    docs = await client.embed(
        ["The total price for the work is $4,500.", "My cat likes to sleep on the sofa."],
        kind="document",
    )
    assert len(q) == client.dim == 384
    assert _cos(q, docs.vectors[0]) > _cos(q, docs.vectors[1])
