"""Embedding clients behind one small protocol.

- `FastEmbedClient`: a local ONNX model (default `bge-small-en-v1.5`), free, runs on CPU. The
  model downloads on first use (cache dir: env `FASTEMBED_CACHE_PATH`).
- `VoyageEmbeddingsClient`: the Voyage AI REST API over httpx. Spend is reserved and settled
  through the same `BudgetGuard` as model calls (purpose "embed").
- `HashEmbeddingsClient`: deterministic feature hashing of words. Tests and scripted mode only:
  it captures word overlap, not meaning.

Queries and documents are embedded with `kind` because asymmetric models (both real ones here)
encode them differently.
"""

import asyncio
import hashlib
import logging
import math
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import EMBEDDING_DIMENSIONS
from app.llm.budget import BudgetGuard
from app.llm.client import Usage
from app.llm.pricing import cost_of, price_for

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

EmbedKind = Literal["document", "query"]

# Our model name -> fastembed's model id.
FASTEMBED_MODELS = {"bge-small-en-v1.5": "BAAI/bge-small-en-v1.5"}
VOYAGE_MODELS = frozenset({"voyage-3.5-lite"})
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
_MILLION = Decimal(1_000_000)
_QUANTUM = Decimal("0.000001")


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    model: str
    tokens: int = 0
    cost_usd: Decimal = Decimal("0")


class EmbeddingsClient(Protocol):
    model: str
    dim: int

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> EmbedResult: ...


class EmbeddingError(Exception):
    """An embedding request failed. Messages never contain the API key or input text."""


class EmbeddingTransientError(EmbeddingError):
    """Rate limit, 5xx, timeout or network failure; the caller may retry later."""


# --- hash fake ------------------------------------------------------------------------------

_WORD = re.compile(r"\w+", re.UNICODE)


class HashEmbeddingsClient:
    """Signed feature hashing of lowercase words, L2-normalized. Deterministic across runs."""

    def __init__(self, model: str = "hash-v1", dim: int = 384) -> None:
        self.model = model
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vec[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> EmbedResult:
        return EmbedResult(vectors=[self._vector(t) for t in texts], model=self.model)


# --- fastembed ------------------------------------------------------------------------------

_models: dict[str, "TextEmbedding"] = {}
_models_lock = threading.Lock()


def _load(model_id: str) -> "TextEmbedding":
    with _models_lock:
        if model_id not in _models:
            from fastembed import TextEmbedding

            _models[model_id] = TextEmbedding(model_name=model_id)
        return _models[model_id]


class FastEmbedClient:
    """Local model; loaded on first `embed` (not at construction) and shared per process."""

    def __init__(self, model: str = "bge-small-en-v1.5") -> None:
        if model not in FASTEMBED_MODELS:
            raise ValueError(f"EMBED_MODEL {model!r} is not a fastembed model")
        self.model = model
        self.dim = EMBEDDING_DIMENSIONS[model]
        self._model_id = FASTEMBED_MODELS[model]

    def _embed_sync(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        engine = _load(self._model_id)
        stream = engine.query_embed(texts) if kind == "query" else engine.passage_embed(texts)
        return [vec.tolist() for vec in stream]

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=[], model=self.model)
        vectors = await asyncio.to_thread(self._embed_sync, list(texts), kind)
        return EmbedResult(vectors=vectors, model=self.model)


# --- Voyage ---------------------------------------------------------------------------------


class VoyageEmbeddingsClient:
    def __init__(
        self,
        *,
        api_key: str,
        budget: BudgetGuard,
        model: str = "voyage-3.5-lite",
        batch_size: int = 128,
        timeout_seconds: float = 60.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if model not in VOYAGE_MODELS:
            raise ValueError(f"EMBED_MODEL {model!r} is not a Voyage model")
        self.model = model
        self.dim = EMBEDDING_DIMENSIONS[model]
        self._key = api_key
        self._budget = budget
        self._batch = batch_size
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    def _estimate(self, texts: Sequence[str]) -> Decimal:
        # Conservative: ~3 characters per token, plus one per text; rounded up so a reservation
        # is never free.
        tokens = sum(len(t) // 3 + 1 for t in texts)
        cost = tokens * price_for(self.model).input / _MILLION
        return max(cost.quantize(_QUANTUM, rounding=ROUND_CEILING), _QUANTUM)

    async def _batch_call(
        self, texts: list[str], kind: EmbedKind
    ) -> tuple[list[list[float]], int, Decimal]:
        reservation = await self._budget.reserve("embed", self.model, self._estimate(texts), None)
        try:
            response = await self._http.post(
                VOYAGE_URL,
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "input": texts,
                    "model": self.model,
                    "input_type": kind,
                    "output_dimension": self.dim,
                },
            )
        except httpx.TransportError as exc:
            # The request may have been processed and billed: keep the estimate.
            await asyncio.shield(self._budget.settle_estimate(reservation))
            raise EmbeddingTransientError(f"voyage request failed: {type(exc).__name__}") from None
        except BaseException:
            await asyncio.shield(self._budget.settle_estimate(reservation))
            raise

        if response.status_code != 200:
            # A rejected request is not billed.
            await asyncio.shield(self._budget.settle(reservation, Usage(), self.model))
            status = response.status_code
            if status == 429 or status >= 500:
                raise EmbeddingTransientError(f"voyage returned HTTP {status}")
            raise EmbeddingError(f"voyage returned HTTP {status}")

        try:
            body: dict[str, Any] = response.json()
            tokens = int(body["usage"]["total_tokens"])
            data = sorted(body["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in data]
        except (ValueError, KeyError, TypeError):
            # Billed but unreadable: keep the estimate.
            await asyncio.shield(self._budget.settle_estimate(reservation))
            raise EmbeddingError("voyage returned a malformed response body") from None
        await asyncio.shield(
            self._budget.settle(reservation, Usage(input_tokens=tokens), self.model)
        )
        if len(vectors) != len(texts):
            raise EmbeddingError(f"voyage returned {len(vectors)} vectors for {len(texts)} texts")
        if any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"voyage returned a wrong-dimension vector (want {self.dim})")
        return vectors, tokens, cost_of(self.model, Usage(input_tokens=tokens))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> EmbedResult:
        vectors: list[list[float]] = []
        tokens = 0
        cost = Decimal("0")  # the sum of what each batch settled in the budget ledger
        for start in range(0, len(texts), self._batch):
            batch_vectors, batch_tokens, batch_cost = await self._batch_call(
                list(texts[start : start + self._batch]), kind
            )
            vectors.extend(batch_vectors)
            tokens += batch_tokens
            cost += batch_cost
        return EmbedResult(vectors=vectors, model=self.model, tokens=tokens, cost_usd=cost)


# --- factory --------------------------------------------------------------------------------


def make_embeddings_client(
    settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]
) -> EmbeddingsClient:
    provider = settings.embeddings_provider
    model = settings.embed_model
    if provider == "hash":
        return HashEmbeddingsClient()
    if model not in EMBEDDING_DIMENSIONS:
        raise ValueError(f"EMBED_MODEL {model!r} has no vector index (see EMBEDDING_DIMENSIONS)")
    if provider == "fastembed":
        return FastEmbedClient(model)
    if settings.voyage_api_key is None:
        raise ValueError("EMBEDDINGS_PROVIDER=voyage needs VOYAGE_API_KEY")
    return VoyageEmbeddingsClient(
        api_key=settings.voyage_api_key.get_secret_value(),
        budget=BudgetGuard(sessionmaker, settings.daily_budget_usd),
        model=model,
    )
