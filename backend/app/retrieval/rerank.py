"""Rerankers: score (query, passage) pairs jointly, more accurately than embeddings alone.

`FastEmbedReranker` runs a small local cross-encoder (ms-marco MiniLM, ~80 MB, CPU), loaded
on first use and shared per process; scoring runs off the event loop. Search uses it in
`hybrid_rerank` mode on the top ~20 hybrid candidates only, so its cost stays bounded.
"""

import asyncio
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    model: str

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage, higher is better (any scale)."""
        ...


_models: dict[str, "TextCrossEncoder"] = {}
_lock = threading.Lock()


def _load(model: str) -> "TextCrossEncoder":
    with _lock:
        if model not in _models:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            _models[model] = TextCrossEncoder(model_name=model)
        return _models[model]


class FastEmbedReranker:
    def __init__(self, model: str = DEFAULT_RERANK_MODEL) -> None:
        self.model = model

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in _load(self.model).rerank(query, passages)]

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._score_sync, query, list(passages))
