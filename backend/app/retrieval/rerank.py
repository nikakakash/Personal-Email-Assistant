"""Rerankers: score (query, passage) pairs jointly, more accurately than embeddings alone."""

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    model: str

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage, higher is better (any scale)."""
        ...
