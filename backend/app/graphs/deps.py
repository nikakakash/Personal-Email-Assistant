"""Runtime dependencies for graph nodes.

State versus runtime context: *state* is data the workflow produces and that LangGraph
checkpoints (serializable, persisted, resumable). *Runtime context* is read-only machinery
supplied fresh on every invocation: the LLM client, mail/calendar providers, DB sessionmaker,
settings, a clock. It is not serializable and must never be checkpointed.

Wire it in with `StateGraph(EmailState, context_schema=GraphDeps)`, invoke with
`graph.ainvoke(input, context=GraphDeps(...))`, and read it in a node by declaring a
`runtime: Runtime[GraphDeps]` parameter and using `runtime.context`. Because the deps are
re-supplied on resume, a paused workflow can continue in a new process.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.llm.client import LLMClient
from app.providers.interfaces import CalendarProvider, MailProvider
from app.retrieval.contextualize import Contextualizer
from app.retrieval.embeddings import EmbeddingsClient
from app.retrieval.rerank import Reranker


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class GraphDeps:
    llm: LLMClient
    mail: MailProvider
    calendar: CalendarProvider
    sessionmaker: Any  # async_sessionmaker[AsyncSession]
    settings: Settings
    # Injectable clock so tests can fix "now".
    now: Callable[[], datetime] = field(default=_utc_now)
    # Retrieval over past email; None when RAG is disabled.
    embeddings: EmbeddingsClient | None = None
    reranker: Reranker | None = None
    # Index-time chunk context (contextual retrieval); None unless RAG_CONTEXTUALIZE.
    contextualizer: Contextualizer | None = None
