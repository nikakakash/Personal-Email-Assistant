"""`retrieve_history`: the first node of the response subgraph (RAG over past email).

It searches the retrieval index for past messages related to the current email and stores them
in `retrieved_context`; `user_content` then shows them to the planner and drafter as untrusted
data, citable as `history:<provider_message_id>`, and validation accepts exactly those refs.

Boundaries, enforced in code rather than by the prompt:
- Scope: always the run's account, never the current thread, and by default
  (`RETRIEVAL_SCOPE=participants`) only past messages that every external participant of the
  current thread was party to. Replies can only go to those participants, so no retrieved
  message can reach someone who was not on it; and an outsider who CCs a contact into a new
  thread cannot pull that contact's private history into the reply.
- Authority: retrieved text is context only. Recipients are still limited to the current
  thread's participants by `validate_payload`, whatever a past email asks for.
- Failure: a search error or timeout is recorded as a `history` finding and the run goes on
  without history; it never fails the workflow.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from langgraph.runtime import Runtime

from app.graphs.deps import GraphDeps
from app.graphs.events import emit_now
from app.graphs.nodes.reads import TOOL_TIMEOUT_SECONDS
from app.graphs.nodes.shared import is_from_owner, owner_identities, thread_of, thread_participants
from app.graphs.state import EmailState
from app.llm.client import BudgetExceeded
from app.mail.content import model_text
from app.retrieval.chunking import clean_body
from app.retrieval.embeddings import EmbeddingError
from app.retrieval.search import Hit, SearchMode, search

log = logging.getLogger(__name__)

QUERY_MAX_CHARS = 1000
HIT_MAX_CHARS = 1200
HISTORY_FINDING_KEY = "history:search"


def history_query(state: EmailState, identities: list[str]) -> str:
    """Subject plus the latest message not sent by the owner (cleaned of quotes, bounded)."""
    thread = thread_of(state)
    inbound = [m for m in thread.messages if not is_from_owner(m, identities)]
    latest = (inbound or thread.messages)[-1] if thread.messages else None
    if latest is None:
        return ""
    body = clean_body(model_text(latest.body_text, latest.body_html))
    return f"{latest.subject}\n{body}"[:QUERY_MAX_CHARS].strip()


def _finding(status: str, count: int = 0) -> dict[str, Any]:
    return {"key": HISTORY_FINDING_KEY, "kind": "history", "status": status, "count": count}


async def retrieve_history(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    deps, settings = runtime.context, runtime.context.settings
    if not settings.rag_enabled or deps.embeddings is None:
        return {"retrieved_context": []}

    identities = owner_identities(settings)
    query = history_query(state, identities)
    participants = (
        thread_participants(thread_of(state), identities)
        if settings.retrieval_scope == "participants"
        else None
    )
    mode: SearchMode = settings.retrieval_mode
    if mode == "hybrid_rerank" and deps.reranker is None:
        mode = "hybrid"
    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            async with deps.sessionmaker() as session:
                hits = await search(
                    session,
                    deps.embeddings,
                    account_id=state["account_id"],
                    query=query,
                    k=settings.retrieval_top_k,
                    mode=mode,
                    exclude_provider_thread_id=state["provider_thread_id"],
                    participants=participants,
                    reranker=deps.reranker,
                )
    except (TimeoutError, EmbeddingError, BudgetExceeded, OSError) as exc:
        log.warning("history search unavailable (%s); continuing without it", type(exc).__name__)
        return {"retrieved_context": [], "read_findings": [_finding("unavailable")]}
    except Exception:
        log.exception("history search failed; continuing without it")
        return {"retrieved_context": [], "read_findings": [_finding("unavailable")]}

    context = context_from_hits(hits)
    await emit_now(
        deps,
        state["workflow_id"],
        [("tool_result", {"tool": "search_history", "key": HISTORY_FINDING_KEY})],
    )
    return {"retrieved_context": context, "read_findings": [_finding("ok", len(context))]}


def context_from_hits(hits: Sequence[Hit]) -> list[dict[str, Any]]:
    """The JSON-serializable `retrieved_context` entries for search hits."""
    return [
        {
            "ref": f"history:{h.provider_message_id}",
            "provider_message_id": h.provider_message_id,
            "provider_thread_id": h.provider_thread_id,
            "sender": h.sender,
            "subject": h.subject,
            "date": h.received_at.date().isoformat(),
            "text": h.text[:HIT_MAX_CHARS],
            "score": round(h.score, 6),
        }
        for h in hits
    ]
