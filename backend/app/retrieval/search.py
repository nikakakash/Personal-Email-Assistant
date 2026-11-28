"""Search the retrieval index: keyword, vector, hybrid (reciprocal rank fusion) and rerank.

- keyword: Postgres full-text search over the generated `tsv` column. Any of the question's
  words can match, and a chunk scores the sum of the BM25 IDF of the distinct query words it
  contains, so a rare word (a name, an identifier, "hosting") outweighs common ones repeated
  many times. `ts_rank_cd` (length-normalized) only breaks ties. This is BM25 without term
  frequency or length saturation, and it fills the same role: exact names, numbers and
  identifiers that embeddings blur. Plain `ts_rank_cd` has no IDF and collapsed under
  realistic noise (Enron eval: recall@5 0.41).
- vector: cosine distance over the model's partial HNSW index (`embedding::vector(dim)`),
  with pgvector's iterative scan, so the account and thread filters cannot starve the
  result list.
- hybrid: reciprocal rank fusion (RRF, k=60) of both candidate lists.
- hybrid_rerank: hybrid candidates re-scored by a cross-encoder.

Every query is scoped to one account (mandatory) and can exclude the current thread and be
limited to messages that all given participants were party to. Results
are one hit per message: its best-ranked chunk.
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.embeddings import EmbeddingsClient
from app.retrieval.rerank import Reranker

SearchMode = Literal["keyword", "vector", "hybrid", "hybrid_rerank"]
RRF_K = 60
MIN_CANDIDATES = 30
RERANK_POOL = 20
MAX_QUERY_TERMS = 128  # bounds the per-term document-frequency lookups of one keyword query
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9._-]{1,100}$")

# plainto_tsquery normalizes the text safely (no operator injection); its AND-ed lexemes are
# turned into an OR query. The ::tsquery cast re-reads quoted lexemes without re-stemming.
_TSQUERY = "replace(plainto_tsquery('english', :query)::text, '&', '|')::tsquery"


@dataclass(frozen=True)
class Hit:
    chunk_id: uuid.UUID
    message_id: uuid.UUID
    provider_message_id: str
    provider_thread_id: str
    sender: str
    subject: str
    received_at: datetime
    text: str
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None


def rrf_fuse(rankings: Sequence[Sequence[Any]], k: int = RRF_K) -> list[tuple[Any, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))


# Every requested participant was a party (sender, to or cc) of the past message. "All", not
# "any": otherwise an outsider could CC someone into a new thread and pull that person's private
# history into a reply the outsider receives. Addresses are compared bare and lowercased
# ("Name <a@x>" -> "a@x").
_PARTICIPANT_MATCH = (
    "ARRAY(SELECT coalesce(lower(substring(p.addr from '<([^>]+)>')), lower(btrim(p.addr))) "
    "FROM unnest(m.to_addrs || m.cc_addrs || ARRAY[m.sender]) AS p(addr)) "
    "@> CAST(:participants AS text[])"
)


def _filters(
    params: dict[str, Any],
    exclude_provider_thread_id: str | None,
    sender: str | None,
    since: datetime | None,
    participants: Sequence[str] | None = None,
) -> str:
    clauses = ["c.account_id = :account_id"]
    if participants is not None:
        clauses.append(_PARTICIPANT_MATCH)
        params["participants"] = sorted({p.strip().lower() for p in participants})
    if exclude_provider_thread_id is not None:
        clauses.append("t.provider_thread_id <> :exclude_thread")
        params["exclude_thread"] = exclude_provider_thread_id
    if sender is not None:
        clauses.append("lower(c.sender) = lower(:sender)")
        params["sender"] = sender
    if since is not None:
        clauses.append("c.received_at >= :since")
        params["since"] = since
    return " AND ".join(clauses)


def vector_candidates_sql(
    *,
    model: str,
    dim: int,
    vector: Sequence[float],
    account_id: str,
    limit: int,
    exclude_provider_thread_id: str | None = None,
    sender: str | None = None,
    since: datetime | None = None,
    participants: Sequence[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """The vector candidate query. `model` and `dim` are inlined (validated) so the planner
    can match the partial HNSW index `WHERE model = '<model>'` on `embedding::vector(dim)`."""
    if not _SAFE_MODEL.match(model) or not (isinstance(dim, int) and 0 < dim <= 16000):
        raise ValueError("invalid embedding model or dimension")
    params: dict[str, Any] = {
        "account_id": account_id,
        "qvec": "[" + ",".join(repr(float(x)) for x in vector) + "]",
        "limit": limit,
    }
    where = _filters(params, exclude_provider_thread_id, sender, since, participants)
    sql = (
        f"SELECT c.id AS chunk_id, e.embedding::vector({dim}) <=> CAST(:qvec AS vector({dim})) "
        "AS distance "
        "FROM chunk_embeddings e "
        "JOIN message_chunks c ON c.id = e.chunk_id "
        "JOIN email_threads t ON t.id = c.thread_id "
        "JOIN email_messages m ON m.id = c.message_id "
        f"WHERE e.model = '{model}' AND {where} "
        f"ORDER BY e.embedding::vector({dim}) <=> CAST(:qvec AS vector({dim})) "
        "LIMIT :limit"
    )
    return sql, params


async def _keyword(
    session: AsyncSession, query: str, limit: int, filters: dict[str, Any]
) -> list[tuple[uuid.UUID, float]]:
    params: dict[str, Any] = {"query": query, "limit": limit, "account_id": filters["account_id"]}
    where = _filters(
        params, filters["exclude"], filters["sender"], filters["since"], filters["participants"]
    )
    params["max_terms"] = MAX_QUERY_TERMS
    sql = (
        # The query's distinct lexemes (same parser as the index), each with its BM25 IDF
        # over this account's chunks: ln(1 + (N - df + 0.5) / (df + 0.5)).
        "WITH terms AS ("
        "  SELECT DISTINCT lex FROM unnest(tsvector_to_array(to_tsvector('english', :query)))"
        "  AS lex LIMIT :max_terms"
        "), n AS ("
        "  SELECT count(*)::float AS total FROM message_chunks WHERE account_id = :account_id"
        "), weighted AS ("
        "  SELECT quote_literal(terms.lex)::tsquery AS tq,"
        "         ln(1 + (n.total - d.df + 0.5) / (d.df + 0.5)) AS idf"
        "  FROM terms CROSS JOIN n CROSS JOIN LATERAL ("
        "    SELECT count(*)::float AS df FROM message_chunks d"
        "    WHERE d.account_id = :account_id AND d.tsv @@ quote_literal(terms.lex)::tsquery"
        "  ) d"
        ") "
        f"SELECT c.id, sum(w.idf) AS rank, ts_rank_cd(c.tsv, {_TSQUERY}, 1) AS tie "
        "FROM message_chunks c JOIN email_threads t ON t.id = c.thread_id "
        "JOIN email_messages m ON m.id = c.message_id "
        "JOIN weighted w ON c.tsv @@ w.tq "
        f"WHERE {where} "
        "GROUP BY c.id "
        "ORDER BY rank DESC, tie DESC, c.received_at DESC, c.id LIMIT :limit"
    )
    rows = (await session.execute(text(sql), params)).all()
    return [(row[0], float(row[1])) for row in rows]


async def _vector(
    session: AsyncSession,
    embedder: EmbeddingsClient,
    query: str,
    limit: int,
    filters: dict[str, Any],
) -> list[tuple[uuid.UUID, float]]:
    vector = (await embedder.embed([query], kind="query")).vectors[0]
    sql, params = vector_candidates_sql(
        model=embedder.model,
        dim=embedder.dim,
        vector=vector,
        account_id=filters["account_id"],
        limit=limit,
        exclude_provider_thread_id=filters["exclude"],
        sender=filters["sender"],
        since=filters["since"],
        participants=filters["participants"],
    )
    # Keep scanning the HNSW graph until `limit` rows pass the filters (pgvector >= 0.8).
    await session.execute(text("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"))
    rows = (await session.execute(text(sql), params)).all()
    # relaxed_order may return slightly out of order: sort by the exact distance.
    return sorted(((row[0], 1.0 - float(row[1])) for row in rows), key=lambda r: -r[1])


async def _details(session: AsyncSession, chunk_ids: list[uuid.UUID]) -> dict[uuid.UUID, Any]:
    if not chunk_ids:
        return {}
    rows = await session.execute(
        text(
            "SELECT c.id, c.message_id, m.provider_message_id, t.provider_thread_id, c.sender, "
            "m.subject, c.received_at, c.text "
            "FROM message_chunks c "
            "JOIN email_messages m ON m.id = c.message_id "
            "JOIN email_threads t ON t.id = c.thread_id "
            "WHERE c.id = ANY(:ids)"
        ),
        {"ids": chunk_ids},
    )
    return {row[0]: row for row in rows}


async def search(
    session: AsyncSession,
    embedder: EmbeddingsClient,
    *,
    account_id: str,
    query: str,
    k: int = 5,
    mode: SearchMode = "hybrid",
    exclude_provider_thread_id: str | None = None,
    sender: str | None = None,
    since: datetime | None = None,
    participants: Sequence[str] | None = None,
    reranker: Reranker | None = None,
) -> list[Hit]:
    """`participants` (bare addresses) limits results to messages that all of them were party
    to, so nobody's mail can surface in a reply that reaches someone who was not on it."""
    if not account_id:
        raise ValueError("account_id is required: retrieval is always scoped to one account")
    if mode == "hybrid_rerank" and reranker is None:
        raise ValueError("mode hybrid_rerank needs a reranker")
    if not query.strip() or k <= 0 or (participants is not None and not participants):
        return []
    limit = max(MIN_CANDIDATES, 4 * k)
    filters = {
        "account_id": account_id,
        "exclude": exclude_provider_thread_id,
        "sender": sender,
        "since": since,
        "participants": participants,
    }

    keyword = await _keyword(session, query, limit, filters) if mode != "vector" else []
    vector = await _vector(session, embedder, query, limit, filters) if mode != "keyword" else []
    keyword_rank = {cid: i for i, (cid, _) in enumerate(keyword, 1)}
    vector_rank = {cid: i for i, (cid, _) in enumerate(vector, 1)}

    if mode == "keyword":
        ranked = keyword
    elif mode == "vector":
        ranked = vector
    else:
        ranked = rrf_fuse([[c for c, _ in keyword], [c for c, _ in vector]])

    details = await _details(session, [cid for cid, _ in ranked])
    hits: list[Hit] = []
    seen: set[uuid.UUID] = set()
    pool = RERANK_POOL if mode == "hybrid_rerank" else k
    for chunk_id, score in ranked:
        row = details.get(chunk_id)
        if row is None or row[1] in seen:
            continue
        seen.add(row[1])
        hits.append(
            Hit(
                chunk_id=chunk_id,
                message_id=row[1],
                provider_message_id=row[2],
                provider_thread_id=row[3],
                sender=row[4],
                subject=row[5],
                received_at=row[6],
                text=row[7],
                score=score,
                keyword_rank=keyword_rank.get(chunk_id),
                vector_rank=vector_rank.get(chunk_id),
            )
        )
        if len(hits) >= pool:
            break

    if mode == "hybrid_rerank" and reranker is not None and hits:
        scores = await reranker.score(query, [h.text for h in hits])
        rescored = [replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)]
        hits = sorted(rescored, key=lambda h: -h.score)
    return hits[:k]
