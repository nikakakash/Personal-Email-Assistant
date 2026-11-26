"""Retrieval evaluation: index the synthetic mailbox, run the golden queries, report metrics.

Usage (from backend/, with the database up):

    uv run python ../evals/run_retrieval_eval.py                        # fastembed, all modes
    uv run python ../evals/run_retrieval_eval.py --embedders hash,fastembed --scope mailbox
    uv run python ../evals/run_retrieval_eval.py --embedders voyage --yes  # costs a few cents
    uv run python ../evals/run_retrieval_eval.py --enron ~/data/enron/maildir --enron-limit 10000
    uv run python ../evals/run_retrieval_eval.py --check-baseline ../evals/retrieval/baseline.json

It uses its own database (`email_assistant_eval`, created and migrated on first run; override
with --db-url) and wipes its mail tables on every run. Each query is a new incoming email; it
goes through the same `search()` the app uses, with the query sender as the participant scope
(`--scope participants`, the app default) or the whole mailbox (`--scope mailbox`, harder:
every correspondent's mail competes).

Metrics, averaged over queries: recall@5 and recall@10 (share of the relevant messages found),
MRR (1 / rank of the first relevant hit), nDCG@10 (binary relevance) and search latency p50/p95.
Results go to `evals/results/retrieval-<date>.json`; `--check-baseline` exits 1 when a metric
falls below the stored minimum (used by CI).
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))

from evals.retrieval.corpus import MESSAGES, QUERIES  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.retrieval.embeddings import EmbeddingsClient  # noqa: E402
from app.retrieval.indexer import reindex_all  # noqa: E402
from app.retrieval.rerank import Reranker  # noqa: E402
from app.retrieval.search import SearchMode, search  # noqa: E402

ACCOUNT = "eval"
EVAL_DB = "email_assistant_eval"
ALL_MODES: tuple[SearchMode, ...] = ("keyword", "vector", "hybrid", "hybrid_rerank")
RESULTS_DIR = REPO / "evals" / "results"


@dataclass
class QueryResult:
    query_id: str
    category: str
    relevant: list[str]
    retrieved: list[str]
    latency_ms: float


@dataclass
class Report:
    embedder: str
    mode: str
    scope: str
    queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    recall_at_5_by_category: dict[str, float] = field(default_factory=dict)
    misses_at_5: list[str] = field(default_factory=list)


# --- metrics (pure) -----------------------------------------------------------------------------


def recall_at(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return len(set(relevant) & set(retrieved[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], 1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize(embedder: str, mode: str, scope: str, results: Sequence[QueryResult]) -> Report:
    by_category: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(recall_at(r.relevant, r.retrieved, 5))
    latencies = [r.latency_ms for r in results]
    return Report(
        embedder=embedder,
        mode=mode,
        scope=scope,
        queries=len(results),
        recall_at_5=round(
            statistics.fmean(recall_at(r.relevant, r.retrieved, 5) for r in results), 4
        ),
        recall_at_10=round(
            statistics.fmean(recall_at(r.relevant, r.retrieved, 10) for r in results), 4
        ),
        mrr=round(statistics.fmean(reciprocal_rank(r.relevant, r.retrieved) for r in results), 4),
        ndcg_at_10=round(
            statistics.fmean(ndcg_at(r.relevant, r.retrieved, 10) for r in results), 4
        ),
        latency_p50_ms=round(percentile(latencies, 0.5), 1),
        latency_p95_ms=round(percentile(latencies, 0.95), 1),
        recall_at_5_by_category={
            c: round(statistics.fmean(v), 4) for c, v in sorted(by_category.items())
        },
        misses_at_5=[r.query_id for r in results if recall_at(r.relevant, r.retrieved, 5) < 1.0],
    )


def markdown_table(reports: Sequence[Report]) -> str:
    lines = [
        "| embedder | mode | scope | recall@5 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.embedder} | {r.mode} | {r.scope} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} "
            f"| {r.mrr:.3f} | {r.ndcg_at_10:.3f} "
            f"| {r.latency_p50_ms:.1f} | {r.latency_p95_ms:.1f} |"
        )
    return "\n".join(lines)


def check_baseline(reports: Sequence[Report], baseline: list[dict[str, Any]]) -> list[str]:
    """Each baseline entry: {embedder, mode, scope, metric, min}. Returns the failures."""
    failures = []
    index = {(r.embedder, r.mode, r.scope): r for r in reports}
    for entry in baseline:
        report = index.get((entry["embedder"], entry["mode"], entry["scope"]))
        if report is None:
            failures.append(f"missing run for {entry}")
            continue
        value = float(getattr(report, entry["metric"]))
        if value < float(entry["min"]):
            failures.append(
                f"{entry['embedder']}/{entry['mode']}/{entry['scope']} {entry['metric']} "
                f"{value:.3f} < {entry['min']}"
            )
    return failures


# --- database -----------------------------------------------------------------------------------


def default_db_url() -> str:
    user = os.environ.get("POSTGRES_USER", "app")
    password = os.environ.get("POSTGRES_PASSWORD", "app")
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5433")
    creds = f"{quote(user, safe='')}:{quote(password, safe='')}"
    return f"postgresql+psycopg://{creds}@{host}:{port}/{EVAL_DB}"


def prepare_database(url: str) -> None:
    """Create the eval database if needed and migrate it to head."""
    import psycopg
    from alembic import command
    from alembic.config import Config
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    admin = parsed.set(
        drivername="postgresql", database=os.environ.get("POSTGRES_DB", "email_assistant")
    )
    with psycopg.connect(admin.render_as_string(hide_password=False), autocommit=True) as conn:
        if not conn.execute(
            "select 1 from pg_database where datname = %s", (parsed.database,)
        ).fetchone():
            conn.execute(f'create database "{parsed.database}"')
    cfg = Config(str(REPO / "backend" / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "backend" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(cfg, "head")


async def load_corpus(maker: async_sessionmaker[AsyncSession]) -> int:
    async with maker() as session, session.begin():
        await session.execute(
            text("TRUNCATE email_threads, email_messages, message_chunks, chunk_embeddings CASCADE")
        )
        threads: dict[str, uuid.UUID] = {}
        for mid, thread, sender, to, day, subject, body in MESSAGES:
            if thread not in threads:
                threads[thread] = await session.scalar(
                    text(
                        "insert into email_threads (account_id, provider_thread_id) "
                        "values (:a, :t) returning id"
                    ),
                    {"a": ACCOUNT, "t": thread},
                )
            await session.execute(
                text(
                    "insert into email_messages (thread_id, account_id, provider_message_id, "
                    "sender, "
                    "to_addrs, subject, body_text, received_at, content_hash) values "
                    "(:th, :a, :m, :s, :to, :subj, :body, :at, :h)"
                ),
                {
                    "th": threads[thread],
                    "a": ACCOUNT,
                    "m": mid,
                    "s": sender,
                    "to": to,
                    "subj": subject,
                    "body": body,
                    "at": datetime.fromisoformat(day).replace(tzinfo=UTC),
                    "h": mid,
                },
            )
    return len(MESSAGES)


# --- runs -------------------------------------------------------------------------------------


def make_embedder(
    name: str, maker: async_sessionmaker[AsyncSession], budget: Decimal
) -> EmbeddingsClient:
    from app.llm.budget import BudgetGuard
    from app.retrieval.embeddings import (
        FastEmbedClient,
        HashEmbeddingsClient,
        VoyageEmbeddingsClient,
    )

    if name == "hash":
        return HashEmbeddingsClient()
    if name == "fastembed":
        return FastEmbedClient("bge-small-en-v1.5")
    if name == "voyage":
        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise SystemExit("--embedders voyage needs VOYAGE_API_KEY")
        return VoyageEmbeddingsClient(api_key=key, budget=BudgetGuard(maker, budget))
    raise SystemExit(f"unknown embedder {name!r} (hash, fastembed, voyage)")


async def run_queries(
    maker: async_sessionmaker[AsyncSession],
    embedder: EmbeddingsClient,
    mode: SearchMode,
    scope: str,
    reranker: Reranker | None,
) -> list[QueryResult]:
    results = []
    for qid, sender, subject, body, relevant, category in QUERIES:
        started = time.perf_counter()
        async with maker() as session:
            hits = await search(
                session,
                embedder,
                account_id=ACCOUNT,
                query=f"{subject}\n{body}",
                k=10,
                mode=mode,
                exclude_provider_thread_id=f"query-{qid}",
                participants=[sender] if scope == "participants" else None,
                reranker=reranker,
            )
        latency = (time.perf_counter() - started) * 1000
        results.append(
            QueryResult(
                qid, category, list(relevant), [h.provider_message_id for h in hits], latency
            )
        )
    return results


def estimate_voyage_tokens() -> int:
    chars = sum(len(m[5]) + len(m[6]) + 120 for m in MESSAGES) + sum(
        len(q[2]) + len(q[3]) for q in QUERIES
    )
    return chars // 3


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--embedders", default="fastembed", help="comma list: hash,fastembed,voyage"
    )
    parser.add_argument(
        "--modes", default="keyword,vector,hybrid", help=f"comma list of {ALL_MODES}"
    )
    parser.add_argument("--scope", choices=["participants", "mailbox", "both"], default="both")
    parser.add_argument(
        "--db-url", default=None, help=f"default: the compose Postgres, db {EVAL_DB}"
    )
    parser.add_argument(
        "--enron", type=Path, help="local Enron maildir to add as noise (mailbox scope)"
    )
    parser.add_argument("--enron-limit", type=int, default=10_000)
    parser.add_argument(
        "--rerank", action="store_true", help="also run hybrid_rerank (local cross-encoder)"
    )
    parser.add_argument(
        "--contextualize", action="store_true", help="add Claude-written chunk context (paid)"
    )
    parser.add_argument("--daily-budget", type=Decimal, default=Decimal("1.00"))
    parser.add_argument(
        "--yes", action="store_true", help="allow paid steps (Voyage, contextualize)"
    )
    parser.add_argument("--check-baseline", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.rerank and "hybrid_rerank" not in modes:
        modes.append("hybrid_rerank")
    if bad := [m for m in modes if m not in ALL_MODES]:
        raise SystemExit(f"unknown modes {bad}")
    paid = "voyage" in embedders or args.contextualize
    if paid and not args.yes:
        print(
            f"Paid steps requested. Voyage tokens ~{estimate_voyage_tokens():,} "
            "(about $0.001 at $0.02/M);"
        )
        print("contextualize: one small Claude call per chunk. Re-run with --yes to proceed.")
        return 2

    url = args.db_url or default_db_url()
    prepare_database(url)
    engine = create_async_engine(url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    reports: list[Report] = []
    try:
        count = await load_corpus(maker)
        if args.enron:
            from evals.retrieval.load_enron import load_enron

            count += await load_enron(maker, args.enron, ACCOUNT, limit=args.enron_limit)
        print(f"corpus: {count} messages, {len(QUERIES)} queries")
        reranker = None
        if "hybrid_rerank" in modes:
            from app.retrieval.rerank import FastEmbedReranker

            reranker = FastEmbedReranker()
        scopes = ["participants", "mailbox"] if args.scope == "both" else [args.scope]
        for name in embedders:
            embedder = make_embedder(name, maker, args.daily_budget)
            started = time.perf_counter()
            if args.contextualize:
                from evals.retrieval.contextualize import contextualize_all

                await contextualize_all(maker, ACCOUNT, args.daily_budget)
            stats = await reindex_all(maker, embedder, account_id=ACCOUNT)
            print(f"[{name}] indexed {stats.chunks} chunks in {time.perf_counter() - started:.1f}s")
            for scope in scopes:
                for mode in modes:
                    if mode == "keyword" and name != embedders[0]:
                        continue  # keyword ranking does not depend on the embedder
                    results = await run_queries(maker, embedder, mode, scope, reranker)  # type: ignore[arg-type]
                    label = "-" if mode == "keyword" else name
                    reports.append(summarize(label, mode, scope, results))
    finally:
        await engine.dispose()

    print()
    print(markdown_table(reports))
    out = args.out or RESULTS_DIR / f"retrieval-{datetime.now(UTC).date().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"corpus_messages": count, "reports": [asdict(r) for r in reports]}, indent=2)
        + "\n"
    )
    print(f"\nwrote {out}")
    if args.check_baseline:
        failures = check_baseline(reports, json.loads(args.check_baseline.read_text()))
        for failure in failures:
            print(f"BASELINE FAILED: {failure}")
        if failures:
            return 1
        print("baseline: ok")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # psycopg async
    sys.exit(asyncio.run(main()))
