# Personal Email Assistant

An AI email assistant that triages incoming mail, drafts replies and meeting proposals, and
**remembers past conversations**, but never sends anything you have not approved.

Stack: Python 3.12, LangGraph, FastAPI, PostgreSQL 17 + pgvector, React + TypeScript, Claude.
It runs locally with Docker Compose. This milestone uses mock mail and calendar providers;
Gmail and Google Calendar adapters come next, behind the same interfaces.

```
 browser ──► web (Caddy) ──/api──► api (FastAPI)          owner auth · CSRF · reviews · SSE
                                     │ records decisions, enqueues jobs
                                     ▼
                       Postgres + pgvector ◄────── worker (LangGraph)
     app tables · job queue · checkpoints ·          │ triage → retrieve history → plan →
     retrieval index (chunks, HNSW, tsvector)        │ draft → validate → approval interrupt
                                                     ▼  → preflight → execute (exactly once)
                                   Claude · local embedding + rerank models · mock providers
```

## What it does

- **Triage**: decides whether each email can be ignored, should be shown to you, or needs a
  reply, using structured outputs and quoted evidence.
- **Agentic drafting**: a LangGraph loop plans, reads the calendar, asks you clarifying
  questions, and drafts a reply or a meeting proposal.
- **Human in the loop**: every send is a proposal you approve as an exact, hashed revision.
  The run pauses at a LangGraph interrupt that survives restarts. Preflight re-checks the
  thread and the calendar before anything is sent, and an idempotent executor makes sure a
  send happens at most once.
- **Retrieval over past email (RAG)**: drafts recall facts from earlier conversations and cite
  each one (`history:<id>`). The review screen lists the cited emails so you can check them.
- **Safety in code, not just in prompts**: recipients are limited to the thread, citations must
  point at real inputs, untrusted text is fenced, spend has a daily budget, and kill switches
  stop all sending.

## RAG design and results

Full design: [docs/rag.md](docs/rag.md).

- **Indexing**: incremental jobs. Quote- and signature-aware chunking, then local `bge-small`
  embeddings (or Voyage), stored in pgvector with one partial HNSW index per embedding model.
- **Search**: Postgres full text and vector search, fused with reciprocal rank fusion, then
  optionally reranked by a local cross-encoder.
- **Leak prevention**: a past email is retrieved only if *every* participant of the current
  thread was a party to it. Without this rule, an outsider could CC your contact into a new
  thread and fish out that contact's private history. A regression test covers that attack.
- **Contextual retrieval** (optional): Claude writes one situating sentence per chunk. It is
  indexed, but never shown to the drafter.

**Retrieval eval.** 60 golden queries over a hand-written mailbox of 151 emails. It is built to
be hard: superseded facts, two people with the same first name, near-duplicate threads,
identifiers, paraphrases and distractors. The table uses mailbox scope, so every
correspondent's mail competes. Local models on CPU, `evals/run_retrieval_eval.py`:

| mode | recall@5 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|
| keyword (Postgres FTS, BM25 IDF) | 0.875 | 0.846 | 0.852 | 16 ms |
| vector (bge-small) | 0.958 | 0.872 | 0.894 | 42 ms |
| hybrid (RRF) | 0.942 | 0.877 | 0.897 | 46 ms |
| **hybrid + cross-encoder rerank** | **0.975** | **0.940** | **0.939** | 536 ms |

**Under realistic noise.** The same 60 queries, with 3,000 real Enron emails added to the
mailbox (3,151 messages, 3,498 chunks; `--enron … --enron-limit 3000`):

| mode | recall@5 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|
| keyword, plain `ts_rank_cd` (before) | 0.408 | 0.203 | 0.272 | 29 ms |
| keyword, BM25 IDF (now) | 0.775 | 0.727 | 0.725 | 21 ms |
| vector (bge-small) | 0.808 | 0.796 | 0.800 | 30 ms |
| hybrid (RRF) | 0.858 | 0.778 | 0.794 | 48 ms |
| **hybrid + cross-encoder rerank** | **0.933** | **0.926** | **0.913** | 3.8 s |

What we learned:
- **The noise test found a real bug.** Postgres's `ts_rank_cd` has no IDF. Long Enron mail
  that repeated a question's common words ("cost", "say") outranked the one email holding
  its rare, decisive word ("hosting"). Keyword recall@5 collapsed to 0.41 and dragged hybrid
  below vector alone. Weighting each word by its BM25 IDF fixed it: keyword 0.41 → 0.78,
  hybrid 0.72 → 0.86. The small clean set had hidden this entirely.
- **Under noise, the reranker matters most.** It turns 0.86 into 0.93 recall@5 and lifts MRR
  from 0.78 to 0.93. Its CPU cost grows with chunk length (about 0.5 s on short synthetic
  mail, 3.8 s on long Enron chunks), so it is opt-in (`RETRIEVAL_MODE=hybrid_rerank`), and
  a GPU or a hosted reranker would be the production choice.
- **Fusion rescues a weak embedder** (hash vectors go from 0.69 to 0.89 recall@5 on the clean
  set). With a good embedder on clean data, it barely moves recall. With noise, it adds about
  0.05.
- **Vector numbers vary a little between runs.** Three identical Enron runs gave vector
  recall@5 of 0.73, 0.81 and 0.78. HNSW was not the cause (exact search matched it); the
  local CPU embedder is not bit-for-bit deterministic, and near-ties among thousands of
  similar emails flip. On 60 queries, 0.04 is two or three queries. Keyword results are
  identical run to run.
- **The remaining misses are superseded facts**, where both the old and the new message are
  needed in the top 5.
- **The participant scope (the default) is nearly saturated** at recall@5 of about 0.99,
  because scoping already removes most candidates.

**CI quality gate.** The retrieval eval runs in CI with the local models, and fails if any
metric drops below [evals/retrieval/baseline.json](evals/retrieval/baseline.json).

## Run it

```bash
cp .env.example .env      # set OWNER_PASSPHRASE, SESSION_SECRET, ANTHROPIC_API_KEY
docker compose up -d --build
open http://localhost:8080
```

Details, development without containers, tests, E2E and evals are in
[docs/setup.md](docs/setup.md). The architecture and its reliability boundaries are in
[docs/architecture.md](docs/architecture.md). A guided tour of the LangGraph design is in
[docs/learn-langgraph.md](docs/learn-langgraph.md).

## Tests

- Backend: about 560 tests (pytest, real Postgres), ruff and mypy strict.
- Frontend: vitest unit tests and a strict TypeScript build.
- End to end: Playwright journeys against the full Compose stack, including a RAG journey.
- Evals: retrieval (free, in CI), triage and end-to-end RAG drafts (paid, opt-in).
