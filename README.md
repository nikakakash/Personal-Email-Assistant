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
| keyword (Postgres FTS) | 0.875 | 0.851 | 0.841 | 12 ms |
| vector (bge-small) | 0.958 | 0.872 | 0.894 | 30 ms |
| hybrid (RRF) | 0.958 | 0.874 | 0.891 | 34 ms |
| **hybrid + cross-encoder rerank** | **0.975** | **0.940** | **0.939** | 494 ms |

What we learned:
- **Fusion and reranking do different jobs.** Fusion rescues a weak embedder (hash vectors go
  from 0.69 to 0.89 recall@5). With a good embedder, fusion barely moves recall. The
  cross-encoder is what improves ranking: MRR goes from 0.87 to 0.94, for about 0.5 s of CPU
  per search.
- **Keyword search still wins on exact identifiers** (booking codes, ticket numbers). That is
  why hybrid stays the default when the reranker's latency is unwanted.
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
