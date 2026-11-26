# Architecture (milestone 2: mock email + retrieval over past email)

```
 browser ──► web (Caddy :8080) ──/api/*──► api (FastAPI)
                │ static SPA                 │  owner auth, CSRF, reviews, decisions, SSE
                                             ▼
                                   Postgres 17+pgv ◄────────── worker (LangGraph)
                         app tables · job queue · checkpoints   │ claims jobs, runs graphs
                                                                ▼
                                       Anthropic API (triage/plan/draft) · mock mail · mock calendar
```

The **API never runs the model or touches providers**. It records decisions, enqueues jobs
and serves the state. The **worker** claims jobs from a Postgres queue, runs the LangGraph
workflow with a Postgres checkpointer, and is the only process that calls the model or a
provider. Everything durable lives in Postgres. The API and worker can restart at any time,
and a pending review survives.

## Services

| Service | Image | Role |
|---|---|---|
| `db` | `pgvector/pgvector:0.8.6-pg17-trixie` | app tables, the job queue, LangGraph checkpoints, the event log, the retrieval index |
| `migrate` | backend | one-shot schema migration plus checkpoint setup, before `api`/`worker` start |
| `api` | backend | HTTP API: login, review queue, decisions, preferences, flags, usage, SSE `/events`, dev tools |
| `worker` | backend | job loop: `start_workflow` / `resume` / `index_message` jobs, leases, retries, crash recovery; local embedding and rerank models (`modelcache` volume) |
| `web` | Caddy + built SPA | serves the React app, proxies `/api/*` (SSE unbuffered) |

## Who owns what

| Concern | Owner | Where |
|---|---|---|
| Deciding what to do (classify, plan, draft) | the model, through structured outputs | `app/graphs/nodes/{triage,plan,draft}.py`, `app/graphs/schemas.py` |
| Whether a draft is acceptable (recipients in thread, evidence cited, meeting fits a free slot) | deterministic code | `app/graphs/nodes/validate.py` |
| Proposals: immutable revisions plus a payload hash | app tables | `app/actions/proposals.py` |
| Approving a specific revision, hash and interrupt | the owner, through the API | `app/actions/decisions.py`, `app/api/decisions.py` |
| Re-checking before acting (approval, TTL, thread changed, slot taken, kill switch) | deterministic code | `app/graphs/nodes/execute.py` (`preflight`) |
| External writes (send, create event), exactly once | the executor and its action ledger | `app/actions/executor.py` |
| Workflow progress and pauses | LangGraph plus the Postgres checkpointer | `app/graphs/parent.py`, `app/graphs/response.py` |
| Running, retrying and recovering work | the job queue and worker | `app/jobs/queue.py`, `app/jobs/worker.py` |
| Spend limits | the budget guard (reserve before calling, settle after) | `app/llm/budget.py`, `app/llm/anthropic_client.py` |
| What the UI shows | projections from app tables (never checkpoints) | `app/api/views.py` |
| Live updates | the append-only `workflow_events` log, streamed as SSE | `app/graphs/events.py`, `app/api/events.py` |
| Secrets | backend settings only; never in graph state, prompts, responses or events | `app/config.py` |
| Remembering past email (RAG): index, search, scope | the retrieval package and the `retrieve_history` node; boundaries enforced in code | `app/retrieval/`, `app/graphs/nodes/retrieve.py`, [rag.md](rag.md) |

## Reliability boundaries

- **LLM output is a proposal, never an action.** A reply is sent only after the owner approves
  that exact revision. Preflight then re-verifies it, and the executor writes an `actions` row
  keyed by an idempotency key before it calls the provider. A retry reconciles against the
  ledger instead of sending again.
- **Ambiguous outcomes are not retried blindly.** A provider timeout after acceptance
  becomes `outcome_unknown`, which the UI shows with "check your Sent folder".
- **Interrupt nodes contain only `interrupt()`.** Side effects run in their own nodes and are
  idempotent by app-table keys, because LangGraph re-runs an interrupted node from its start
  when it resumes.
- **No transaction stays open across a model or provider call.**
- **Kill switches.** `OUTBOUND_EXECUTION_ENABLED=false` (the default) blocks every execution in
  preflight. The `outbound_paused` and `ingestion_paused` flags can be toggled from Settings.
- **Recoverable stops pause, they do not end the run.** Budget or step limit reached, outbound
  sending off, or a provider outage at preflight: the run waits at `pause_interrupt` and the
  workflow page offers **Retry**, which re-runs every pre-send check before anything is sent.
  A send whose outcome is unknown is never retried.
- **Automated mail.** Bounces and auto-replies are skipped at ingestion. Other machine-sent
  mail (no-reply senders, `Auto-Submitted`, `List-Id`) is triaged so the owner can be
  notified, but code turns a `respond` classification into `notify`: it is never answered.

## Data flow of one email

1. Ingestion stores the message. Owner-sent mail marks that thread's pending proposals stale.
   Automated mail is skipped. Otherwise ingestion enqueues `start_workflow`. With RAG on, owner
   and ordinary inbound mail also gets an `index_message` job.
2. The worker runs the parent graph: `load_context` → `load_preferences` → `triage` →
   `record_triage`, then branches into ignore, notify (a review item) or the `response`
   subgraph.
3. `response` starts with `retrieve_history` (RAG; a no-op when off), then loops
   `plan_next_step` → read tools / ask the owner / draft → `validate_proposal`
   → `persist_proposal` → `approval_interrupt` (the run pauses here; the checkpoint is in Postgres).
4. The owner decides in the UI. The API records the approval and enqueues a `resume` job.
5. The worker resumes: `preflight` → `execute_action` (executor and ledger) → `plan_next_step`
   (usually `finish`) → `finalize`.

Later milestones swap the mock providers for Gmail and Google Calendar behind the same
interfaces (`app/providers/interfaces.py`).
