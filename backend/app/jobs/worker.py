"""The durable worker: claims jobs from Postgres and runs the LangGraph workflow for each.

How a workflow survives a restart (the concepts, in order):

- **Checkpointer.** The parent graph is compiled with `AsyncPostgresSaver`, LangGraph's
  Postgres checkpointer. After every step (one "super-step" of nodes) LangGraph writes a
  checkpoint: the full state, which nodes run next, and any pending interrupts. The checkpoint
  tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) are created by
  `AsyncPostgresSaver.setup()`, which `python -m app migrate` runs.
- **`thread_id`.** Checkpoints are grouped by `config={"configurable": {"thread_id": ...}}`.
  One email message is one workflow run is one LangGraph thread, with id
  `f"{account_id}:{provider_message_id}:{GRAPH_VERSION}"` (`WorkflowRun.graph_thread_id`).
  Invoking the graph again with the same thread id continues that run instead of starting a
  new one. A subgraph (the `response` node) stores its checkpoints under the same thread id
  in its own namespace.
- **Durability.** Every invocation passes `durability="sync"`: each checkpoint is written
  before the next step starts. A crash therefore loses at most the step that was running,
  never a completed one, and completed steps (with their side effects) are not replayed.
- **Pausing.** An `interrupt()` call stops the run and is stored in the checkpoint. Nothing
  waits in memory: the worker finishes the job, `projection.py` copies the interrupt id onto
  the review item, and the process may stop for days.
- **Resuming.** The owner's decision creates a `resume_workflow` job (in the same
  transaction as the decision). The worker reads the checkpoint with `aget_state`, checks
  that the interrupt the owner answered is still pending (spec §9), and calls
  `ainvoke(Command(resume=value), config, ...)`. LangGraph loads the checkpoint, re-runs the
  interrupted node from its top, and this time `interrupt()` returns `value`. The graph object
  and the saver can be brand new (a new process): everything needed is in Postgres, and the
  runtime dependencies (`GraphDeps`) are passed fresh on each call, never checkpointed.
- **Continuing after a crash.** If a worker dies mid-run, its job lease expires and another
  worker claims the job again. The checkpoint then has no pending interrupt but a non-empty
  `snap.next` (the step that was running). `ainvoke(None, config, ...)` means "continue from
  the last checkpoint": only the unfinished step runs again. A node that may already have
  done its external write (`execute_action`) consults the action ledger first, so the write
  is reconciled, never repeated (see `app/graphs/nodes/execute.py`).

Job outcomes: success → projection, then `done`. `ActionInFlight` → the same job is queued
again `IN_FLIGHT_RETRY` later (not a failure). `LLMTransientError` → the client already made
its bounded attempts, so the run is `failed` (`llm_transient`) and the job fails without a
retry. Any other exception → the run is `failed` (`internal_error`); the traceback goes to the
server log, never into events or the job row.
"""

import asyncio
import contextlib
import logging
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import AppFlags, EmailMessage, EmailThread, Job, WorkflowRun
from app.db.session import make_sessionmaker
from app.graphs.deps import GraphDeps
from app.graphs.events import emit
from app.graphs.nodes.execute import ActionInFlight
from app.graphs.nodes.finalize import FINAL_STATUSES
from app.graphs.parent import GRAPH_VERSION, build_parent_graph, graph_thread_id
from app.graphs.state import EmailState
from app.jobs import queue
from app.jobs.projection import pending_interrupts, project_interrupts
from app.llm.client import BudgetExceeded, LLMTransientError
from app.llm.factory import make_llm_client
from app.prompts import PROMPT_VERSIONS
from app.providers.mock_calendar import MockCalendarProvider
from app.providers.mock_mail import MockFaults, MockMailProvider

log = logging.getLogger(__name__)

Graph = CompiledStateGraph[EmailState, GraphDeps, EmailState, EmailState]
DepsFactory = Callable[[Settings, async_sessionmaker[AsyncSession], MockFaults], GraphDeps]

IN_FLIGHT_RETRY = timedelta(seconds=30)


def make_deps(
    settings: Settings, sessionmaker: async_sessionmaker[AsyncSession], faults: MockFaults
) -> GraphDeps:
    """Runtime dependencies for this milestone: the configured LLM and the mock providers."""
    return GraphDeps(
        llm=make_llm_client(settings, sessionmaker),
        mail=MockMailProvider(sessionmaker, faults),
        calendar=MockCalendarProvider(sessionmaker, None, "UTC", faults),
        sessionmaker=sessionmaker,
        settings=settings,
    )


async def prepare_storage(settings: Settings) -> None:
    """Post-migration setup, safe to repeat: checkpoint tables and the `app_flags` row.

    `AsyncPostgresSaver.setup()` creates or migrates LangGraph's checkpoint tables. The
    `app_flags` singleton (id 1, both pauses off) must exist so the executor's `FOR SHARE`
    read always has a row to lock; an existing row keeps its values.
    """
    async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_database_url) as saver:
        await saver.setup()
    sessionmaker = make_sessionmaker(settings)
    try:
        async with sessionmaker() as session, session.begin():
            await session.execute(
                insert(AppFlags)
                .values(id=1, ingestion_paused=False, outbound_paused=False)
                .on_conflict_do_nothing(index_elements=[AppFlags.id])
            )
    finally:
        await sessionmaker.kw["bind"].dispose()


@dataclass(frozen=True)
class _Target:
    """The run a job acts on."""

    workflow_id: uuid.UUID
    thread_id: str
    initial_state: dict[str, Any] | None = None  # start_workflow only

    @property
    def config(self) -> RunnableConfig:
        return {"configurable": {"thread_id": self.thread_id}}


async def _start_target(deps: GraphDeps, payload: dict[str, Any]) -> _Target:
    """Create the WorkflowRun for a `start_workflow` job, or reuse the one that exists."""
    account_id, message_id = payload["account_id"], payload["provider_message_id"]
    thread_id = graph_thread_id(account_id, message_id)
    async with deps.sessionmaker() as session, session.begin():
        message_row_id, provider_thread_id = (
            await session.execute(
                select(EmailMessage.id, EmailThread.provider_thread_id)
                .join(EmailThread, EmailMessage.thread_id == EmailThread.id)
                .where(
                    EmailMessage.account_id == account_id,
                    EmailMessage.provider_message_id == message_id,
                )
            )
        ).one()
        await session.execute(
            insert(WorkflowRun)
            .values(
                graph_thread_id=thread_id,
                message_id=message_row_id,
                provider_thread_id=provider_thread_id,
                graph_version=GRAPH_VERSION,
                prompt_versions=dict(PROMPT_VERSIONS),
            )
            .on_conflict_do_nothing(index_elements=[WorkflowRun.graph_thread_id])
        )
        workflow_id = (
            await session.execute(
                select(WorkflowRun.id).where(WorkflowRun.graph_thread_id == thread_id)
            )
        ).scalar_one()
    initial = {
        "workflow_id": str(workflow_id),
        "account_id": account_id,
        "provider_message_id": message_id,
        "provider_thread_id": provider_thread_id,
    }
    return _Target(workflow_id, thread_id, initial)


async def _resume_target(deps: GraphDeps, payload: dict[str, Any]) -> _Target:
    workflow_id = uuid.UUID(str(payload["workflow_id"]))
    async with deps.sessionmaker() as session:
        run = await session.get_one(WorkflowRun, workflow_id)
        return _Target(workflow_id, run.graph_thread_id)


async def _is_final(deps: GraphDeps, workflow_id: uuid.UUID) -> bool:
    async with deps.sessionmaker() as session:
        status = await session.scalar(
            select(WorkflowRun.status).where(WorkflowRun.id == workflow_id)
        )
    return status in FINAL_STATUSES


async def _invoke(graph: Graph, deps: GraphDeps, target: _Target, value: Any) -> None:
    """Mark the run `running` (unless final), then run the graph until it ends or pauses.

    `value` is the initial state (a new run), `Command(resume=...)` (answer an interrupt) or
    `None` (continue from the last checkpoint).
    """
    async with deps.sessionmaker() as session, session.begin():
        await session.execute(
            update(WorkflowRun)
            .where(
                WorkflowRun.id == target.workflow_id,
                WorkflowRun.status.not_in(FINAL_STATUSES),
            )
            .values(status="running", updated_at=datetime.now(UTC))
        )
    await graph.ainvoke(value, target.config, context=deps, durability="sync")


def _answers_pending(snap: StateSnapshot, payload: dict[str, Any]) -> bool:
    """The job's interrupt is still pending (by id; or by review item, if the id was unknown)."""
    pending = pending_interrupts(snap)
    expected = payload.get("interrupt_id")
    if expected is not None:
        return any(i.id == expected for i in pending)
    item_id = payload.get("review_item_id")
    return item_id is not None and any(
        isinstance(i.value, dict) and i.value.get("review_item_id") == item_id for i in pending
    )


async def _run_start(graph: Graph, deps: GraphDeps, target: _Target) -> None:
    snap = await graph.aget_state(target.config, subgraphs=True)
    if snap.metadata is None:  # no checkpoint yet: a brand-new run
        await _invoke(graph, deps, target, target.initial_state)
    elif pending_interrupts(snap):
        log.info("start job for %s: already waiting for review", target.thread_id)
    elif snap.next and not await _is_final(deps, target.workflow_id):
        await _invoke(graph, deps, target, None)  # an earlier attempt stopped mid-way
    else:
        log.info("start job for %s: run already finished", target.thread_id)


async def _run_resume(
    graph: Graph, deps: GraphDeps, target: _Target, payload: dict[str, Any]
) -> None:
    snap = await graph.aget_state(target.config, subgraphs=True)
    if _answers_pending(snap, payload):
        await _invoke(graph, deps, target, Command(resume=payload["resume"]))
    elif (
        not pending_interrupts(snap) and snap.next and not await _is_final(deps, target.workflow_id)
    ):
        # Resumed before, then stopped mid-way (crash or ActionInFlight): continue.
        await _invoke(graph, deps, target, None)
    else:
        log.info("resume job for %s: interrupt no longer pending; no-op", target.thread_id)


async def _set_run_outcome(
    session: AsyncSession, workflow_id: uuid.UUID, status: str, error_code: str
) -> None:
    """Move a non-final run to `failed` (emitting `run_failed`) or `paused`."""
    updated = (
        await session.execute(
            update(WorkflowRun)
            .where(WorkflowRun.id == workflow_id, WorkflowRun.status.not_in(FINAL_STATUSES))
            .values(status=status, error_code=error_code, updated_at=datetime.now(UTC))
            .returning(WorkflowRun.id)
        )
    ).scalar_one_or_none()
    if updated is not None and status == "failed":
        # A fixed code only: exception text never goes into events.
        await emit(session, workflow_id, "run_failed", {"status": status, "error_code": error_code})


async def _keep_alive(
    sessionmaker: async_sessionmaker[AsyncSession],
    job_id: uuid.UUID,
    worker_id: str,
    lease: timedelta,
) -> None:
    """Renew the lease every `lease / 3` until cancelled or the lease is lost."""
    while True:
        await asyncio.sleep(lease.total_seconds() / 3)
        try:
            async with sessionmaker() as session, session.begin():
                alive = await queue.heartbeat(session, job_id, worker_id, lease)
        except Exception:
            log.exception("heartbeat for job %s failed", job_id)
            continue
        if not alive:
            log.warning("job %s: lease lost to another worker", job_id)
            return


async def _finish(
    deps: GraphDeps,
    job: Job,
    worker_id: str,
    target: _Target | None,
    outcome: str,
    graph: Graph,
) -> None:
    """Record the job's outcome in one transaction, only while this worker holds the lease."""
    snap = (
        await graph.aget_state(target.config, subgraphs=True)
        if target is not None and outcome == "done"
        else None
    )
    async with deps.sessionmaker() as session, session.begin():
        if outcome == "in_flight":
            owned = await queue.reschedule(session, job.id, worker_id, IN_FLIGHT_RETRY)
        elif outcome in ("done", "budget"):
            owned = await queue.complete(session, job.id, worker_id)
        else:
            owned = await queue.fail(session, job.id, worker_id, outcome)
        if not owned:
            log.warning("job %s: lease lost before %s; not overwriting", job.id, outcome)
            return
        if target is None:
            return
        if snap is not None:
            await project_interrupts(session, target.workflow_id, snap)
        elif outcome == "budget":
            await _set_run_outcome(session, target.workflow_id, "paused", "budget")
        elif outcome in ("llm_transient", "internal_error"):
            await _set_run_outcome(session, target.workflow_id, "failed", outcome)


async def _handle(
    graph: Graph, deps: GraphDeps, job: Job, worker_id: str, lease: timedelta
) -> None:
    heart = asyncio.create_task(_keep_alive(deps.sessionmaker, job.id, worker_id, lease))
    target: _Target | None = None
    try:
        try:
            if job.kind == "start_workflow":
                target = await _start_target(deps, job.payload)
                await _run_start(graph, deps, target)
            elif job.kind == "resume_workflow":
                target = await _resume_target(deps, job.payload)
                await _run_resume(graph, deps, target, job.payload)
            else:
                raise ValueError(f"unknown job kind {job.kind!r}")
            outcome = "done"
        except ActionInFlight:
            log.info("job %s: action in flight elsewhere; retrying later", job.id)
            outcome = "in_flight"
        except LLMTransientError:
            log.warning("job %s: model unavailable after bounded retries", job.id)
            outcome = "llm_transient"
        except BudgetExceeded:  # nodes normally finalize as paused; kept as a safety net
            log.warning("job %s: daily model budget exhausted", job.id)
            outcome = "budget"
        except Exception:
            log.exception("job %s (%s) failed", job.id, job.kind)
            outcome = "internal_error"
    finally:
        heart.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heart
    await _finish(deps, job, worker_id, target, outcome, graph)


async def run_once(graph: Graph, deps: GraphDeps, *, worker_id: str, lease: timedelta) -> bool:
    """Claim and handle at most one job; return whether one was handled."""
    async with deps.sessionmaker() as session, session.begin():
        job = await queue.claim(session, worker_id, lease)
    if job is None:
        return False
    log.info("job %s (%s) claimed, attempt %d", job.id, job.kind, job.attempts)
    await _handle(graph, deps, job, worker_id, lease)
    return True


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


async def run_worker(
    settings: Settings,
    stop: asyncio.Event,
    *,
    worker_id: str | None = None,
    deps_factory: DepsFactory = make_deps,
    poll_interval: float = 1.0,
) -> None:
    """Run jobs until `stop` is set; the job in progress is finished first.

    The saver (one Postgres connection) is opened and the graph compiled once, then reused for
    every job: the compiled graph is stateless, all run state lives in the checkpoints.
    """
    worker_id = worker_id or default_worker_id()
    lease = timedelta(seconds=settings.worker_lease_seconds)
    sessionmaker = make_sessionmaker(settings)
    deps = deps_factory(settings, sessionmaker, MockFaults())
    try:
        async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_database_url) as saver:
            graph = build_parent_graph(saver)
            log.info("worker %s started", worker_id)
            while not stop.is_set():
                try:
                    handled = await run_once(graph, deps, worker_id=worker_id, lease=lease)
                except Exception:  # e.g. the database is briefly unreachable
                    log.exception("worker %s: claim loop error", worker_id)
                    handled = False
                if not handled:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stop.wait(), poll_interval)
            log.info("worker %s stopped", worker_id)
    finally:
        await sessionmaker.kw["bind"].dispose()
