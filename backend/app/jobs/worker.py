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
  `ainvoke(Command(resume={interrupt_id: value}), config, ...)`. Keying the resume value by
  the interrupt's id (a "resume map") binds it to exactly the interrupt that was checked: if
  some other interrupt were pending by then, LangGraph would give it nothing, and its node
  would simply pause again. LangGraph loads the checkpoint, re-runs the interrupted node from
  its top, and this time `interrupt()` returns `value`. The graph object and the saver can be
  brand new (a new process): everything needed is in Postgres, and the runtime dependencies
  (`GraphDeps`) are passed fresh on each call, never checkpointed.
- **Final runs.** Once a run is final (cancelled, done, failed), only a `cancelled` or
  `superseded` resume is delivered, so the paused node ends the graph; any other resume (an
  answer or approval queued just before a cancel) is a no-op and the job completes.
- **Continuing after a crash.** If a worker dies mid-run, its job lease expires and another
  worker claims the job again. The checkpoint then has no pending interrupt but a non-empty
  `snap.next` (the step that was running). `ainvoke(None, config, ...)` means "continue from
  the last checkpoint": only the unfinished step runs again. A node that may already have
  done its external write (`execute_action`) consults the action ledger first, so the write
  is reconciled, never repeated (see `app/graphs/nodes/execute.py`).

Leases. The handler runs as its own task, supervised by `_supervise`, which renews the lease
every `lease / 3`. If a renewal reports the lease lost, or no renewal has succeeded by
`last_ok + lease - lease / 6` (the database may be unreachable), the handler is cancelled at
once and nothing more is written for the job: another worker may own it now, and two workers
must never drive the same thread.

Job outcomes: success → projection, then `done`. `ActionInFlight` → the same job is queued
again `IN_FLIGHT_RETRY` later (not a failure, and not counted as an attempt). `LLMTransientError`
→ the client already made its bounded attempts, so the run is `failed` (`llm_transient`) and the
job fails without a retry. Any other exception → the run is `failed` (`internal_error`); the
traceback goes to the server log, never into events or the job row. A job claimed
`MAX_JOB_ATTEMPTS` times without finishing (its workers kept dying) is failed by `claim` itself
(`max_attempts`), and its run with it (`outcome_unknown` if an action may have been sent).
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
from langgraph.types import Command, Interrupt, StateSnapshot
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import (
    Action,
    AppFlags,
    Approval,
    EmailMessage,
    EmailThread,
    Job,
    Proposal,
    ProposalRevision,
    WorkflowRun,
)
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
    engine = create_async_engine(settings.database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                insert(AppFlags)
                .values(id=1, ingestion_paused=False, outbound_paused=False)
                .on_conflict_do_nothing(index_elements=[AppFlags.id])
            )
    finally:
        await engine.dispose()


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
        "automated": bool(payload.get("automated", False)),
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

    `value` is the initial state (a new run), `Command(resume={interrupt_id: ...})` (answer
    that interrupt) or `None` (continue from the last checkpoint).
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


def _matching_interrupt(snap: StateSnapshot, payload: dict[str, Any]) -> Interrupt | None:
    """The pending interrupt this job answers (by id; or by review item, if the id was unknown)."""
    pending = pending_interrupts(snap)
    expected = payload.get("interrupt_id")
    if expected is not None:
        return next((i for i in pending if i.id == expected), None)
    item_id = payload.get("review_item_id")
    if item_id is None:
        return None
    return next(
        (
            i
            for i in pending
            if isinstance(i.value, dict) and i.value.get("review_item_id") == item_id
        ),
        None,
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


# The only resumes delivered to a final (e.g. cancelled) run: they end the graph cleanly.
ENDING_DECISIONS = ("cancelled", "superseded")


async def _run_resume(
    graph: Graph, deps: GraphDeps, target: _Target, payload: dict[str, Any]
) -> None:
    decision = (payload.get("resume") or {}).get("decision")
    if decision not in ENDING_DECISIONS and await _is_final(deps, target.workflow_id):
        # E.g. an answer queued just before the owner cancelled: the run must not continue.
        log.info("resume job for %s: run is final; %r not delivered", target.thread_id, decision)
        return
    snap = await graph.aget_state(target.config, subgraphs=True)
    matched = _matching_interrupt(snap, payload)
    if matched is not None:
        # Keyed by id: the value can only answer the interrupt validated just now.
        await _invoke(graph, deps, target, Command(resume={matched.id: payload["resume"]}))
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


async def _exhausted_run_id(session: AsyncSession, payload: dict[str, Any]) -> uuid.UUID | None:
    """The run a job acts on: `workflow_id` (resume jobs) or the thread id (start jobs)."""
    if "workflow_id" in payload:
        return uuid.UUID(str(payload["workflow_id"]))
    if {"account_id", "provider_message_id"} <= payload.keys():
        thread_id = graph_thread_id(payload["account_id"], payload["provider_message_id"])
        return await session.scalar(
            select(WorkflowRun.id).where(WorkflowRun.graph_thread_id == thread_id)
        )
    return None


async def _has_unresolved_action(session: AsyncSession, workflow_id: uuid.UUID) -> bool:
    """Whether some action of the run may or may not have reached the provider."""
    found = await session.scalar(
        select(Action.id)
        .join(Approval, Action.approval_id == Approval.id)
        .join(ProposalRevision, Approval.proposal_revision_id == ProposalRevision.id)
        .join(Proposal, ProposalRevision.proposal_id == Proposal.id)
        .where(
            Proposal.workflow_id == workflow_id,
            Action.status.in_(("executing", "outcome_unknown")),
        )
        .limit(1)
    )
    return found is not None


async def fail_exhausted_run(session: AsyncSession, job: Job) -> None:
    """`claim` gave up on `job` (too many attempts): fail its run too, if it has one.

    The error code is `outcome_unknown` if one of the run's actions is still `executing` or
    `outcome_unknown` (the write may have happened, so the owner must check), otherwise
    `internal_error`. A final run is left alone.

    This runs inside the claim's transaction, so it must never raise: a failure here would roll
    back the claim, leave the job runnable and wedge every later claim on it. Its writes go in
    a savepoint (`begin_nested`), so on any error only they are undone, the error is logged,
    and the job is still marked `failed`.
    """
    job_id, kind, attempts, payload = job.id, job.kind, job.attempts, dict(job.payload)
    log.error("job %s (%s) failed after %d attempts", job_id, kind, attempts)
    try:
        async with session.begin_nested():
            workflow_id = await _exhausted_run_id(session, payload)
            if workflow_id is None:
                return
            unresolved = await _has_unresolved_action(session, workflow_id)
            code = "outcome_unknown" if unresolved else "internal_error"
            await _set_run_outcome(session, workflow_id, "failed", code)
    except Exception:
        log.exception("job %s (%s): could not fail the run of an exhausted job", job_id, kind)


async def _work(graph: Graph, deps: GraphDeps, job: Job) -> tuple[str, _Target | None]:
    """Run the job's handler; return its outcome code and the run it acted on."""
    target: _Target | None = None
    try:
        if job.kind == "start_workflow":
            target = await _start_target(deps, job.payload)
            await _run_start(graph, deps, target)
        elif job.kind == "resume_workflow":
            target = await _resume_target(deps, job.payload)
            await _run_resume(graph, deps, target, job.payload)
        else:
            raise ValueError(f"unknown job kind {job.kind!r}")
        return "done", target
    except ActionInFlight:
        log.info("job %s: action in flight elsewhere; retrying later", job.id)
        return "in_flight", target
    except LLMTransientError:
        log.warning("job %s: model unavailable after bounded retries", job.id)
        return "llm_transient", target
    except BudgetExceeded:  # nodes normally finalize as paused; kept as a safety net
        log.warning("job %s: daily model budget exhausted", job.id)
        return "budget", target
    except Exception:
        log.exception("job %s (%s) failed", job.id, job.kind)
        return "internal_error", target


async def _supervise(
    work: "asyncio.Task[Any]",
    sessionmaker: async_sessionmaker[AsyncSession],
    job_id: uuid.UUID,
    worker_id: str,
    lease: timedelta,
) -> bool:
    """Heartbeat while `work` runs; cancel it if the lease is lost. True if it was cancelled.

    `last_ok` is when the last successful renewal *started* (the claim counts as one), so the
    lease in the database lasts at least until `last_ok + lease`. The handler is cancelled as
    soon as a renewal reports the lease gone, or once `last_ok + lease - lease / 6` passes
    without a successful renewal (heartbeats failing, e.g. the database is unreachable).
    """
    loop = asyncio.get_running_loop()
    interval, budget = lease.total_seconds() / 3, lease.total_seconds() * 5 / 6
    last_ok = loop.time()
    next_beat = last_ok + interval
    while True:
        deadline = last_ok + budget
        await asyncio.wait({work}, timeout=max(0.0, min(next_beat, deadline) - loop.time()))
        if work.done():
            return False
        started = loop.time()
        if started >= deadline:
            reason = "no successful heartbeat in time"
            break
        next_beat = started + interval
        try:
            async with asyncio.timeout(deadline - started):
                async with sessionmaker() as session, session.begin():
                    alive = await queue.heartbeat(session, job_id, worker_id, lease)
        except Exception:  # TimeoutError included; retried until the deadline
            log.warning("heartbeat for job %s failed", job_id, exc_info=True)
            continue
        if not alive:
            reason = "lease taken over"
            break
        last_ok = started
    log.warning("job %s: lease lost (%s); cancelling the handler", job_id, reason)
    work.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await work
    return True


async def _finish(
    graph: Graph,
    deps: GraphDeps,
    job: Job,
    worker_id: str,
    target: _Target | None,
    outcome: str,
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
    work = asyncio.create_task(_work(graph, deps, job))
    try:
        if await _supervise(work, deps.sessionmaker, job.id, worker_id, lease):
            return  # another worker may own the job now: write nothing
    finally:
        if not work.done():  # we were cancelled ourselves (shutdown, a killed process)
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
    outcome, target = work.result()
    await _finish(graph, deps, job, worker_id, target, outcome)


async def run_once(graph: Graph, deps: GraphDeps, *, worker_id: str, lease: timedelta) -> bool:
    """Claim and handle at most one job; return whether one was handled."""
    async with deps.sessionmaker() as session, session.begin():
        job = await queue.claim(session, worker_id, lease, on_exhausted=fail_exhausted_run)
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
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
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
        await engine.dispose()
