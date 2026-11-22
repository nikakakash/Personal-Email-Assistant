"""The durable worker end to end: ingestion → worker → interrupt projection → decision → resume.

Every test uses the real `AsyncPostgresSaver` on the test DB. A "worker restart" is a new
saver and a new compiled graph (`open_graph`) plus fresh `GraphDeps`; nothing survives in
memory, only in Postgres.
"""

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from langgraph.types import Command
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.__main__ import main
from app.actions.decisions import DecisionConflict, DecisionIn, decide_proposal
from app.actions.executor import operation_id_for
from app.actions.proposals import active_revision
from app.db.models import (
    Action,
    AppFlags,
    Job,
    MockOutbox,
    Owner,
    Proposal,
    ReviewItem,
    WorkflowEvent,
    WorkflowRun,
)
from app.graphs.deps import GraphDeps
from app.graphs.nodes.notify import notification_item_id
from app.graphs.parent import GRAPH_VERSION
from app.ingestion.ingest import ingest_message
from app.jobs import queue, worker
from app.jobs.projection import pending_interrupts
from app.jobs.queue import MAX_JOB_ATTEMPTS, claim, enqueue
from app.jobs.worker import Graph, fail_exhausted_run, run_once
from app.llm.client import BudgetExceeded, LLMCall, LLMResult, LLMTransientError, T
from app.llm.scripted import ScriptedLLMClient
from app.prompts import PROMPT_VERSIONS
from app.providers.fixtures import load_fixture
from app.providers.interfaces import OutgoingReply, ProviderResult, ProviderTimeout, ThreadSnapshot
from app.providers.mock_mail import MockFaults, MockMailProvider
from tests.jobs.conftest import DepsBuilder, open_graph

Session = async_sessionmaker[AsyncSession]
LEASE = timedelta(seconds=60)
OWNER = "owner@example.com"
FIXTURE = "sched-tomorrow-review"
MID = "sched-tomorrow-review-m1"
THREAD = "thr-sched-tomorrow-review"
SENDER = "sara.ng@work.example.com"
FRIDAY = "2026-11-20"
AVAILABILITY = f"availability:{FRIDAY}:{FRIDAY}"

# Inline scripted model outputs: triage → free/busy read → draft a reply → (after approval) finish.
BEFORE_APPROVAL: dict[str, Any] = {
    MID: {
        "triage": [
            {
                "classification": "respond",
                "urgency": "normal",
                "reason": "Sara asks for a 30-minute review tomorrow afternoon.",
                "evidence_refs": [f"message:{MID}"],
                "needs_clarification": False,
            }
        ],
        "plan": [
            {
                "action": "get_freebusy",
                "reason": "Check tomorrow's calendar.",
                "freebusy": {"start_date": FRIDAY, "end_date": FRIDAY},
            },
            {"action": "draft_reply", "reason": "Offer a free slot."},
        ],
        "draft_reply": [
            {
                "to": [SENDER],
                "cc": [],
                "body_text": "Friday at 13:00 UTC works for me.",
                "evidence_refs": [f"message:{MID}", AVAILABILITY],
            }
        ],
    }
}
AFTER_APPROVAL: dict[str, Any] = {
    MID: {"plan": [{"action": "finish", "reason": "The reply was sent."}]}
}


# ------------------------------------------------------------------------------- helpers


async def _ingest(sessionmaker: Session) -> None:
    fixture = load_fixture(FIXTURE)
    async with sessionmaker() as s:
        for message in fixture.thread:
            assert (await ingest_message(s, message, [OWNER])).status == "queued"
        await s.commit()


async def _work(graph: Graph, deps: GraphDeps, worker_id: str = "w1") -> bool:
    return await run_once(graph, deps, worker_id=worker_id, lease=LEASE)


async def _run(sessionmaker: Session) -> WorkflowRun:
    async with sessionmaker() as s:
        return (await s.execute(select(WorkflowRun))).scalar_one()


async def _jobs(sessionmaker: Session, kind: str) -> list[Job]:
    async with sessionmaker() as s:
        rows = await s.execute(select(Job).where(Job.kind == kind).order_by(Job.scheduled_at))
        return list(rows.scalars())


async def _count(sessionmaker: Session, model: Any, *where: Any) -> int:
    async with sessionmaker() as s:
        return int(await s.scalar(select(func.count()).select_from(model).where(*where)) or 0)


async def _decision(
    sessionmaker: Session, *, request_id: str | None = None
) -> tuple[uuid.UUID, DecisionIn]:
    """An approval of the pending proposal, using the interrupt id the projection stored."""
    async with sessionmaker() as s:
        item = (
            await s.execute(
                select(ReviewItem).where(
                    ReviewItem.kind == "proposal", ReviewItem.status == "pending"
                )
            )
        ).scalar_one()
        assert item.interrupt_id is not None, "projection did not set interrupt_id"
        assert item.proposal_id is not None
        revision = await active_revision(s, await s.get_one(Proposal, item.proposal_id))
        body = DecisionIn(
            decision="approve",
            proposal_revision=revision.revision,
            payload_hash=revision.payload_hash,
            interrupt_id=item.interrupt_id,
            request_id=request_id or uuid.uuid4().hex,
        )
        return item.proposal_id, body


async def _approve(sessionmaker: Session, owner_id: uuid.UUID, mail: MockMailProvider) -> None:
    proposal_id, body = await _decision(sessionmaker)
    async with sessionmaker() as s:
        await decide_proposal(s, owner_id, proposal_id, body, mail=mail)


async def _pause_for_approval(sessionmaker: Session, make_deps: DepsBuilder, url: str) -> None:
    """Ingest the fixture and run the start job until the approval interrupt."""
    await _ingest(sessionmaker)
    async with open_graph(url) as graph:
        assert await _work(graph, make_deps(BEFORE_APPROVAL))


async def _action_in_flight(
    sessionmaker: Session, owner_id: uuid.UUID, mail: MockMailProvider, status: str = "executing"
) -> uuid.UUID:
    """Approve the pending proposal, then record its action as `status` by another caller."""
    proposal_id, body = await _decision(sessionmaker)
    async with sessionmaker() as s:
        outcome = await decide_proposal(s, owner_id, proposal_id, body, mail=mail)
    action_id = uuid.uuid4()
    async with sessionmaker() as s, s.begin():
        s.add(
            Action(
                id=action_id,
                operation_id=operation_id_for(proposal_id, outcome.revision),
                approval_id=outcome.approval_id,
                kind="send_reply",
                request_fingerprint="in-flight",
                status=status,
                attempts=1,
                updated_at=datetime.now(UTC),
            )
        )
    return action_id


async def _make_due(sessionmaker: Session, job_id: uuid.UUID) -> None:
    async with sessionmaker() as s, s.begin():
        await s.execute(update(Job).where(Job.id == job_id).values(scheduled_at=func.now()))


async def _exhaust(sessionmaker: Session, job_id: uuid.UUID) -> None:
    """The job has crashed its worker MAX_JOB_ATTEMPTS times; its last lease has expired."""
    async with sessionmaker() as s, s.begin():
        await s.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                state="running",
                attempts=MAX_JOB_ATTEMPTS,
                lease_owner="w-dead",
                lease_until=func.now() - LEASE,
                scheduled_at=func.now() - 2 * LEASE,  # claimed ahead of any job queued now
            )
        )


async def _claim_exhausting(sessionmaker: Session) -> Job | None:
    async with sessionmaker() as s, s.begin():
        return await claim(s, "w2", LEASE, on_exhausted=fail_exhausted_run)


async def _run_failed_payloads(sessionmaker: Session) -> list[dict[str, Any]]:
    async with sessionmaker() as s:
        rows = await s.execute(
            select(WorkflowEvent.payload).where(WorkflowEvent.event_type == "run_failed")
        )
        return list(rows.scalars())


class CountingLLM:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        self.calls += 1
        raise self.exc


# ------------------------------------------------------------------------------- tests


async def test_pending_review_survives_worker_restart(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    await _ingest(sessionmaker)

    async with open_graph(url) as before:  # worker 1 runs until the approval interrupt
        assert await _work(before, make_deps(BEFORE_APPROVAL), "w1")
    # Worker 1 is gone: its saver is closed and its graph object discarded.

    run = await _run(sessionmaker)
    assert run.graph_thread_id == f"mock-account:{MID}:{GRAPH_VERSION}"
    assert (run.status, run.provider_thread_id, run.graph_version) == (
        "waiting_review",
        THREAD,
        GRAPH_VERSION,
    )
    assert run.prompt_versions == PROMPT_VERSIONS
    (start,) = await _jobs(sessionmaker, "start_workflow")
    assert (start.state, start.attempts) == ("done", 1)
    assert await _count(sessionmaker, MockOutbox) == 0

    await _approve(sessionmaker, owner_id, mail)

    async with open_graph(url) as after:  # worker 2: new saver, new graph, fresh deps
        assert await _work(after, make_deps(AFTER_APPROVAL), "w2")
        assert not await _work(after, make_deps(), "w2")  # the queue is empty

    assert await _count(sessionmaker, MockOutbox) == 1
    run = await _run(sessionmaker)
    assert (run.status, run.error_code) == ("done_with_actions", None)
    (resume,) = await _jobs(sessionmaker, "resume_workflow")
    assert resume.state == "done"
    async with sessionmaker() as s:
        assert (await s.execute(select(Proposal.status))).scalar_one() == "executed"


async def test_projection_sets_interrupt_id_for_subgraph_interrupt(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    url = worker_settings.checkpoint_database_url
    await _pause_for_approval(sessionmaker, make_deps, url)

    async with open_graph(url) as graph:
        run = await _run(sessionmaker)
        snap = await graph.aget_state(
            {"configurable": {"thread_id": run.graph_thread_id}}, subgraphs=True
        )
    # The approval interrupt is raised inside the `response` subgraph.
    (interrupt,) = pending_interrupts(snap)
    assert interrupt.value["type"] == "approval"
    async with sessionmaker() as s:
        item = await s.get_one(ReviewItem, uuid.UUID(interrupt.value["review_item_id"]))
    assert (item.kind, item.status, item.interrupt_id) == ("proposal", "pending", interrupt.id)
    assert run.status == "waiting_review"


async def test_projection_sets_interrupt_id_for_parent_interrupt(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    url = worker_settings.checkpoint_database_url
    await _ingest(sessionmaker)
    triage = dict(BEFORE_APPROVAL[MID]["triage"][0], classification="notify")
    async with open_graph(url) as graph:
        assert await _work(graph, make_deps({MID: {"triage": [triage]}}))
        run = await _run(sessionmaker)
        snap = await graph.aget_state(
            {"configurable": {"thread_id": run.graph_thread_id}}, subgraphs=True
        )
    # The notification interrupt is raised by a leaf node of the parent graph itself.
    assert snap.next == ("notification_interrupt",)
    (interrupt,) = pending_interrupts(snap)
    item_id = notification_item_id(str(run.id))
    assert interrupt.value == {"type": "notification", "review_item_id": str(item_id)}
    async with sessionmaker() as s:
        item = await s.get_one(ReviewItem, item_id)
    assert (item.kind, item.status, item.interrupt_id) == ("notification", "pending", interrupt.id)
    assert run.status == "waiting_review"


async def test_duplicate_resume_job_is_noop(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    await _pause_for_approval(sessionmaker, make_deps, url)
    await _approve(sessionmaker, owner_id, mail)
    (resume,) = await _jobs(sessionmaker, "resume_workflow")
    async with sessionmaker() as s, s.begin():  # the same interrupt resumed a second time
        await enqueue(s, "resume_workflow", "resume:duplicate", dict(resume.payload))

    async with open_graph(url) as graph:
        assert await _work(graph, make_deps(AFTER_APPROVAL))
        # No model script left: a real second resume would fail loudly.
        assert await _work(graph, make_deps())

    assert await _count(sessionmaker, MockOutbox) == 1
    assert [j.state for j in await _jobs(sessionmaker, "resume_workflow")] == ["done", "done"]
    assert (await _run(sessionmaker)).status == "done_with_actions"
    assert (
        await _count(sessionmaker, WorkflowEvent, WorkflowEvent.event_type == "run_completed") == 1
    )


class CrashAfterAccept:
    """The provider accepts the send, then the worker process dies before recording it."""

    def __init__(self, inner: MockMailProvider) -> None:
        self.inner = inner
        self.accepted = asyncio.Event()

    async def get_thread(self, provider_thread_id: str, limit: int = 20) -> ThreadSnapshot:
        return await self.inner.get_thread(provider_thread_id, limit)

    async def send_reply(self, req: OutgoingReply, operation_id: str) -> ProviderResult:
        try:
            return await self.inner.send_reply(req, operation_id)
        except ProviderTimeout:
            self.accepted.set()
            await asyncio.Event().wait()  # hangs until the test kills the "process"
            raise

    async def find_sent(self, operation_id: str) -> ProviderResult | None:
        return await self.inner.find_sent(operation_id)


async def test_crash_after_provider_accept_recovers_via_ledger(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
    faults: MockFaults,
) -> None:
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    await _pause_for_approval(sessionmaker, make_deps, url)
    await _approve(sessionmaker, owner_id, mail)
    faults.send = "timeout_after_accept"
    crashing = CrashAfterAccept(mail)

    async with open_graph(url) as doomed:
        task = asyncio.create_task(_work(doomed, make_deps(mail=crashing), "w1"))
        await asyncio.wait_for(crashing.accepted.wait(), 10)
        task.cancel()  # the worker process is killed mid-call
        with contextlib.suppress(asyncio.CancelledError):
            await task

    (resume,) = await _jobs(sessionmaker, "resume_workflow")
    assert (resume.state, resume.lease_owner) == ("running", "w1")
    async with sessionmaker() as s:
        action = (await s.execute(select(Action))).scalar_one()
    assert action.status == "executing"  # claimed, sent, never recorded
    assert await _count(sessionmaker, MockOutbox) == 1
    # Time passes: w1's lease expires and the in-flight grace for its claim runs out.
    async with sessionmaker() as s, s.begin():
        await s.execute(
            update(Job).where(Job.id == resume.id).values(lease_until=func.now() - LEASE)
        )
        await s.execute(
            update(Action)
            .where(Action.id == action.id)
            .values(updated_at=func.now() - timedelta(minutes=10))
        )
    faults.send = "ok"

    async with open_graph(url) as restarted:
        assert await _work(restarted, make_deps(AFTER_APPROVAL), "w2")

    async with sessionmaker() as s:
        action = await s.get_one(Action, action.id)
        job = await s.get_one(Job, resume.id)
    assert action.status == "succeeded"  # reconciled from the provider, not re-sent
    assert await _count(sessionmaker, MockOutbox) == 1
    assert (job.state, job.lease_owner, job.attempts) == ("done", None, 2)
    assert (await _run(sessionmaker)).status == "done_with_actions"


async def test_resume_continues_after_action_in_flight(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    await _pause_for_approval(sessionmaker, make_deps, url)
    action_id = await _action_in_flight(sessionmaker, owner_id, mail)  # another caller sends

    async with open_graph(url) as graph:
        deps = make_deps(AFTER_APPROVAL)
        assert await _work(graph, deps)

        (job,) = await _jobs(sessionmaker, "resume_workflow")
        async with sessionmaker() as s:
            due_at, db_now = (
                await s.execute(select(Job.scheduled_at, func.now()).where(Job.id == job.id))
            ).one()
        wait = due_at - db_now
        # Waiting for another caller is not an attempt: the claim's increment is undone.
        assert (job.state, job.attempts, job.error, job.lease_owner) == ("queued", 0, None, None)
        assert timedelta(seconds=25) < wait <= timedelta(seconds=30)
        assert not await _work(graph, deps)  # not due yet

        async with sessionmaker() as s, s.begin():  # the other caller's send succeeded
            await s.execute(
                update(Action)
                .where(Action.id == action_id)
                .values(status="succeeded", provider_ref="sent-by-other-caller")
            )
        await _make_due(sessionmaker, job.id)
        assert await _work(graph, deps)

    (job,) = await _jobs(sessionmaker, "resume_workflow")
    assert (job.state, job.attempts) == ("done", 1)
    assert (await _run(sessionmaker)).status == "done_with_actions"
    assert await _count(sessionmaker, Action) == 1
    assert await _count(sessionmaker, MockOutbox) == 0  # nothing sent twice


async def test_in_flight_reschedules_do_not_exhaust_attempts(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    await _pause_for_approval(sessionmaker, make_deps, url)
    action_id = await _action_in_flight(sessionmaker, owner_id, mail)
    (job,) = await _jobs(sessionmaker, "resume_workflow")

    async with open_graph(url) as graph:
        deps = make_deps(AFTER_APPROVAL)
        for _ in range(MAX_JOB_ATTEMPTS + 1):  # the other caller is still sending each time
            assert await _work(graph, deps)
            (job,) = await _jobs(sessionmaker, "resume_workflow")
            assert (job.state, job.attempts, job.error) == ("queued", 0, None)
            await _make_due(sessionmaker, job.id)

        async with sessionmaker() as s, s.begin():  # the other caller's send succeeded
            await s.execute(
                update(Action)
                .where(Action.id == action_id)
                .values(status="succeeded", provider_ref="sent-by-other-caller")
            )
        assert await _work(graph, deps)

    (job,) = await _jobs(sessionmaker, "resume_workflow")
    assert (job.state, job.attempts, job.error) == ("done", 1, None)
    assert (await _run(sessionmaker)).status == "done_with_actions"
    assert await _count(sessionmaker, Action) == 1
    assert await _count(sessionmaker, MockOutbox) == 0  # nothing sent twice


async def test_concurrent_approvals_one_wins(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    await _pause_for_approval(sessionmaker, make_deps, url)
    proposal_id, first = await _decision(sessionmaker, request_id="click-1")
    second = first.model_copy(update={"request_id": "click-2"})

    async def submit(body: DecisionIn) -> Any:
        async with sessionmaker() as s:
            return await decide_proposal(s, owner_id, proposal_id, body, mail=mail)

    results = await asyncio.gather(submit(first), submit(second), return_exceptions=True)

    accepted = [r for r in results if not isinstance(r, BaseException)]
    conflicts = [r for r in results if isinstance(r, DecisionConflict)]
    assert (len(accepted), len(conflicts)) == (1, 1)
    assert len(await _jobs(sessionmaker, "resume_workflow")) == 1
    async with open_graph(url) as graph:
        assert await _work(graph, make_deps(AFTER_APPROVAL))
    assert await _count(sessionmaker, MockOutbox) == 1


async def test_budget_exhausted_pauses_workflow(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    await _ingest(sessionmaker)
    llm = CountingLLM(BudgetExceeded("daily budget reached"))

    async with open_graph(worker_settings.checkpoint_database_url) as graph:
        assert await _work(graph, make_deps(llm=llm))

    run = await _run(sessionmaker)
    assert (run.status, run.error_code) == ("paused", "budget")
    (job,) = await _jobs(sessionmaker, "start_workflow")
    assert job.state == "done"
    assert llm.calls == 1


async def test_llm_transient_marks_run_failed_without_rerun(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    await _ingest(sessionmaker)
    llm = CountingLLM(LLMTransientError("upstream 529"))

    async with open_graph(worker_settings.checkpoint_database_url) as graph:
        assert await _work(graph, make_deps(llm=llm))
        assert not await _work(graph, make_deps(llm=llm))  # nothing was re-queued

    assert llm.calls == 1  # the client's own bounded retries are the only retries
    run = await _run(sessionmaker)
    assert (run.status, run.error_code) == ("failed", "llm_transient")
    (job,) = await _jobs(sessionmaker, "start_workflow")
    assert (job.state, job.error, job.attempts) == ("failed", "llm_transient", 1)
    async with sessionmaker() as s:
        (payload,) = (
            await s.execute(
                select(WorkflowEvent.payload).where(WorkflowEvent.event_type == "run_failed")
            )
        ).scalars()
    assert payload == {"status": "failed", "error_code": "llm_transient"}  # no exception text


async def test_migrate_seeds_app_flags_idempotently(
    sessionmaker: Session,
    worker_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", worker_settings.database_url)
    monkeypatch.setenv("CHECKPOINT_DATABASE_URL", worker_settings.checkpoint_database_url)

    assert await asyncio.to_thread(main, ["migrate"]) == 0
    async with sessionmaker() as s:
        flags = await s.get_one(AppFlags, 1)
        assert (flags.ingestion_paused, flags.outbound_paused) == (False, False)
        flags.outbound_paused = True  # the owner pauses outbound; a re-migrate keeps it
        await s.commit()

    assert await asyncio.to_thread(main, ["migrate"]) == 0

    assert await _count(sessionmaker, AppFlags) == 1
    async with sessionmaker() as s:
        assert (await s.get_one(AppFlags, 1)).outbound_paused is True


class SlowBudgetLLM:
    """Takes several lease periods to answer, then reports the budget as exhausted."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.started = asyncio.Event()

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        self.started.set()
        await asyncio.sleep(self.seconds)
        raise BudgetExceeded("daily budget reached")


async def test_heartbeat_keeps_lease_while_handler_runs(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    await _ingest(sessionmaker)
    lease = timedelta(seconds=2)
    llm = SlowBudgetLLM(seconds=5.0)

    async with open_graph(worker_settings.checkpoint_database_url) as graph:
        task = asyncio.create_task(run_once(graph, make_deps(llm=llm), worker_id="w1", lease=lease))
        await asyncio.wait_for(llm.started.wait(), 10)
        await asyncio.sleep(4)  # two lease lengths: only heartbeats keep it alive
        async with sessionmaker() as s, s.begin():
            assert await claim(s, "w2", lease) is None
        assert await task

    (job,) = await _jobs(sessionmaker, "start_workflow")
    assert (job.state, job.attempts) == ("done", 1)


class GraphSpy:
    """Delegates to a compiled graph and records every input passed to `ainvoke`."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.inputs: list[Any] = []

    @property
    def resumes(self) -> list[Any]:
        return [i.resume for i in self.inputs if isinstance(i, Command)]

    async def ainvoke(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        self.inputs.append(value)
        return await self.graph.ainvoke(value, *args, **kwargs)

    async def aget_state(self, *args: Any, **kwargs: Any) -> Any:
        return await self.graph.aget_state(*args, **kwargs)


class HangingThreadRead:
    """Mail provider whose thread read (preflight's first call) hangs until cancelled."""

    def __init__(self, inner: MockMailProvider) -> None:
        self.inner = inner
        self.entered = asyncio.Event()
        self.cancelled = False

    async def get_thread(self, provider_thread_id: str, limit: int = 20) -> ThreadSnapshot:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")

    async def send_reply(self, req: OutgoingReply, operation_id: str) -> ProviderResult:
        return await self.inner.send_reply(req, operation_id)

    async def find_sent(self, operation_id: str) -> ProviderResult | None:
        return await self.inner.find_sent(operation_id)


@pytest.mark.parametrize("heartbeat_result", ["lease_taken", "db_error"])
async def test_handler_cancelled_when_lease_lost(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
    monkeypatch: pytest.MonkeyPatch,
    heartbeat_result: str,
) -> None:
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    lease = timedelta(seconds=6)  # cancelled at 2 s (lease taken) or 5 s (no renewal)
    await _pause_for_approval(sessionmaker, make_deps, url)
    await _approve(sessionmaker, owner_id, mail)
    (resume,) = await _jobs(sessionmaker, "resume_workflow")
    real_heartbeat = queue.heartbeat

    async def broken_heartbeat(*args: Any, **kwargs: Any) -> bool:
        if heartbeat_result == "db_error":
            raise OSError("database unreachable")
        return False

    monkeypatch.setattr(queue, "heartbeat", broken_heartbeat)
    hanging = HangingThreadRead(mail)
    async with open_graph(url) as graph:
        spy = GraphSpy(graph)
        task = asyncio.create_task(
            run_once(
                cast(Graph, spy),
                make_deps(AFTER_APPROVAL, mail=hanging),
                worker_id="w1",
                lease=lease,
            )
        )
        await asyncio.wait_for(hanging.entered.wait(), 10)
        assert await asyncio.wait_for(task, 10)  # run_once returns: the handler was cancelled
        assert hanging.cancelled
        async with sessionmaker() as s, s.begin():  # ...before w1's lease ran out in the DB
            assert await claim(s, "w2", lease) is None

    (job,) = await _jobs(sessionmaker, "resume_workflow")
    assert (job.state, job.lease_owner, job.error) == ("running", "w1", None)  # nothing written
    assert await _count(sessionmaker, MockOutbox) == 0
    assert spy.resumes == [{resume.payload["interrupt_id"]: resume.payload["resume"]}]

    monkeypatch.setattr(queue, "heartbeat", real_heartbeat)
    async with sessionmaker() as s, s.begin():  # w1's lease expires
        await s.execute(update(Job).where(Job.id == job.id).values(lease_until=func.now()))
    async with open_graph(url) as graph:
        spy2 = GraphSpy(graph)
        assert await _work(cast(Graph, spy2), make_deps(AFTER_APPROVAL), "w2")

    assert spy2.inputs == [None]  # w2 continues; the resume reached the graph only once
    assert await _count(sessionmaker, MockOutbox) == 1
    assert (await _run(sessionmaker)).status == "done_with_actions"


async def test_resume_with_old_interrupt_id_answers_nothing(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    """The owner gave feedback on proposal 1; a late resume for its interrupt must not answer
    the newer interrupt for proposal 2, neither through the worker nor through LangGraph."""
    owner_id = owner.id
    url = worker_settings.checkpoint_database_url
    await _pause_for_approval(sessionmaker, make_deps, url)
    proposal_id, approve = await _decision(sessionmaker)
    old_interrupt = approve.interrupt_id
    feedback = DecisionIn(
        decision="feedback",
        proposal_revision=approve.proposal_revision,
        payload_hash=approve.payload_hash,
        interrupt_id=old_interrupt,
        request_id=uuid.uuid4().hex,
        feedback="Shorter, please.",
    )
    async with sessionmaker() as s:
        await decide_proposal(s, owner_id, proposal_id, feedback, mail=mail)
    redraft: dict[str, Any] = {
        MID: {
            "plan": [{"action": "draft_reply", "reason": "Apply the feedback."}],
            "draft_reply": [
                {
                    "to": [SENDER],
                    "cc": [],
                    "body_text": "Friday 13:00 UTC works.",
                    "evidence_refs": [f"message:{MID}", AVAILABILITY],
                }
            ],
        }
    }
    run = await _run(sessionmaker)
    config: Any = {"configurable": {"thread_id": run.graph_thread_id}}
    async with open_graph(url) as graph:
        assert await _work(graph, make_deps(redraft))  # pauses on proposal 2
        (newer,) = pending_interrupts(await graph.aget_state(config, subgraphs=True))
        assert newer.id != old_interrupt
        async with sessionmaker() as s, s.begin():  # a late resume for the old interrupt
            await enqueue(
                s,
                "resume_workflow",
                "resume:late",
                {
                    "workflow_id": str(run.id),
                    "interrupt_id": old_interrupt,
                    "resume": {"decision": "reject"},
                },
            )

        spy = GraphSpy(graph)
        assert await _work(cast(Graph, spy), make_deps())
        assert spy.inputs == []  # the worker never invoked the graph

        # Even straight through LangGraph, a value keyed by the old id answers nothing.
        await graph.ainvoke(
            Command(resume={old_interrupt: {"decision": "reject"}}),
            config,
            context=make_deps(),
            durability="sync",
        )
        (still,) = pending_interrupts(await graph.aget_state(config, subgraphs=True))

    assert still.id == newer.id
    late = next(
        j for j in await _jobs(sessionmaker, "resume_workflow") if j.dedup_key == "resume:late"
    )
    assert late.state == "done"
    assert (await _run(sessionmaker)).status == "waiting_review"
    async with sessionmaker() as s:
        statuses = (await s.execute(select(Proposal.status).order_by(Proposal.created_at))).all()
    assert [p for (p,) in statuses] == ["superseded", "awaiting_approval"]


async def test_claim_fails_job_after_max_attempts(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    owner_id = owner.id
    await _pause_for_approval(sessionmaker, make_deps, worker_settings.checkpoint_database_url)
    await _approve(sessionmaker, owner_id, mail)
    (resume,) = await _jobs(sessionmaker, "resume_workflow")
    await _exhaust(sessionmaker, resume.id)
    async with sessionmaker() as s, s.begin():
        other = await enqueue(s, "start_workflow", "other-job", {"n": 1})

    claimed = await _claim_exhausting(sessionmaker)

    assert claimed is not None and claimed.id == other.id  # skipped to the next job
    async with sessionmaker() as s:
        job = await s.get_one(Job, resume.id)
    assert (job.state, job.error, job.lease_owner, job.attempts) == (
        "failed",
        "max_attempts",
        None,
        MAX_JOB_ATTEMPTS,
    )
    run = await _run(sessionmaker)
    assert (run.status, run.error_code) == ("failed", "internal_error")
    assert await _run_failed_payloads(sessionmaker) == [
        {"status": "failed", "error_code": "internal_error"}
    ]


async def test_exhausted_start_job_finds_run_by_thread_id(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    await _pause_for_approval(sessionmaker, make_deps, worker_settings.checkpoint_database_url)
    (start,) = await _jobs(sessionmaker, "start_workflow")
    assert "workflow_id" not in start.payload  # found through graph_thread_id instead
    await _exhaust(sessionmaker, start.id)

    assert await _claim_exhausting(sessionmaker) is None

    run = await _run(sessionmaker)
    assert (run.status, run.error_code) == ("failed", "internal_error")
    assert await _run_failed_payloads(sessionmaker) == [
        {"status": "failed", "error_code": "internal_error"}
    ]


async def test_exhausted_job_leaves_final_run_alone(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    await _pause_for_approval(sessionmaker, make_deps, worker_settings.checkpoint_database_url)
    async with sessionmaker() as s, s.begin():
        await s.execute(update(WorkflowRun).values(status="done_no_action", error_code=None))
    (start,) = await _jobs(sessionmaker, "start_workflow")
    await _exhaust(sessionmaker, start.id)

    assert await _claim_exhausting(sessionmaker) is None

    async with sessionmaker() as s:
        assert (await s.get_one(Job, start.id)).state == "failed"
    run = await _run(sessionmaker)
    assert (run.status, run.error_code) == ("done_no_action", None)
    assert await _run_failed_payloads(sessionmaker) == []


@pytest.mark.parametrize("status", ["executing", "outcome_unknown"])
async def test_exhausted_run_with_unresolved_action_is_outcome_unknown(
    sessionmaker: Session,
    make_deps: DepsBuilder,
    worker_settings: Any,
    owner: Owner,
    mail: MockMailProvider,
    status: str,
) -> None:
    """The send may have happened: the owner must check, so not a plain internal error."""
    owner_id = owner.id
    await _pause_for_approval(sessionmaker, make_deps, worker_settings.checkpoint_database_url)
    await _action_in_flight(sessionmaker, owner_id, mail, status)
    (resume,) = await _jobs(sessionmaker, "resume_workflow")
    await _exhaust(sessionmaker, resume.id)

    assert await _claim_exhausting(sessionmaker) is None

    run = await _run(sessionmaker)
    assert (run.status, run.error_code) == ("failed", "outcome_unknown")
    assert await _run_failed_payloads(sessionmaker) == [
        {"status": "failed", "error_code": "outcome_unknown"}
    ]


@pytest.mark.parametrize("failure", ["malformed_workflow_id", "db_error"])
async def test_failing_exhausted_hook_does_not_wedge_claims(
    sessionmaker: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    if failure == "db_error":

        async def broken_outcome(session: AsyncSession, *args: Any) -> None:
            await session.execute(text("SELECT 1 / 0"))  # aborts the statement's savepoint

        monkeypatch.setattr(worker, "_set_run_outcome", broken_outcome)
        workflow_id = str(uuid.uuid4())
    else:
        workflow_id = "not-a-uuid"
    async with sessionmaker() as s, s.begin():
        broken = await enqueue(s, "resume_workflow", "broken", {"workflow_id": workflow_id})
        other = await enqueue(s, "start_workflow", "other-job", {"n": 1})
    await _exhaust(sessionmaker, broken.id)

    claimed = await _claim_exhausting(sessionmaker)

    assert claimed is not None and claimed.id == other.id
    async with sessionmaker() as s:
        job = await s.get_one(Job, broken.id)
    assert (job.state, job.error, job.lease_owner) == ("failed", "max_attempts", None)
    assert any(
        r.levelname == "ERROR" and "could not fail the run" in r.getMessage()
        for r in caplog.records
    )
    assert await _claim_exhausting(sessionmaker) is None  # the broken job stays failed


class HangOnPlan:
    """Scripted model that hangs on its first planning call (the worker dies there)."""

    def __init__(self, scripts: dict[str, Any]) -> None:
        self.inner = ScriptedLLMClient(scripts)
        self.planning = asyncio.Event()

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        if call.purpose == "plan":
            self.planning.set()
            await asyncio.Event().wait()
        return await self.inner.structured(call)


async def test_start_job_continues_after_crash(
    sessionmaker: Session, make_deps: DepsBuilder, worker_settings: Any
) -> None:
    url = worker_settings.checkpoint_database_url
    await _ingest(sessionmaker)
    llm = HangOnPlan(BEFORE_APPROVAL)
    async with open_graph(url) as doomed:
        task = asyncio.create_task(_work(doomed, make_deps(llm=llm), "w1"))
        await asyncio.wait_for(llm.planning.wait(), 10)
        task.cancel()  # killed after triage, during the first planning call
        with contextlib.suppress(asyncio.CancelledError):
            await task
    (start,) = await _jobs(sessionmaker, "start_workflow")
    assert (start.state, start.lease_owner) == ("running", "w1")
    async with sessionmaker() as s, s.begin():
        await s.execute(update(Job).where(Job.id == start.id).values(lease_until=func.now()))

    # No triage output left: a fresh start (instead of a continuation) would fail.
    rest: dict[str, Any] = {MID: {k: v for k, v in BEFORE_APPROVAL[MID].items() if k != "triage"}}
    async with open_graph(url) as restarted:
        spy = GraphSpy(restarted)
        assert await _work(cast(Graph, spy), make_deps(rest), "w2")

    assert spy.inputs == [None]  # ainvoke(None): continue from the last checkpoint
    (start,) = await _jobs(sessionmaker, "start_workflow")
    assert (start.state, start.attempts) == ("done", 2)
    assert (await _run(sessionmaker)).status == "waiting_review"
    await _decision(sessionmaker)  # the approval interrupt was projected onto its item
