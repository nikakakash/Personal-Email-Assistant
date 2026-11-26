"""Helpers for compiled-graph tests: a seeded run, runtime deps, scripted outputs, decisions.

Graphs run with `InMemorySaver` (no Postgres checkpointer at this level); the app tables, the
mock providers and the approval rows are real.
"""

import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.decisions import DecisionIn, decide_proposal
from app.actions.proposals import active_revision
from app.config import Settings
from app.db.models import EmailMessage, Job, Owner, Proposal, ReviewItem, WorkflowRun
from app.graphs.deps import GraphDeps
from app.graphs.parent import build_parent_graph
from app.graphs.state import EmailState
from app.llm.client import LLMClient, Purpose
from app.llm.scripted import ScriptedLLMClient
from app.providers.interfaces import MailProvider
from app.providers.mock_calendar import MockCalendarProvider
from app.providers.mock_mail import MockFaults, MockMailProvider

# Monday 23 Nov 2026, 08:00 UTC. The owner has no stored preferences, so the zone is UTC.
NOW = datetime(2026, 11, 23, 8, 0, tzinfo=UTC)
OWNER = "owner@example.com"
TUESDAY = "2026-11-24"
AVAILABILITY_KEY = f"availability:{TUESDAY}:{TUESDAY}"

Graph = CompiledStateGraph[EmailState, GraphDeps, EmailState, EmailState]


@dataclass(frozen=True)
class SeededRun:
    workflow_id: str
    provider_message_id: str
    provider_thread_id: str
    thread_row_id: uuid.UUID

    @property
    def input(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "account_id": "mock-account",
            "provider_message_id": self.provider_message_id,
            "provider_thread_id": self.provider_thread_id,
        }

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": f"mock-account:{self.provider_message_id}:wf1"}}


@pytest.fixture
async def seeded(
    session: AsyncSession, workflow_factory: Callable[[], Awaitable[WorkflowRun]]
) -> SeededRun:
    """A WorkflowRun whose one inbound message asks the owner for a meeting."""
    run = await workflow_factory()
    message = await session.get_one(EmailMessage, run.message_id)
    message.rfc_message_id = f"<{message.provider_message_id}@example.com>"
    message.to_addrs = [OWNER]
    message.subject = "Meeting next week?"
    message.body_text = "Could we meet on Tuesday morning?"
    message.received_at = NOW - timedelta(hours=1)
    seeded = SeededRun(
        workflow_id=str(run.id),
        provider_message_id=message.provider_message_id,
        provider_thread_id=run.provider_thread_id,
        thread_row_id=message.thread_id,
    )
    await session.commit()
    return seeded


@pytest.fixture
def faults() -> MockFaults:
    return MockFaults()


@pytest.fixture
def mail(sessionmaker: async_sessionmaker[AsyncSession], faults: MockFaults) -> MockMailProvider:
    return MockMailProvider(sessionmaker, faults)


@pytest.fixture
def calendar(
    sessionmaker: async_sessionmaker[AsyncSession], faults: MockFaults
) -> MockCalendarProvider:
    busy: dict[date, list[tuple[time, time]]] = {}
    return MockCalendarProvider(sessionmaker, busy, "UTC", faults)


DepsFactory = Callable[..., GraphDeps]


@pytest.fixture
def make_deps(
    base_env: None,
    sessionmaker: async_sessionmaker[AsyncSession],
    mail: MockMailProvider,
    calendar: MockCalendarProvider,
) -> DepsFactory:
    def make(
        scripts: Mapping[str, Mapping[Purpose, Sequence[dict[str, Any]]]] | None = None,
        *,
        llm: LLMClient | None = None,
        outbound: bool = True,
        mail_provider: MailProvider | None = None,
    ) -> GraphDeps:
        settings = Settings(_env_file=None, outbound_execution_enabled=outbound)  # type: ignore[call-arg]
        return GraphDeps(
            llm=llm or ScriptedLLMClient(scripts or {}),
            mail=mail_provider or mail,
            calendar=calendar,
            sessionmaker=sessionmaker,
            settings=settings,
            now=lambda: NOW,
        )

    return make


@pytest.fixture
def graph() -> Graph:
    return build_parent_graph(InMemorySaver())


async def run_graph(
    graph: Graph, payload: dict[str, Any] | Command[Any], seeded: SeededRun, deps: GraphDeps
) -> dict[str, Any]:
    """One invocation, the way the worker runs it."""
    return await graph.ainvoke(payload, seeded.config, context=deps, durability="sync")  # type: ignore[arg-type]


async def subgraph_values(graph: Graph, seeded: SeededRun) -> dict[str, Any]:
    """State inside the paused response subgraph.

    While the run is paused in the subgraph, the parent's state (what `ainvoke` returns) does
    not yet include the subgraph's updates: they flow back only when the subgraph finishes.
    """
    snapshot = await graph.aget_state(seeded.config, subgraphs=True)  # type: ignore[arg-type]
    for task in snapshot.tasks:
        if task.name == "response" and isinstance(task.state, StateSnapshot):
            return dict(task.state.values)
    raise AssertionError("the run is not paused inside the response subgraph")


async def resume(
    graph: Graph, value: dict[str, Any], seeded: SeededRun, deps: GraphDeps
) -> dict[str, Any]:
    return await run_graph(graph, Command(resume=value), seeded, deps)


# ---------------------------------------------------------------- scripted model outputs


def triage_out(classification: str, mid: str, **extra: Any) -> dict[str, Any]:
    return {
        "classification": classification,
        "urgency": "normal",
        "reason": f"Classified as {classification}.",
        "evidence_refs": [f"message:{mid}"],
        "needs_clarification": False,
        **extra,
    }


def plan_out(action: str, **extra: Any) -> dict[str, Any]:
    return {"action": action, "reason": f"Next: {action}.", **extra}


def freebusy_plan() -> dict[str, Any]:
    return plan_out("get_freebusy", freebusy={"start_date": TUESDAY, "end_date": TUESDAY})


def reply_out(mid: str, body: str = "Tuesday at 09:00 UTC works for me.") -> dict[str, Any]:
    return {
        "to": ["alice@example.com"],
        "cc": [],
        "body_text": body,
        "evidence_refs": [f"message:{mid}", AVAILABILITY_KEY],
    }


def meeting_out() -> dict[str, Any]:
    return {
        "title": "Meeting with Alice",
        "start_local": f"{TUESDAY}T09:00:00",
        "end_local": f"{TUESDAY}T09:30:00",
        "time_zone": "UTC",
        "attendees": ["alice@example.com"],
        "location": None,
        "send_updates": "none",
        "evidence_refs": [AVAILABILITY_KEY],
    }


# ---------------------------------------------------------------- decisions and DB reads


async def decide(
    sessionmaker: async_sessionmaker[AsyncSession],
    owner: Owner,
    result: dict[str, Any],
    mail: MockMailProvider,
    decision: str = "approve",
    **fields: Any,
) -> dict[str, Any]:
    """Record a real decision on the pending proposal and return its resume value.

    The interrupt id is copied onto the review item first: in production Task 11's projection
    does that after each run.
    """
    interrupt = result["__interrupt__"][0]
    async with sessionmaker() as s:
        item = await s.get_one(ReviewItem, uuid.UUID(interrupt.value["review_item_id"]))
        item.interrupt_id = interrupt.id
        assert item.proposal_id is not None
        proposal_id = item.proposal_id
        revision = await active_revision(s, await s.get_one(Proposal, proposal_id))
        body = DecisionIn(
            decision=decision,  # type: ignore[arg-type]
            proposal_revision=revision.revision,
            payload_hash=revision.payload_hash,
            interrupt_id=interrupt.id,
            request_id=uuid.uuid4().hex,
            **fields,
        )
        await s.commit()
    async with sessionmaker() as s:
        outcome = await decide_proposal(s, owner.id, proposal_id, body, mail=mail)
        assert outcome.job_id is not None
        job = await s.get_one(Job, outcome.job_id)
        return dict(job.payload["resume"])


async def count(sessionmaker: async_sessionmaker[AsyncSession], model: Any, *where: Any) -> int:
    async with sessionmaker() as s:
        return int(await s.scalar(select(func.count()).select_from(model).where(*where)) or 0)


async def run_row(sessionmaker: async_sessionmaker[AsyncSession], seeded: SeededRun) -> WorkflowRun:
    async with sessionmaker() as s:
        return await s.get_one(WorkflowRun, uuid.UUID(seeded.workflow_id))
