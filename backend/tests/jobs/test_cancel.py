"""Cancel really stops future work: queued answers are not delivered to a cancelled run, and a
graph that is still running when the owner cancels opens no proposal or review item.

Real worker, real `AsyncPostgresSaver`, real queue; only the model is scripted.
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.reviews import answer_question, cancel_workflow
from app.db.models import Job, Proposal, ReviewItem, WorkflowRun
from app.graphs.parent import GRAPH_VERSION
from app.ingestion.ingest import ingest_message
from app.jobs.worker import run_once
from app.llm.client import LLMCall, LLMResult, T
from app.llm.scripted import ScriptedLLMClient
from app.providers.fixtures import load_fixture
from tests.jobs.conftest import DepsBuilder, open_graph

Session = async_sessionmaker[AsyncSession]
LEASE = timedelta(seconds=60)
FIXTURE = "sched-tomorrow-review"
MID = "sched-tomorrow-review-m1"
SENDER = "sara.ng@work.example.com"
FRIDAY = "2026-11-20"

TRIAGE_RESPOND = {
    "classification": "respond",
    "urgency": "normal",
    "reason": "Sara asks for a review slot.",
    "evidence_refs": [f"message:{MID}"],
    "needs_clarification": False,
}
FREEBUSY = {
    "action": "get_freebusy",
    "reason": "Check tomorrow.",
    "freebusy": {"start_date": FRIDAY, "end_date": FRIDAY},
}
DRAFT = {
    "to": [SENDER],
    "cc": [],
    "body_text": "Friday at 13:00 UTC works for me.",
    "evidence_refs": [f"message:{MID}", f"availability:{FRIDAY}:{FRIDAY}"],
}


class RecordingLLM:
    """The scripted model, recording each call's purpose; `before(purpose)` runs first."""

    def __init__(
        self,
        scripts: dict[str, Any],
        before: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._inner = ScriptedLLMClient(scripts)
        self._before = before
        self.purposes: list[str] = []

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        self.purposes.append(call.purpose)
        if self._before is not None:
            await self._before(call.purpose)
        return await self._inner.structured(call)


async def _ingest(sessionmaker: Session) -> None:
    async with sessionmaker() as s:
        for message in load_fixture(FIXTURE).thread:
            assert (await ingest_message(s, message, ["owner@example.com"])).status == "queued"
        await s.commit()


async def _run(sessionmaker: Session) -> WorkflowRun:
    async with sessionmaker() as s:
        return (await s.execute(select(WorkflowRun))).scalar_one()


async def _count(sessionmaker: Session, model: Any, *where: Any) -> int:
    async with sessionmaker() as s:
        return int(await s.scalar(select(func.count()).select_from(model).where(*where)) or 0)


async def test_cancel_after_answer_does_not_revive(
    sessionmaker: Session, make_deps: DepsBuilder, checkpoints: str
) -> None:
    await _ingest(sessionmaker)
    scripts = {
        MID: {
            "triage": [TRIAGE_RESPOND],
            "plan": [
                {"action": "ask_owner", "reason": "Unclear.", "question": "Which project?"},
                # Only reached if the answer were delivered after the cancel:
                FREEBUSY,
                {"action": "draft_reply", "reason": "Offer a slot."},
            ],
            "draft_reply": [DRAFT],
        }
    }
    llm = RecordingLLM(scripts)
    async with open_graph(checkpoints) as graph:
        assert await run_once(graph, make_deps(llm=llm), worker_id="w1", lease=LEASE)
    async with sessionmaker() as s:
        item = (await s.execute(select(ReviewItem))).scalar_one()
    assert item.kind == "question" and item.interrupt_id is not None
    run = await _run(sessionmaker)

    # The owner answers, then cancels before the worker picks up the resume job.
    async with sessionmaker() as s:
        await answer_question(
            s, item.id, answer="Project X", request_id="a-1", interrupt_id=item.interrupt_id
        )
    async with sessionmaker() as s:
        outcome = await cancel_workflow(s, run.id, request_id="c-1")
    assert outcome.cancelled_review_items == [] and outcome.job_id is None
    calls_before = list(llm.purposes)

    async with open_graph(checkpoints) as graph:
        assert await run_once(graph, make_deps(llm=llm), worker_id="w2", lease=LEASE)

    assert llm.purposes == calls_before  # the graph never ran again
    assert (await _run(sessionmaker)).status == "cancelled"
    assert await _count(sessionmaker, Proposal) == 0
    assert await _count(sessionmaker, ReviewItem) == 1  # only the answered question
    async with sessionmaker() as s:
        resume = (await s.execute(select(Job).where(Job.kind == "resume_workflow"))).scalar_one()
    assert resume.state == "done"


async def test_cancel_during_drafting_does_not_create_proposal(
    sessionmaker: Session, make_deps: DepsBuilder, checkpoints: str
) -> None:
    await _ingest(sessionmaker)

    async def cancel_when_drafting(purpose: str) -> None:
        if purpose == "draft_reply":  # the run is `running`, mid-graph
            run = await _run(sessionmaker)
            async with sessionmaker() as s:
                await cancel_workflow(s, run.id, request_id="c-1")

    scripts = {
        MID: {
            "triage": [TRIAGE_RESPOND],
            "plan": [FREEBUSY, {"action": "draft_reply", "reason": "Offer a slot."}],
            "draft_reply": [DRAFT],
        }
    }
    llm = RecordingLLM(scripts, before=cancel_when_drafting)
    async with open_graph(checkpoints) as graph:
        assert await run_once(graph, make_deps(llm=llm), worker_id="w1", lease=LEASE)
        snap = await graph.aget_state(
            {"configurable": {"thread_id": f"mock-account:{MID}:{GRAPH_VERSION}"}}, subgraphs=True
        )

    assert "draft_reply" in llm.purposes
    assert await _count(sessionmaker, Proposal) == 0
    assert await _count(sessionmaker, ReviewItem) == 0
    assert (await _run(sessionmaker)).status == "cancelled"
    assert not snap.next  # the graph ended instead of pausing at an approval interrupt


async def test_cancel_during_triage_opens_no_notification(
    sessionmaker: Session, make_deps: DepsBuilder, checkpoints: str
) -> None:
    await _ingest(sessionmaker)

    async def cancel_first(purpose: str) -> None:
        run = await _run(sessionmaker)
        async with sessionmaker() as s:
            await cancel_workflow(s, run.id, request_id=f"c-{uuid.uuid4().hex}")

    notify = {**TRIAGE_RESPOND, "classification": "notify"}
    llm = RecordingLLM({MID: {"triage": [notify]}}, before=cancel_first)
    async with open_graph(checkpoints) as graph:
        assert await run_once(graph, make_deps(llm=llm), worker_id="w1", lease=LEASE)

    assert await _count(sessionmaker, ReviewItem) == 0
    assert (await _run(sessionmaker)).status == "cancelled"
