"""Owner answers to questions and notifications, and cancelling a workflow.

Each function runs one transaction on the session it is given and commits it (or rolls it
back and raises), like `decide_proposal`: give it a dedicated session.

Answering a review item: the item row is locked `FOR UPDATE`, it must still be `pending`
and carry the interrupt id the owner saw (otherwise `DecisionConflict`, HTTP 409), it is
resolved, and the `resume_workflow` job is enqueued in the same transaction. The request id
is stored in the item's payload as `decision_request_id`, so a repeated click with the same
id returns the original outcome instead of a 409.

Cancelling: locks the run's pending review items, then its proposals (the order ingestion
and `decide_proposal` use), then the run. Pending items become `cancelled`; proposals not
yet executed become `cancelled_before_execution`; a proposal with an Action row (a write
that started or finished) is left alone and reported. The run becomes `cancelled` unless it
is already final, a `run_cancelled` event is written, and the pending interrupt (if any) is
resumed with `{"decision": "cancelled"}` so the graph ends cleanly.
"""

import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.decisions import DecisionConflict
from app.db.models import (
    Action,
    Approval,
    AuditEvent,
    Proposal,
    ProposalRevision,
    ReviewItem,
    WorkflowEvent,
    WorkflowRun,
)
from app.graphs.events import emit
from app.graphs.nodes.finalize import FINAL_STATUSES
from app.jobs.queue import enqueue

_UNEXECUTED = ("awaiting_approval", "approved")


class ReviewNotFound(LookupError):
    """No review item (of the expected kind) or workflow with this id: HTTP 404."""


class ReviewOutcome(BaseModel):
    status: Literal["accepted"] = "accepted"
    review_item_id: uuid.UUID
    job_id: uuid.UUID


class CompletedAction(BaseModel):
    proposal_id: uuid.UUID
    action_id: uuid.UUID
    kind: str
    status: str
    provider_ref: str | None


class CancelOutcome(BaseModel):
    status: Literal["accepted"] = "accepted"
    workflow_id: uuid.UUID
    run_status: str
    cancelled_review_items: list[uuid.UUID]
    cancelled_proposals: list[uuid.UUID]
    # Proposals whose write already started or finished: not cancelled, reported here.
    completed_actions: list[CompletedAction]
    job_id: uuid.UUID | None


async def _provider_thread_id(session: AsyncSession, workflow_id: uuid.UUID) -> str | None:
    return await session.scalar(
        select(WorkflowRun.provider_thread_id).where(WorkflowRun.id == workflow_id)
    )


def _resume_dedup_key(item: ReviewItem) -> str:
    # The same key `decide_proposal` and ingestion use, so one interrupt gets one resume.
    return f"resume:{item.interrupt_id}" if item.interrupt_id else f"resume:item:{item.id}"


async def _enqueue_resume(
    session: AsyncSession, item: ReviewItem, resume: dict[str, Any]
) -> uuid.UUID:
    job = await enqueue(
        session,
        "resume_workflow",
        _resume_dedup_key(item),
        {
            "workflow_id": str(item.workflow_id),
            "interrupt_id": item.interrupt_id,
            "review_item_id": str(item.id),
            "resume": resume,
        },
        serial_key=await _provider_thread_id(session, item.workflow_id),
    )
    return job.id


async def _resolve_item(
    session: AsyncSession,
    item_id: uuid.UUID,
    kind: Literal["question", "notification"],
    *,
    request_id: str,
    interrupt_id: str,
    resume: dict[str, Any],
) -> ReviewOutcome:
    item = (
        await session.execute(
            select(ReviewItem)
            .where(ReviewItem.id == item_id, ReviewItem.kind == kind)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if item is None:
        raise ReviewNotFound(str(item_id))
    if item.payload.get("decision_request_id") == request_id:
        job_id = await _enqueue_resume(session, item, item.payload["decision"])  # existing row
        return ReviewOutcome(review_item_id=item.id, job_id=job_id)
    if item.status != "pending":
        raise DecisionConflict("not_pending")
    if item.interrupt_id is None or item.interrupt_id != interrupt_id:
        raise DecisionConflict("wrong_interrupt")

    item.status = "resolved"
    item.resolved_at = datetime.now(UTC)
    item.payload = {**item.payload, "decision_request_id": request_id, "decision": resume}
    job_id = await _enqueue_resume(session, item, resume)
    session.add(
        AuditEvent(
            workflow_id=item.workflow_id,
            event_type="decision_recorded",
            detail={
                "review_item_id": str(item.id),
                "kind": kind,
                "decision": resume["decision"],
            },
        )
    )
    await session.flush()
    return ReviewOutcome(review_item_id=item.id, job_id=job_id)


async def _committed[T](session: AsyncSession, work: Awaitable[T]) -> T:
    """Await `work`, then commit; roll back on any error."""
    try:
        result = await work
        await session.commit()
        return result
    except BaseException:
        await session.rollback()
        raise


async def answer_question(
    session: AsyncSession, item_id: uuid.UUID, *, answer: str, request_id: str, interrupt_id: str
) -> ReviewOutcome:
    """Resolve a pending question with the owner's answer and enqueue the resume; commits."""
    return await _committed(
        session,
        _resolve_item(
            session,
            item_id,
            "question",
            request_id=request_id,
            interrupt_id=interrupt_id,
            resume={"decision": "answer", "answer": answer},
        ),
    )


async def decide_notification(
    session: AsyncSession,
    item_id: uuid.UUID,
    *,
    decision: Literal["dismiss", "respond"],
    request_id: str,
    interrupt_id: str,
) -> ReviewOutcome:
    """Dismiss a notification or ask for a response, and enqueue the resume; commits."""
    return await _committed(
        session,
        _resolve_item(
            session,
            item_id,
            "notification",
            request_id=request_id,
            interrupt_id=interrupt_id,
            resume={"decision": decision},
        ),
    )


async def _stored_cancel(
    session: AsyncSession, workflow_id: uuid.UUID, request_id: str
) -> CancelOutcome | None:
    event = (
        await session.execute(
            select(WorkflowEvent)
            .where(
                WorkflowEvent.workflow_id == workflow_id,
                WorkflowEvent.event_type == "run_cancelled",
                WorkflowEvent.payload["request_id"].astext == request_id,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    return None if event is None else CancelOutcome.model_validate(event.payload["outcome"])


async def _run_actions(session: AsyncSession, workflow_id: uuid.UUID) -> list[CompletedAction]:
    """Every Action row (a write that started or finished) of the run's proposals."""
    rows = await session.execute(
        select(Proposal.id, Action)
        .join(ProposalRevision, ProposalRevision.proposal_id == Proposal.id)
        .join(Approval, Approval.proposal_revision_id == ProposalRevision.id)
        .join(Action, Action.approval_id == Approval.id)
        .where(Proposal.workflow_id == workflow_id)
        .order_by(Action.updated_at, Action.id)
    )
    return [
        CompletedAction(
            proposal_id=proposal_id,
            action_id=action.id,
            kind=action.kind,
            status=action.status,
            provider_ref=action.provider_ref,
        )
        for proposal_id, action in rows
    ]


async def _cancel(session: AsyncSession, workflow_id: uuid.UUID, request_id: str) -> CancelOutcome:
    if await session.get(WorkflowRun, workflow_id) is None:
        raise ReviewNotFound(str(workflow_id))
    stored = await _stored_cancel(session, workflow_id, request_id)
    if stored is not None:
        return stored

    items = list(
        (
            await session.execute(
                select(ReviewItem)
                .where(ReviewItem.workflow_id == workflow_id, ReviewItem.status == "pending")
                .order_by(ReviewItem.created_at, ReviewItem.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).scalars()
    )
    proposals = list(
        (
            await session.execute(
                select(Proposal)
                .where(Proposal.workflow_id == workflow_id, Proposal.status.in_(_UNEXECUTED))
                .order_by(Proposal.created_at, Proposal.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).scalars()
    )
    run = await session.get_one(
        WorkflowRun, workflow_id, with_for_update=True, populate_existing=True
    )
    # Read after the proposal locks: an executor claim holds the proposal FOR SHARE while it
    # inserts its Action row, so any row it commits is visible here.
    completed = await _run_actions(session, workflow_id)
    started_ids = {a.proposal_id for a in completed}
    cancelled_proposals = [p for p in proposals if p.id not in started_ids]
    if run.status in FINAL_STATUSES and not items and not cancelled_proposals:
        raise DecisionConflict(
            "already_cancelled" if run.status == "cancelled" else "already_final"
        )

    now = datetime.now(UTC)
    for item in items:
        item.status = "cancelled"
        item.resolved_at = now
    for proposal in cancelled_proposals:
        proposal.status = "cancelled_before_execution"
    if run.status not in FINAL_STATUSES:
        run.status = "cancelled"
        run.updated_at = now

    # Resume the interrupt the graph is paused on, so it reaches its normal end.
    interrupted = next((i for i in items if i.interrupt_id is not None), None)
    job_id = (
        await _enqueue_resume(session, interrupted, {"decision": "cancelled"})
        if interrupted is not None
        else None
    )
    outcome = CancelOutcome(
        workflow_id=workflow_id,
        run_status=run.status,
        cancelled_review_items=[i.id for i in items],
        cancelled_proposals=[p.id for p in cancelled_proposals],
        completed_actions=completed,
        job_id=job_id,
    )
    await emit(
        session,
        workflow_id,
        "run_cancelled",
        {"request_id": request_id, "outcome": outcome.model_dump(mode="json")},
    )
    session.add(
        AuditEvent(
            workflow_id=workflow_id,
            event_type="run_cancelled",
            detail=outcome.model_dump(mode="json"),
        )
    )
    await session.flush()
    return outcome


async def cancel_workflow(
    session: AsyncSession, workflow_id: uuid.UUID, *, request_id: str
) -> CancelOutcome:
    """Stop future work on a run and report effects that already happened; commits."""
    return await _committed(session, _cancel(session, workflow_id, request_id))
