"""Read projections for the API: the review queue, workflow lists and the workflow detail.

These only read. Every HTML body passes through `sanitize_html` before it leaves the
server, and nothing here returns secrets: actions expose their status, provider reference
and fixed error code, never the stored provider result.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Select, and_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.retry import is_retryable
from app.api.pagination import Page, decode_cursor, encode_cursor
from app.db.models import (
    Action,
    Approval,
    EmailMessage,
    Proposal,
    ProposalRevision,
    ReviewItem,
    WorkflowEvent,
    WorkflowRun,
)
from app.mail.content import sanitize_html

EVENT_LIMIT = 100


class ReviewItemOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    workflow_id: uuid.UUID
    proposal_id: uuid.UUID | None
    interrupt_id: str | None
    sender: str
    subject: str
    received_at: datetime
    classification: str | None
    reason: str | None
    created_at: datetime


class WorkflowSummaryOut(BaseModel):
    id: uuid.UUID
    status: str
    error_code: str | None
    retryable: bool  # the UI offers Retry (see app/actions/retry.py)
    classification: str | None
    reason: str | None
    sender: str
    subject: str
    received_at: datetime
    created_at: datetime
    updated_at: datetime


class RunOut(BaseModel):
    id: uuid.UUID
    status: str
    error_code: str | None
    retryable: bool
    classification: str | None
    reason: str | None
    step_count: int
    graph_version: str
    provider_thread_id: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: uuid.UUID
    provider_message_id: str
    sender: str
    to: list[str]
    cc: list[str]
    subject: str
    body_text: str
    body_html: str | None  # sanitized
    received_at: datetime


class ReviewItemDetailOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    interrupt_id: str | None
    proposal_id: uuid.UUID | None
    payload: dict[str, Any]
    created_at: datetime
    resolved_at: datetime | None


class ProposalOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    revision: int
    payload: dict[str, Any]
    payload_hash: str
    author: str
    created_at: datetime


class ActionOut(BaseModel):
    id: uuid.UUID
    proposal_id: uuid.UUID
    kind: str
    status: str
    provider_ref: str | None
    error_code: str | None
    updated_at: datetime


class EventOut(BaseModel):
    id: int
    workflow_id: uuid.UUID
    type: str
    created_at: datetime
    payload: dict[str, Any]

    @classmethod
    def of(cls, row: WorkflowEvent) -> "EventOut":
        return cls(
            id=row.id,
            workflow_id=row.workflow_id,
            type=row.event_type,
            created_at=row.created_at,
            payload=row.payload,
        )


class WorkflowOut(BaseModel):
    run: RunOut
    thread: list[MessageOut]
    review_items: list[ReviewItemDetailOut]
    proposals: list[ProposalOut]
    actions: list[ActionOut]
    events: list[EventOut]  # the latest EVENT_LIMIT, oldest first


def _page_after[*Ts](
    stmt: Select[*Ts], created_at: Any, row_id: Any, cursor: str | None
) -> Select[*Ts]:
    """Newest first; with a cursor, only rows strictly older than its key."""
    if cursor is not None:
        at, after_id = decode_cursor(cursor)
        stmt = stmt.where(tuple_(created_at, row_id) < tuple_(at, after_id))
    return stmt.order_by(created_at.desc(), row_id.desc())


async def list_reviews(
    session: AsyncSession, *, status: str, kind: str | None, cursor: str | None, limit: int
) -> Page[ReviewItemOut]:
    stmt = (
        select(ReviewItem, WorkflowRun, EmailMessage)
        .join(WorkflowRun, WorkflowRun.id == ReviewItem.workflow_id)
        .join(EmailMessage, EmailMessage.id == WorkflowRun.message_id)
        .where(ReviewItem.status == status)
    )
    if kind is not None:
        stmt = stmt.where(ReviewItem.kind == kind)
    stmt = _page_after(stmt, ReviewItem.created_at, ReviewItem.id, cursor).limit(limit + 1)
    rows = (await session.execute(stmt)).all()
    items = [
        ReviewItemOut(
            id=item.id,
            kind=item.kind,
            status=item.status,
            workflow_id=item.workflow_id,
            proposal_id=item.proposal_id,
            interrupt_id=item.interrupt_id,
            sender=message.sender,
            subject=message.subject,
            received_at=message.received_at,
            # A notification item also carries triage's outcome in its own payload.
            classification=run.classification or item.payload.get("classification"),
            reason=run.triage_reason or item.payload.get("reason"),
            created_at=item.created_at,
        )
        for item, run, message in rows[:limit]
    ]
    last = items[-1] if len(rows) > limit else None
    return Page(items=items, next_cursor=encode_cursor(last.created_at, last.id) if last else None)


async def list_workflows(
    session: AsyncSession, *, statuses: list[str], cursor: str | None, limit: int
) -> Page[WorkflowSummaryOut]:
    stmt = select(WorkflowRun, EmailMessage).join(
        EmailMessage, EmailMessage.id == WorkflowRun.message_id
    )
    if statuses:
        stmt = stmt.where(WorkflowRun.status.in_(statuses))
    stmt = _page_after(stmt, WorkflowRun.created_at, WorkflowRun.id, cursor).limit(limit + 1)
    rows = (await session.execute(stmt)).all()
    items = [
        WorkflowSummaryOut(
            id=run.id,
            status=run.status,
            error_code=run.error_code,
            retryable=is_retryable(run.status, run.error_code),
            classification=run.classification,
            reason=run.triage_reason,
            sender=message.sender,
            subject=message.subject,
            received_at=message.received_at,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )
        for run, message in rows[:limit]
    ]
    last = items[-1] if len(rows) > limit else None
    return Page(items=items, next_cursor=encode_cursor(last.created_at, last.id) if last else None)


async def _thread(session: AsyncSession, run: WorkflowRun) -> list[MessageOut]:
    thread_id = await session.scalar(
        select(EmailMessage.thread_id).where(EmailMessage.id == run.message_id)
    )
    messages = (
        await session.execute(
            select(EmailMessage)
            .where(EmailMessage.thread_id == thread_id)
            .order_by(EmailMessage.received_at, EmailMessage.provider_message_id)
        )
    ).scalars()
    return [
        MessageOut(
            id=m.id,
            provider_message_id=m.provider_message_id,
            sender=m.sender,
            to=list(m.to_addrs),
            cc=list(m.cc_addrs),
            subject=m.subject,
            body_text=m.body_text,
            body_html=sanitize_html(m.body_html) if m.body_html else None,
            received_at=m.received_at,
        )
        for m in messages
    ]


async def _proposals(session: AsyncSession, workflow_id: uuid.UUID) -> list[ProposalOut]:
    rows = await session.execute(
        select(Proposal, ProposalRevision)
        .join(
            ProposalRevision,
            and_(
                ProposalRevision.proposal_id == Proposal.id,
                ProposalRevision.revision == Proposal.active_revision,
            ),
        )
        .where(Proposal.workflow_id == workflow_id)
        .order_by(Proposal.created_at, Proposal.id)
    )
    return [
        ProposalOut(
            id=p.id,
            kind=p.kind,
            status=p.status,
            revision=r.revision,
            payload=r.payload,
            payload_hash=r.payload_hash,
            author=r.author,
            created_at=p.created_at,
        )
        for p, r in rows
    ]


async def _actions(session: AsyncSession, workflow_id: uuid.UUID) -> list[ActionOut]:
    rows = await session.execute(
        select(Proposal.id, Action)
        .join(ProposalRevision, ProposalRevision.proposal_id == Proposal.id)
        .join(Approval, Approval.proposal_revision_id == ProposalRevision.id)
        .join(Action, Action.approval_id == Approval.id)
        .where(Proposal.workflow_id == workflow_id)
        .order_by(Action.updated_at, Action.id)
    )
    return [
        ActionOut(
            id=a.id,
            proposal_id=proposal_id,
            kind=a.kind,
            status=a.status,
            provider_ref=a.provider_ref,
            error_code=a.error,  # the executor stores fixed codes only
            updated_at=a.updated_at,
        )
        for proposal_id, a in rows
    ]


async def workflow_detail(session: AsyncSession, workflow_id: uuid.UUID) -> WorkflowOut | None:
    run = await session.get(WorkflowRun, workflow_id)
    if run is None:
        return None
    items = (
        await session.execute(
            select(ReviewItem)
            .where(ReviewItem.workflow_id == workflow_id)
            .order_by(ReviewItem.created_at, ReviewItem.id)
        )
    ).scalars()
    events = (
        await session.execute(
            select(WorkflowEvent)
            .where(WorkflowEvent.workflow_id == workflow_id)
            .order_by(WorkflowEvent.id.desc())
            .limit(EVENT_LIMIT)
        )
    ).scalars()
    return WorkflowOut(
        run=RunOut(
            id=run.id,
            status=run.status,
            error_code=run.error_code,
            retryable=is_retryable(run.status, run.error_code),
            classification=run.classification,
            reason=run.triage_reason,
            step_count=run.step_count,
            graph_version=run.graph_version,
            provider_thread_id=run.provider_thread_id,
            created_at=run.created_at,
            updated_at=run.updated_at,
        ),
        thread=await _thread(session, run),
        review_items=[
            ReviewItemDetailOut(
                id=i.id,
                kind=i.kind,
                status=i.status,
                interrupt_id=i.interrupt_id,
                proposal_id=i.proposal_id,
                payload=i.payload,
                created_at=i.created_at,
                resolved_at=i.resolved_at,
            )
            for i in items
        ],
        proposals=await _proposals(session, workflow_id),
        actions=await _actions(session, workflow_id),
        events=[EventOut.of(e) for e in reversed(list(events))],
    )
