"""Owner decisions on a proposal: approve, edit-and-approve, reject or feedback.

`decide_proposal` binds a decision to the exact revision, payload hash and interrupt the owner
reviewed (spec §9). The approval row, the proposal and review-item transitions, the resume job
and the audit event commit in one transaction, so a crash leaves either all of them or none.

Concurrency: the review item row is locked first, then the proposal row, the same order
ingestion uses when it marks proposals stale, so the two never deadlock. A second accepting
decision on the same revision finds the item no longer pending (or, failing that, trips the
partial unique index) and gets `DecisionConflict`. A repeated `request_id` returns the stored
outcome instead of deciding again.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.hashing import payload_hash
from app.db.models import (
    Approval,
    AuditEvent,
    Job,
    Proposal,
    ProposalRevision,
    ReviewItem,
    WorkflowRun,
)
from app.graphs.nodes.validate import Kind, validate_payload
from app.jobs.queue import enqueue
from app.mail.addresses import normalize_address
from app.providers.interfaces import MailProvider, ThreadSnapshot

Decision = Literal["approve", "edit_approve", "reject", "feedback"]
ACCEPTING: tuple[Decision, ...] = ("approve", "edit_approve")

# The only fields an owner edit may change. Everything else (threading headers, subject,
# calendar id, evidence) is copied from the reviewed revision, so it stays code-derived.
EDITABLE_FIELDS: dict[str, frozenset[str]] = {
    "send_reply": frozenset({"to", "cc", "body_text"}),
    "create_event": frozenset(
        {"title", "start", "end", "time_zone", "attendees", "location", "send_updates"}
    ),
}
_ADDRESS_FIELDS = ("to", "cc", "attendees")
_TIME_FIELDS = ("start", "end")
_NEW_STATUS: dict[Decision, str] = {
    "approve": "approved",
    "edit_approve": "approved",
    "reject": "rejected",
    "feedback": "superseded",
}


class DecisionConflict(Exception):  # noqa: N818 - name fixed by the contract
    """The decision no longer applies (stale revision, wrong hash or interrupt, …): HTTP 409."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class InvalidEdit(Exception):  # noqa: N818 - name fixed by the contract
    """The owner's edited payload is not acceptable: HTTP 422."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__(", ".join(errors))
        self.errors = errors


class ProposalNotFound(LookupError):
    """No proposal (or no review item for it) with this id: HTTP 404."""


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    proposal_revision: int
    payload_hash: str
    interrupt_id: str
    request_id: str
    edited_payload: dict[str, Any] | None = None
    feedback: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # An approval-only request never carries a replacement payload (spec §9).
        if (self.decision == "edit_approve") != (self.edited_payload is not None):
            raise ValueError("edited_payload is required for edit_approve and only for it")
        if self.decision == "feedback" and not (self.feedback and self.feedback.strip()):
            raise ValueError("feedback must not be empty")
        return self


class DecisionOutcome(BaseModel):
    status: Literal["accepted"]
    approval_id: uuid.UUID
    job_id: uuid.UUID
    revision: int


def _normalize_addresses(field: str, value: Any) -> Any:
    """Bare lowercase addresses, de-duplicated in order; non-lists are left to validation."""
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return value
    seen: dict[str, None] = {}
    for raw in value:
        try:
            seen.setdefault(normalize_address(raw), None)
        except ValueError as exc:
            raise InvalidEdit([f"invalid_address:{field}"]) from exc
    return list(seen)


def _normalize_time(value: Any) -> Any:
    """An aware ISO time becomes UTC ISO (as drafts store it); anything else is validated later."""
    if not isinstance(value, str):
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.astimezone(UTC).isoformat() if parsed.tzinfo is not None else value


def merge_edit(kind: str, base: dict[str, Any], edited: dict[str, Any]) -> dict[str, Any]:
    """Apply the owner's edits to `base`, normalizing addresses and times; InvalidEdit if bad."""
    allowed = EDITABLE_FIELDS[kind]
    rejected = sorted(k for k in edited if k not in allowed)
    if rejected:
        raise InvalidEdit([f"field_not_editable:{k[:60]}" for k in rejected])
    merged = dict(base)
    for key, value in edited.items():
        if key in _ADDRESS_FIELDS:
            value = _normalize_addresses(key, value)
        elif key in _TIME_FIELDS:
            value = _normalize_time(value)
        merged[key] = value
    return merged


async def _stored_outcome(
    session: AsyncSession, proposal_id: uuid.UUID, request_id: str
) -> DecisionOutcome | None:
    approval = (
        await session.execute(select(Approval).where(Approval.request_id == request_id))
    ).scalar_one_or_none()
    if approval is None:
        return None
    revision = await session.get_one(ProposalRevision, approval.proposal_revision_id)
    if revision.proposal_id != proposal_id:
        raise DecisionConflict("request_id_reused")
    job = (
        await session.execute(select(Job).where(Job.dedup_key == f"resume:{approval.interrupt_id}"))
    ).scalar_one()
    return DecisionOutcome(
        status="accepted", approval_id=approval.id, job_id=job.id, revision=revision.revision
    )


async def _prepare_edit(
    session: AsyncSession, proposal_id: uuid.UUID, body: DecisionIn, mail: MailProvider
) -> dict[str, Any]:
    """Build and validate the edited payload. Runs before any row lock; ends its read txn
    before the provider call so no transaction is open across it."""
    row = (
        await session.execute(
            select(Proposal.kind, ProposalRevision.payload, WorkflowRun.provider_thread_id)
            .join(ProposalRevision, ProposalRevision.proposal_id == Proposal.id)
            .join(WorkflowRun, WorkflowRun.id == Proposal.workflow_id)
            .where(Proposal.id == proposal_id, ProposalRevision.revision == body.proposal_revision)
        )
    ).one_or_none()
    await session.rollback()
    if row is None:
        raise DecisionConflict("stale_revision")
    kind, base, provider_thread_id = cast(tuple[str, dict[str, Any], str], tuple(row))
    merged = merge_edit(kind, base, body.edited_payload or {})
    try:
        thread: ThreadSnapshot = await mail.get_thread(provider_thread_id)
    except KeyError as exc:
        raise DecisionConflict("thread_unavailable") from exc
    errors = validate_payload(
        cast(Kind, kind),
        merged,
        thread=thread,
        findings=[],
        owner_identities=(),  # unused for owner-authored payloads: any recipient is allowed
        author="owner",
    )
    if errors:
        raise InvalidEdit(errors)
    return merged


async def _audit_decision(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    proposal: Proposal,
    approval: Approval,
    item: ReviewItem,
    revision: int,
) -> None:
    session.add(
        AuditEvent(
            workflow_id=workflow_id,
            event_type="decision_recorded",
            detail={
                "proposal_id": str(proposal.id),
                "proposal_kind": proposal.kind,
                "approval_id": str(approval.id),
                "decision": approval.decision,
                "revision": revision,
                "review_item_id": str(item.id),
                "proposal_status": proposal.status,
            },
        )
    )
    await session.flush()


async def decide_proposal(
    session: AsyncSession,
    owner_id: uuid.UUID,
    proposal_id: uuid.UUID,
    body: DecisionIn,
    *,
    mail: MailProvider,
) -> DecisionOutcome:
    """Record the owner's decision and enqueue the graph resume; commits on success.

    `session` must not hold uncommitted work: this function owns its transactions. Raises
    `DecisionConflict` (409), `InvalidEdit` (422) or `ProposalNotFound` (404). `mail` is only
    used by edit_approve, to load the thread the edited payload is validated against.
    """
    try:
        stored = await _stored_outcome(session, proposal_id, body.request_id)
        if stored is not None:
            await session.rollback()
            return stored
        edited = (
            await _prepare_edit(session, proposal_id, body, mail)
            if body.decision == "edit_approve"
            else None
        )
        outcome = await _decide_locked(session, owner_id, proposal_id, body, edited)
        await session.commit()
        return outcome
    except IntegrityError:
        # A concurrent request won the race to the unique index (same request_id, or another
        # accepting decision on this revision). Report what is stored now, never the raw error.
        await session.rollback()
        stored = await _stored_outcome(session, proposal_id, body.request_id)
        await session.rollback()
        if stored is not None:
            return stored
        raise DecisionConflict("already_decided") from None
    except BaseException:
        await session.rollback()
        raise


async def _decide_locked(
    session: AsyncSession,
    owner_id: uuid.UUID,
    proposal_id: uuid.UUID,
    body: DecisionIn,
    edited: dict[str, Any] | None,
) -> DecisionOutcome:
    # Lock order: review item, then proposal (same as ingestion's stale-marking).
    item = (
        await session.execute(
            select(ReviewItem)
            .where(ReviewItem.proposal_id == proposal_id, ReviewItem.kind == "proposal")
            .order_by(ReviewItem.created_at.desc())
            .limit(1)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    proposal = await session.get(
        Proposal, proposal_id, with_for_update=True, populate_existing=True
    )
    if item is None or proposal is None:
        raise ProposalNotFound(str(proposal_id))
    # A duplicate click that waited on our lock sees the first click's committed outcome.
    stored = await _stored_outcome(session, proposal_id, body.request_id)
    if stored is not None:
        return stored
    if item.status != "pending":
        raise DecisionConflict("not_pending")
    if proposal.status != "awaiting_approval":
        raise DecisionConflict("not_awaiting_approval")
    if proposal.active_revision != body.proposal_revision:
        raise DecisionConflict("stale_revision")
    reviewed = (
        await session.execute(
            select(ProposalRevision).where(
                ProposalRevision.proposal_id == proposal_id,
                ProposalRevision.revision == proposal.active_revision,
            )
        )
    ).scalar_one()
    if reviewed.payload_hash != body.payload_hash:
        raise DecisionConflict("hash_mismatch")
    if item.interrupt_id is None or item.interrupt_id != body.interrupt_id:
        raise DecisionConflict("wrong_interrupt")

    decided = reviewed
    if edited is not None:
        decided = ProposalRevision(
            proposal_id=proposal_id,
            revision=reviewed.revision + 1,
            payload=edited,
            payload_hash=payload_hash(edited),
            source_version=reviewed.source_version,
            preference_version=reviewed.preference_version,
            author="owner",
        )
        session.add(decided)
        await session.flush()
        proposal.active_revision = decided.revision

    approval = Approval(
        proposal_revision_id=decided.id,
        owner_id=owner_id,
        decision=body.decision,
        request_id=body.request_id,
        interrupt_id=body.interrupt_id,
    )
    session.add(approval)
    proposal.status = _NEW_STATUS[body.decision]
    item.status = "resolved"
    item.resolved_at = datetime.now(UTC)
    await session.flush()

    resume: dict[str, Any]
    if body.decision in ACCEPTING:
        resume = {
            "decision": body.decision,
            "approval_id": str(approval.id),
            "revision": decided.revision,
        }
    else:
        resume = {"decision": body.decision, "feedback": body.feedback}
    provider_thread_id = await session.scalar(
        select(WorkflowRun.provider_thread_id).where(WorkflowRun.id == proposal.workflow_id)
    )
    job = await enqueue(
        session,
        "resume_workflow",
        f"resume:{body.interrupt_id}",
        {
            "workflow_id": str(proposal.workflow_id),
            "interrupt_id": body.interrupt_id,
            "resume": resume,
        },
        serial_key=provider_thread_id,
    )
    await _audit_decision(session, proposal.workflow_id, proposal, approval, item, decided.revision)
    return DecisionOutcome(
        status="accepted", approval_id=approval.id, job_id=job.id, revision=decided.revision
    )
