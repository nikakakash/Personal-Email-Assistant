"""Immutable proposals: `create_proposal` writes revision 1 and opens the owner's review.

A proposal is the exact payload of one outgoing operation (a reply or a calendar event). Its
revisions never change after insert; an owner edit adds a new revision (see `decisions.py`), and
approvals bind to one revision and its payload hash. None of these helpers commit: the caller
owns the transaction.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.hashing import payload_hash
from app.db.models import Proposal, ProposalRevision, ReviewItem, WorkflowEvent, WorkflowRun
from app.graphs.nodes.finalize import FINAL_STATUSES

ProposalKind = Literal["send_reply", "create_event"]


class RunIsFinal(Exception):  # noqa: N818 - reads as a state, like ProposalNotFound
    """The workflow run is already final (e.g. cancelled): no new proposal may be opened."""


async def active_revision(session: AsyncSession, proposal: Proposal) -> ProposalRevision:
    """The proposal's currently active revision row."""
    return (
        await session.execute(
            select(ProposalRevision).where(
                ProposalRevision.proposal_id == proposal.id,
                ProposalRevision.revision == proposal.active_revision,
            )
        )
    ).scalar_one()


async def create_proposal(
    session: AsyncSession,
    workflow_id: uuid.UUID,
    kind: ProposalKind,
    payload: dict[str, Any],
    source_version: str | None,
    preference_version: str | None,
    *,
    idempotency_key: str,
) -> tuple[Proposal, ProposalRevision]:
    """Create the proposal, revision 1, a pending review item and an `approval_required` event.

    Idempotent per `idempotency_key` (the graph passes `f"{workflow_id}:{step_count}"`): a
    re-run returns the existing proposal and its active revision and writes nothing; a key hit
    with a different `workflow_id` or `kind` raises ValueError. The hash
    covers the payload only. Flushes; does not commit.

    Raises `RunIsFinal` (writing nothing) if the run is final. The run row is locked first,
    the same order `cancel_workflow` locks in, so a concurrent cancel either committed before
    (and is seen here) or waits and then cancels what this call created.
    """
    # FOR NO KEY UPDATE: serializes with cancel's run lock, but not with the FOR KEY SHARE
    # lock that inserting rows referencing the run takes (see `cancel_workflow`).
    run_status = await session.scalar(
        select(WorkflowRun.status)
        .where(WorkflowRun.id == workflow_id)
        .with_for_update(key_share=True)
    )
    if run_status is None or run_status in FINAL_STATUSES:
        raise RunIsFinal(str(workflow_id))
    proposal_id = (
        await session.execute(
            insert(Proposal)
            .values(workflow_id=workflow_id, kind=kind, idempotency_key=idempotency_key)
            .on_conflict_do_nothing(index_elements=[Proposal.idempotency_key])
            .returning(Proposal.id)
        )
    ).scalar_one_or_none()
    if proposal_id is None:
        existing = (
            await session.execute(
                select(Proposal).where(Proposal.idempotency_key == idempotency_key)
            )
        ).scalar_one()
        if existing.workflow_id != workflow_id or existing.kind != kind:
            raise ValueError(
                f"idempotency key {idempotency_key!r} already used by another workflow or kind"
            )
        return existing, await active_revision(session, existing)

    proposal = await session.get_one(Proposal, proposal_id)
    revision = ProposalRevision(
        proposal_id=proposal_id,
        revision=1,
        payload=payload,
        payload_hash=payload_hash(payload),
        source_version=source_version,
        preference_version=preference_version,
        author="model",
    )
    session.add(revision)
    item = ReviewItem(
        workflow_id=workflow_id,
        kind="proposal",
        status="pending",
        proposal_id=proposal_id,
        payload={"proposal_kind": kind, "revision": 1},
    )
    session.add(item)
    await session.flush()
    session.add(
        WorkflowEvent(
            workflow_id=workflow_id,
            event_type="approval_required",
            payload={
                "proposal_id": str(proposal_id),
                "review_item_id": str(item.id),
                "kind": kind,
                "revision": 1,
                "payload_hash": revision.payload_hash,
            },
        )
    )
    await session.execute(
        update(WorkflowRun)
        .where(WorkflowRun.id == workflow_id, WorkflowRun.status.not_in(FINAL_STATUSES))
        .values(status="waiting_review", updated_at=datetime.now(UTC))
    )
    await session.flush()
    return proposal, revision
