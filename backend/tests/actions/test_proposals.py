"""create_proposal: immutable revision 1, pending review item, event, idempotency."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.hashing import payload_hash
from app.actions.proposals import create_proposal
from app.db.models import Proposal, ProposalRevision, ReviewItem, WorkflowEvent, WorkflowRun
from tests.actions.conftest import SeedFactory


async def _count(session: AsyncSession, model: type) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def test_create_proposal_creates_revision_review_item_and_event(
    session: AsyncSession, seed: SeedFactory
) -> None:
    s = await seed()
    proposal = await session.get_one(Proposal, s.proposal_id)
    revision = (await session.execute(select(ProposalRevision))).scalar_one()
    item = await session.get_one(ReviewItem, s.review_item_id)
    assert proposal.status == "awaiting_approval"
    assert proposal.active_revision == 1
    assert proposal.idempotency_key == f"{s.run_id}:1"
    assert revision.revision == 1
    assert revision.author == "model"
    assert revision.payload_hash == payload_hash(s.payload)
    assert revision.source_version == "sv1"
    assert revision.preference_version == "pv1"
    assert item.kind == "proposal"
    assert item.status == "pending"
    assert item.proposal_id == s.proposal_id
    assert item.workflow_id == s.run_id

    events = (
        (await session.execute(select(WorkflowEvent).where(WorkflowEvent.workflow_id == s.run_id)))
        .scalars()
        .all()
    )
    assert [e.event_type for e in events] == ["approval_required"]
    assert events[0].payload["proposal_id"] == str(s.proposal_id)
    run = await session.get_one(WorkflowRun, s.run_id, populate_existing=True)
    assert run.status == "waiting_review"


async def test_create_proposal_is_idempotent_per_key(
    session: AsyncSession, seed: SeedFactory
) -> None:
    s = await seed()
    again_p, again_r = await create_proposal(
        session,
        s.run_id,
        "send_reply",
        {**s.payload, "body_text": "something else"},
        source_version="sv2",
        preference_version="pv2",
        idempotency_key=f"{s.run_id}:1",
    )
    await session.commit()
    assert again_p.id == s.proposal_id
    assert again_r.revision == 1
    assert again_r.payload == s.payload
    assert await _count(session, Proposal) == 1
    assert await _count(session, ProposalRevision) == 1
    assert await _count(session, ReviewItem) == 1
    assert await _count(session, WorkflowEvent) == 1


async def test_create_proposal_key_reuse_with_other_kind_raises(
    session: AsyncSession, seed: SeedFactory
) -> None:
    s = await seed()
    with pytest.raises(ValueError):
        await create_proposal(
            session,
            s.run_id,
            "create_event",
            s.payload,
            source_version=None,
            preference_version=None,
            idempotency_key=f"{s.run_id}:1",
        )
