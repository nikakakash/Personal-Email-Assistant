"""decide_proposal: binding to revision/hash/interrupt, idempotent request ids, edits, races."""

import asyncio
import uuid
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions import decisions
from app.actions.decisions import (
    DecisionConflict,
    DecisionIn,
    InvalidEdit,
    ProposalNotFound,
    decide_proposal,
)
from app.db.models import (
    Approval,
    AuditEvent,
    Job,
    Proposal,
    ProposalRevision,
    ReviewItem,
)
from app.providers.mock_mail import MockMailProvider
from tests.actions.conftest import Seeded, SeedFactory


def _body(s: Seeded, decision: str = "approve", **overrides: Any) -> DecisionIn:
    fields: dict[str, Any] = {
        "decision": decision,
        "proposal_revision": s.revision,
        "payload_hash": s.payload_hash,
        "interrupt_id": s.interrupt_id,
        "request_id": f"req-{decision}",
    }
    fields.update(overrides)
    return DecisionIn(**fields)


async def _count(session: AsyncSession, model: type) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def _conflict(
    session: AsyncSession, owner_id: uuid.UUID, s: Seeded, body: DecisionIn, mail: MockMailProvider
) -> str:
    with pytest.raises(DecisionConflict) as exc_info:
        await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    return exc_info.value.reason


async def test_approve_resolves_item_and_enqueues_resume(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    out = await decide_proposal(session, owner_id, s.proposal_id, _body(s), mail=mail)
    assert out.status == "accepted"
    assert out.revision == 1

    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    item = await session.get_one(ReviewItem, s.review_item_id, populate_existing=True)
    approval = await session.get_one(Approval, out.approval_id)
    job = await session.get_one(Job, out.job_id)
    assert proposal.status == "approved"
    assert item.status == "resolved" and item.resolved_at is not None
    assert approval.decision == "approve"
    revision = await session.get_one(ProposalRevision, approval.proposal_revision_id)
    assert (revision.proposal_id, revision.revision) == (s.proposal_id, 1)
    assert approval.interrupt_id == s.interrupt_id
    assert job.kind == "resume_workflow"
    assert job.dedup_key == f"resume:{s.interrupt_id}"
    assert job.payload == {
        "workflow_id": str(s.run_id),
        "interrupt_id": s.interrupt_id,
        "resume": {"decision": "approve", "approval_id": str(out.approval_id), "revision": 1},
    }
    audit = (await session.execute(select(AuditEvent))).scalars().all()
    assert [a.event_type for a in audit] == ["decision_recorded"]
    assert "body_text" not in str(audit[0].detail)


async def test_decision_stale_revision_409(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    assert await _conflict(session, owner_id, s, _body(s, proposal_revision=2), mail) == (
        "stale_revision"
    )
    assert await _count(session, Approval) == 0


async def test_decision_wrong_hash_409(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    assert await _conflict(session, owner_id, s, _body(s, payload_hash="0" * 64), mail) == (
        "hash_mismatch"
    )


async def test_decision_wrong_interrupt_409(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    assert await _conflict(session, owner_id, s, _body(s, interrupt_id="other"), mail) == (
        "wrong_interrupt"
    )


async def test_decision_on_stale_item_409(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    item = await session.get_one(ReviewItem, s.review_item_id)
    item.status = "stale"
    proposal = await session.get_one(Proposal, s.proposal_id)
    proposal.status = "stale"
    await session.commit()
    assert await _conflict(session, owner_id, s, _body(s), mail) == "not_pending"
    assert await _count(session, Approval) == 0
    assert await _count(session, Job) == 0


async def test_duplicate_request_id_returns_same_outcome(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    first = await decide_proposal(session, owner_id, s.proposal_id, _body(s), mail=mail)
    second = await decide_proposal(session, owner_id, s.proposal_id, _body(s), mail=mail)
    assert second == first
    assert await _count(session, Approval) == 1
    assert await _count(session, Job) == 1


async def test_reject_records_approval_row_and_resume(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    out = await decide_proposal(session, owner_id, s.proposal_id, _body(s, "reject"), mail=mail)
    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    approval = await session.get_one(Approval, out.approval_id)
    job = await session.get_one(Job, out.job_id)
    assert proposal.status == "rejected"
    assert approval.decision == "reject"
    assert job.payload["resume"] == {"decision": "reject", "feedback": None}


async def test_feedback_supersedes_and_requires_text(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    with pytest.raises(ValidationError):
        _body(s, "feedback", feedback="   ")
    out = await decide_proposal(
        session, owner_id, s.proposal_id, _body(s, "feedback", feedback="Shorter."), mail=mail
    )
    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    job = await session.get_one(Job, out.job_id)
    assert proposal.status == "superseded"
    assert job.payload["resume"] == {"decision": "feedback", "feedback": "Shorter."}


async def test_decision_in_forbids_extra_fields_and_stray_payload(seed: SeedFactory) -> None:
    s = await seed()
    with pytest.raises(ValidationError):
        _body(s, payload={"to": ["x@evil.test"]})
    with pytest.raises(ValidationError):
        _body(s, "approve", edited_payload={"body_text": "x"})
    with pytest.raises(ValidationError):
        _body(s, "edit_approve")


async def test_edit_approve_creates_new_revision_and_old_hash_rejected(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    old_hash = s.payload_hash
    body = _body(
        s,
        "edit_approve",
        edited_payload={"body_text": "Tuesday at noon works.", "cc": ["carol@example.com"]},
    )
    out = await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    assert out.revision == 2

    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    rev2 = (
        await session.execute(
            select(ProposalRevision).where(
                ProposalRevision.proposal_id == s.proposal_id, ProposalRevision.revision == 2
            )
        )
    ).scalar_one()
    approval = await session.get_one(Approval, out.approval_id)
    assert proposal.active_revision == 2 and proposal.status == "approved"
    assert rev2.author == "owner"
    assert rev2.payload["body_text"] == "Tuesday at noon works."
    assert rev2.payload["cc"] == ["carol@example.com"]  # owner may add a recipient
    # Threading headers are copied from the previous revision, never taken from the browser.
    assert rev2.payload["in_reply_to"] == s.payload["in_reply_to"]
    assert rev2.payload_hash != old_hash
    assert approval.decision == "edit_approve"
    assert approval.proposal_revision_id == rev2.id

    for revision in (1, 2):
        stale = _body(s, proposal_revision=revision, payload_hash=old_hash, request_id="again")
        with pytest.raises(DecisionConflict):
            await decide_proposal(session, owner_id, s.proposal_id, stale, mail=mail)
    assert await _count(session, Approval) == 1


async def test_edit_rejects_non_editable_field(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    body = _body(s, "edit_approve", edited_payload={"subject": "Totally new subject"})
    with pytest.raises(InvalidEdit) as exc_info:
        await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    assert exc_info.value.errors == ["field_not_editable:subject"]
    assert await _count(session, ProposalRevision) == 1
    assert await _count(session, Approval) == 0


async def test_edit_rejects_invalid_payload(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    body = _body(s, "edit_approve", edited_payload={"body_text": "   "})
    with pytest.raises(InvalidEdit) as exc_info:
        await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    assert "empty_body" in exc_info.value.errors


async def test_edit_normalizes_addresses(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    bad = _body(s, "edit_approve", edited_payload={"to": ["not an address"]})
    with pytest.raises(InvalidEdit):
        await decide_proposal(session, owner_id, s.proposal_id, bad, mail=mail)

    body = _body(
        s,
        "edit_approve",
        request_id="req-edit-2",
        edited_payload={"to": ["Alice <ALICE@Example.com>", "Bob@Example.COM"]},
    )
    await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    rev2 = (
        await session.execute(select(ProposalRevision).where(ProposalRevision.revision == 2))
    ).scalar_one()
    assert rev2.payload["to"] == ["alice@example.com", "bob@example.com"]


async def test_edit_event_normalizes_times_to_utc(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed("create_event")
    body = _body(
        s,
        "edit_approve",
        edited_payload={
            "start": "2026-11-25T14:00:00+01:00",
            "end": "2026-11-25T15:00:00+01:00",
            "attendees": ["Alice@Example.com", "dave@example.com"],
        },
    )
    out = await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    rev2 = (
        await session.execute(select(ProposalRevision).where(ProposalRevision.revision == 2))
    ).scalar_one()
    assert out.revision == 2
    assert rev2.payload["start"] == "2026-11-25T13:00:00+00:00"
    assert rev2.payload["end"] == "2026-11-25T14:00:00+00:00"
    assert rev2.payload["attendees"] == ["alice@example.com", "dave@example.com"]


async def test_decision_enqueues_resume_in_same_transaction(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed()

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("crash after the approval and job inserts")

    monkeypatch.setattr(decisions, "_audit_decision", boom)
    with pytest.raises(RuntimeError):
        await decide_proposal(session, owner_id, s.proposal_id, _body(s), mail=mail)

    assert await _count(session, Approval) == 0
    assert await _count(session, Job) == 0
    item = await session.get_one(ReviewItem, s.review_item_id, populate_existing=True)
    assert item.status == "pending"


async def test_concurrent_approvals_one_wins(
    sessionmaker: async_sessionmaker[AsyncSession],
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
) -> None:
    s = await seed()

    async def approve(request_id: str) -> object:
        async with sessionmaker() as own:
            return await decide_proposal(
                own, owner_id, s.proposal_id, _body(s, request_id=request_id), mail=mail
            )

    results = await asyncio.gather(approve("r-a"), approve("r-b"), return_exceptions=True)
    accepted = [r for r in results if isinstance(r, decisions.DecisionOutcome)]
    conflicts = [r for r in results if isinstance(r, DecisionConflict)]
    assert len(accepted) == 1, results
    assert len(conflicts) == 1, results
    async with sessionmaker() as check:
        rows = (await check.execute(select(Approval))).scalars().all()
        assert [r.decision for r in rows] == ["approve"]
        assert await _count(check, Job) == 1


async def test_same_request_id_concurrent_returns_same_outcome(
    sessionmaker: async_sessionmaker[AsyncSession],
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
) -> None:
    s = await seed()

    async def approve() -> decisions.DecisionOutcome:
        async with sessionmaker() as own:
            return await decide_proposal(
                own, owner_id, s.proposal_id, _body(s, request_id="same"), mail=mail
            )

    first, second = await asyncio.gather(approve(), approve())
    assert first == second
    async with sessionmaker() as check:
        assert await _count(check, Approval) == 1
        assert await _count(check, Job) == 1


async def test_stale_edit_is_conflict_not_invalid_edit(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    # Both stale and invalid: staleness is checked first, so the owner gets a 409.
    stale = _body(
        s, "edit_approve", payload_hash="0" * 64, edited_payload={"subject": "not editable"}
    )
    assert await _conflict(session, owner_id, s, stale, mail) == "hash_mismatch"
    wrong_rev = _body(s, "edit_approve", proposal_revision=5, edited_payload={"body_text": " "})
    assert await _conflict(session, owner_id, s, wrong_rev, mail) == "stale_revision"


async def test_edit_on_missing_proposal_is_not_found(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    body = _body(s, "edit_approve", edited_payload={"body_text": "x"})
    with pytest.raises(ProposalNotFound):
        await decide_proposal(session, owner_id, uuid.uuid4(), body, mail=mail)
    with pytest.raises(ProposalNotFound):
        await decide_proposal(session, owner_id, uuid.uuid4(), _body(s), mail=mail)


async def test_stored_outcome_tolerates_missing_job(
    session: AsyncSession, owner_id: uuid.UUID, seed: SeedFactory, mail: MockMailProvider
) -> None:
    s = await seed()
    first = await decide_proposal(session, owner_id, s.proposal_id, _body(s), mail=mail)
    await session.execute(delete(Job))
    await session.commit()
    again = await decide_proposal(session, owner_id, s.proposal_id, _body(s), mail=mail)
    assert again.approval_id == first.approval_id
    assert again.job_id is None
