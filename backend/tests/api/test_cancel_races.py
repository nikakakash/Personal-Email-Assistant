"""Cancel racing the other writers of a run, each on its own session (real Postgres locks).

Cancel locks the run row first; the other side holds a review-item, proposal or action lock
and then inserts rows that reference the run (approval audit, events). With a plain
`FOR UPDATE` run lock that is a lock cycle and Postgres aborts one side (40P01). Each race
runs several times; every round must finish without an exception and in a consistent state.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.decisions import DecisionConflict, DecisionIn, DecisionOutcome, decide_proposal
from app.actions.executor import ActionExecutor, ActionOutcome, _Claim, operation_id_for
from app.actions.reviews import CancelOutcome, ReviewOutcome, answer_question, cancel_workflow
from app.config import Settings
from app.db.models import (
    Action,
    Approval,
    Job,
    Owner,
    Proposal,
    ProposalRevision,
    ReviewItem,
    WorkflowRun,
)
from app.providers.mock_calendar import MockCalendarProvider
from app.providers.mock_mail import MockFaults, MockMailProvider
from tests.api.conftest import SeededProposal

Session = async_sessionmaker[AsyncSession]
SeedProposal = Callable[[], Awaitable[SeededProposal]]
SeedItem = Callable[..., Awaitable[ReviewItem]]
ROUNDS = 5


async def _owner_id(sessionmaker: Session) -> uuid.UUID:
    async with sessionmaker() as s, s.begin():
        owner = (await s.execute(select(Owner))).scalars().first()
        if owner is None:
            owner = Owner(email="owner@example.com")
            s.add(owner)
            await s.flush()
        return owner.id


async def _get(sessionmaker: Session, model: Any, row_id: uuid.UUID) -> Any:
    async with sessionmaker() as s:
        return await s.get_one(model, row_id)


async def _count(sessionmaker: Session, model: Any, *where: Any) -> int:
    async with sessionmaker() as s:
        return int(await s.scalar(select(func.count()).select_from(model).where(*where)) or 0)


async def _cancel(sessionmaker: Session, workflow_id: uuid.UUID) -> CancelOutcome:
    async with sessionmaker() as s:
        return await cancel_workflow(s, workflow_id, request_id=f"c-{uuid.uuid4().hex}")


async def test_cancel_races_approval(
    sessionmaker: Session, seed_proposal: SeedProposal, settings: Settings
) -> None:
    owner_id = await _owner_id(sessionmaker)
    mail = MockMailProvider(sessionmaker, MockFaults())
    for _ in range(ROUNDS):
        seeded = await seed_proposal()
        body = DecisionIn(
            decision="approve",
            proposal_revision=seeded.revision,
            payload_hash=seeded.payload_hash,
            interrupt_id=seeded.interrupt_id,
            request_id=uuid.uuid4().hex,
        )

        async def approve(seeded: SeededProposal = seeded, body: DecisionIn = body) -> Any:
            async with sessionmaker() as s:
                return await decide_proposal(s, owner_id, seeded.proposal_id, body, mail=mail)

        decided, cancelled = await asyncio.gather(
            approve(), _cancel(sessionmaker, seeded.workflow_id), return_exceptions=True
        )
        assert isinstance(cancelled, CancelOutcome), repr(cancelled)
        proposal = await _get(sessionmaker, Proposal, seeded.proposal_id)
        item = await _get(sessionmaker, ReviewItem, seeded.review_item_id)
        approvals = await _count(sessionmaker, Approval, Approval.request_id == body.request_id)
        if isinstance(decided, DecisionOutcome):
            # Approval first: cancel found the approved (unexecuted) proposal and cancelled it.
            assert approvals == 1
            assert item.status == "resolved"
            assert cancelled.cancelled_review_items == []
            assert cancelled.cancelled_proposals == [seeded.proposal_id]
        else:
            # Cancel first: the decision is refused with a 409, nothing was approved.
            assert isinstance(decided, DecisionConflict), repr(decided)
            assert decided.reason in ("run_cancelled", "not_pending")
            assert approvals == 0
            assert item.status == "cancelled"
            assert cancelled.cancelled_review_items == [seeded.review_item_id]
        assert proposal.status == "cancelled_before_execution"
        assert (await _get(sessionmaker, WorkflowRun, seeded.workflow_id)).status == "cancelled"


async def test_cancel_races_question_answer(sessionmaker: Session, seed_item: SeedItem) -> None:
    for _ in range(ROUNDS):
        item = await seed_item("question")
        interrupt_id = cast(str, item.interrupt_id)

        async def answer(item: ReviewItem = item, interrupt_id: str = interrupt_id) -> Any:
            async with sessionmaker() as s:
                return await answer_question(
                    s, item.id, answer="Tuesday", request_id="a-1", interrupt_id=interrupt_id
                )

        answered, cancelled = await asyncio.gather(
            answer(), _cancel(sessionmaker, item.workflow_id), return_exceptions=True
        )
        assert isinstance(cancelled, CancelOutcome), repr(cancelled)
        stored = await _get(sessionmaker, ReviewItem, item.id)
        jobs = await _scalars(
            sessionmaker, select(Job).where(Job.dedup_key == f"resume:{interrupt_id}")
        )
        # One interrupt, at most one resume, whoever won.
        assert len(jobs) == 1
        if isinstance(answered, ReviewOutcome):
            assert stored.status == "resolved"
            assert cancelled.cancelled_review_items == [] and cancelled.job_id is None
            assert jobs[0].payload["resume"]["decision"] == "answer"
        else:
            assert isinstance(answered, DecisionConflict), repr(answered)
            assert answered.reason == "not_pending"
            assert stored.status == "cancelled"
            assert cancelled.job_id == jobs[0].id
            assert jobs[0].payload["resume"] == {"decision": "cancelled"}


async def _scalars(sessionmaker: Session, stmt: Any) -> list[Any]:
    async with sessionmaker() as s:
        return list((await s.execute(stmt)).scalars())


async def _executing_action(
    sessionmaker: Session, seeded: SeededProposal, owner_id: uuid.UUID
) -> _Claim:
    """The proposal approved and its action claimed (`executing`), as the executor leaves it."""
    async with sessionmaker() as s, s.begin():
        proposal = await s.get_one(Proposal, seeded.proposal_id)
        proposal.status = "approved"
        item = await s.get_one(ReviewItem, seeded.review_item_id)
        item.status = "resolved"
        revision = (
            await s.execute(
                select(ProposalRevision).where(ProposalRevision.proposal_id == proposal.id)
            )
        ).scalar_one()
        approval = Approval(
            proposal_revision_id=revision.id,
            owner_id=owner_id,
            decision="approve",
            request_id=uuid.uuid4().hex,
        )
        s.add(approval)
        await s.flush()
        operation_id = operation_id_for(proposal.id, revision.revision)
        action = Action(
            operation_id=operation_id,
            approval_id=approval.id,
            kind="send_reply",
            request_fingerprint=revision.payload_hash,
            status="executing",
            attempts=1,
        )
        s.add(action)
        await s.flush()
        return _Claim(
            action_id=action.id,
            workflow_id=seeded.workflow_id,
            proposal_id=proposal.id,
            kind="send_reply",
            operation_id=operation_id,
            status="executing",
            provider_ref=None,
            error=None,
            new=True,
            in_flight=False,
            request=cast(Any, None),  # `_record` does not use the request
        )


async def test_cancel_races_executor_record(
    sessionmaker: Session, seed_proposal: SeedProposal, settings: Settings
) -> None:
    owner_id = await _owner_id(sessionmaker)
    faults = MockFaults()
    executor = ActionExecutor(
        sessionmaker,
        MockMailProvider(sessionmaker, faults),
        MockCalendarProvider(sessionmaker, None, "UTC", faults),
        settings,
    )
    for _ in range(ROUNDS):
        seeded = await seed_proposal()
        claim = await _executing_action(sessionmaker, seeded, owner_id)
        recorded, cancelled = await asyncio.gather(
            executor._record(claim, ActionOutcome("succeeded", "mock-1", None), True),
            _cancel(sessionmaker, seeded.workflow_id),
            return_exceptions=True,
        )
        assert isinstance(recorded, ActionOutcome), repr(recorded)
        assert isinstance(cancelled, CancelOutcome), repr(cancelled)
        action = await _get(sessionmaker, Action, claim.action_id)
        assert action.status == "succeeded" and action.provider_ref == "mock-1"
        # The write happened, so cancel left the proposal alone and reported the action.
        assert (await _get(sessionmaker, Proposal, seeded.proposal_id)).status == "executed"
        assert cancelled.cancelled_proposals == []
        assert [a.action_id for a in cancelled.completed_actions] == [claim.action_id]
