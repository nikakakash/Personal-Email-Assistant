"""ingest_message: dedup, owner/automated skips, pause flag and stale-marking."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AppFlags,
    Base,
    EmailMessage,
    Job,
    Proposal,
    ReviewItem,
    WorkflowRun,
)
from app.ingestion.ingest import ingest_message
from app.providers.interfaces import NormalizedMessage

OWNER = ["owner@example.com", "me@alias.example.com"]


def _msg(
    mid: str = "m1",
    sender: str = "alice@example.com",
    thread: str = "t1",
    headers: dict[str, str] | None = None,
) -> NormalizedMessage:
    return NormalizedMessage(
        provider_message_id=mid,
        provider_thread_id=thread,
        rfc_message_id=f"<{mid}@example.com>",
        sender=sender,
        to=["owner@example.com"],
        cc=[],
        subject="Hello",
        body_text="Can we talk?",
        body_html=None,
        received_at=datetime(2026, 11, 19, 9, 0, tzinfo=UTC),
        headers=headers or {},
    )


async def _count(session: AsyncSession, model: type[Base]) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def test_ingest_queues_once(session: AsyncSession) -> None:
    first = await ingest_message(session, _msg(), OWNER)
    await session.commit()
    second = await ingest_message(session, _msg(), OWNER)
    await session.commit()
    assert first.status == "queued" and first.workflow_job_id is not None
    assert second.status == "duplicate"
    jobs = (await session.execute(select(Job))).scalars().all()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.kind == "start_workflow"
    assert job.dedup_key == "mock-account:m1:start_workflow"
    assert job.serial_key == "t1"
    assert job.id == first.workflow_job_id
    assert await _count(session, EmailMessage) == 1


async def test_ingest_skips_owner_sent_message(session: AsyncSession) -> None:
    senders = {"o1": "Owner <OWNER@example.com>", "o2": "me@alias.example.com"}
    for mid, sender in senders.items():
        res = await ingest_message(session, _msg(mid=mid, sender=sender), OWNER)
        assert res.status == "skipped_owner" and res.workflow_job_id is None
    await session.commit()
    assert await _count(session, Job) == 0
    assert await _count(session, EmailMessage) == 2  # stored, just never workflowed


async def test_ingest_skips_auto_reply(session: AsyncSession) -> None:
    cases = [
        _msg("a", sender="mailer-daemon@example.com"),
        _msg("b", sender="no-reply@shop.example.com"),
        _msg("c", sender="noreply@shop.example.com"),
        _msg("d", headers={"Auto-Submitted": "auto-replied"}),
        _msg("e", headers={"List-Id": "<news.example.com>"}),
    ]
    for m in cases:
        res = await ingest_message(session, m, OWNER)
        assert res.status == "skipped_automated", m.provider_message_id
    res = await ingest_message(session, _msg("f", headers={"Auto-Submitted": "no"}), OWNER)
    assert res.status == "queued"
    await session.commit()
    assert await _count(session, Job) == 1


async def test_ingest_paused_stores_nothing(session: AsyncSession) -> None:
    session.add(AppFlags(id=1, ingestion_paused=True))
    await session.commit()
    res = await ingest_message(session, _msg(), OWNER)
    await session.commit()
    assert res.status == "paused" and res.workflow_job_id is None
    assert await _count(session, EmailMessage) == 0
    assert await _count(session, Job) == 0


async def test_ingest_not_paused_when_flags_row_exists_and_false(session: AsyncSession) -> None:
    session.add(AppFlags(id=1, ingestion_paused=False))
    await session.commit()
    assert (await ingest_message(session, _msg(), OWNER)).status == "queued"


async def _seed_pending_proposal(
    session: AsyncSession, mid: str, thread: str
) -> tuple[ReviewItem, Proposal]:
    first = await ingest_message(session, _msg(mid, thread=thread), OWNER)
    assert first.status == "queued"
    message = (
        await session.execute(select(EmailMessage).where(EmailMessage.provider_message_id == mid))
    ).scalar_one()
    run = WorkflowRun(
        graph_thread_id=f"mock-account:{mid}:wf1",
        message_id=message.id,
        provider_thread_id=thread,
        graph_version="test",
        status="waiting_review",
    )
    session.add(run)
    await session.flush()
    proposal = Proposal(workflow_id=run.id, kind="send_reply")
    session.add(proposal)
    await session.flush()
    item = ReviewItem(
        workflow_id=run.id, kind="proposal", interrupt_id=f"int-{mid}", proposal_id=proposal.id
    )
    session.add(item)
    await session.commit()
    return item, proposal


async def test_ingest_new_message_marks_pending_proposal_stale(session: AsyncSession) -> None:
    item, proposal = await _seed_pending_proposal(session, "m1", "t1")
    other_item, other_proposal = await _seed_pending_proposal(session, "x1", "t-other")

    res = await ingest_message(session, _msg("m2", thread="t1"), OWNER)
    await session.commit()
    assert res.status == "queued"

    await session.refresh(item)
    await session.refresh(proposal)
    assert item.status == "stale" and proposal.status == "stale"
    await session.refresh(other_item)
    await session.refresh(other_proposal)
    assert other_item.status == "pending" and other_proposal.status == "awaiting_approval"

    job = (await session.execute(select(Job).where(Job.kind == "resume_workflow"))).scalar_one()
    assert job.dedup_key == f"resume:superseded:{item.id}"
    assert job.serial_key == "t1"
    assert job.payload == {
        "workflow_id": str(item.workflow_id),
        "review_item_id": str(item.id),
        "interrupt_id": "int-m1",
        "resume": {"decision": "superseded"},
    }


async def test_duplicate_does_not_mark_stale(session: AsyncSession) -> None:
    item, _ = await _seed_pending_proposal(session, "m1", "t1")
    res = await ingest_message(session, _msg("m1", thread="t1"), OWNER)
    await session.commit()
    assert res.status == "duplicate"
    await session.refresh(item)
    assert item.status == "pending"
    assert await _count(session, Job) == 1


async def test_owner_reply_marks_pending_proposal_stale(session: AsyncSession) -> None:
    item, proposal = await _seed_pending_proposal(session, "m1", "t1")
    res = await ingest_message(session, _msg("o1", sender="owner@example.com", thread="t1"), OWNER)
    await session.commit()
    assert res.status == "skipped_owner" and res.workflow_job_id is None
    await session.refresh(item)
    await session.refresh(proposal)
    assert item.status == "stale" and proposal.status == "stale"
    resumes = (await session.execute(select(Job).where(Job.kind == "resume_workflow"))).scalars()
    assert [j.dedup_key for j in resumes] == [f"resume:superseded:{item.id}"]
    starts = await session.execute(select(Job).where(Job.dedup_key.like("%o1:start_workflow")))
    assert starts.scalars().all() == []


async def test_auto_reply_does_not_mark_stale(session: AsyncSession) -> None:
    item, proposal = await _seed_pending_proposal(session, "m1", "t1")
    cases = [
        _msg("a1", sender="mailer-daemon@example.com", thread="t1"),
        _msg("a2", thread="t1", headers={"Auto-Submitted": "auto-replied"}),
        _msg("a3", thread="t1", headers={"List-Id": "<l.example.com>"}),
    ]
    for m in cases:
        assert (await ingest_message(session, m, OWNER)).status == "skipped_automated"
    await session.commit()
    await session.refresh(item)
    await session.refresh(proposal)
    assert item.status == "pending" and proposal.status == "awaiting_approval"
    assert await _count(session, Job) == 1


async def test_approved_proposal_is_not_made_stale(session: AsyncSession) -> None:
    item, proposal = await _seed_pending_proposal(session, "m1", "t1")
    proposal.status = "approved"
    await session.commit()
    res = await ingest_message(session, _msg("m2", thread="t1"), OWNER)
    await session.commit()
    assert res.status == "queued"
    await session.refresh(item)
    await session.refresh(proposal)
    assert proposal.status == "approved" and item.status == "pending"
    jobs = (await session.execute(select(Job).where(Job.kind == "resume_workflow"))).scalars()
    assert jobs.all() == []
