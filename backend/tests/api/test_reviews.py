"""Review queue, workflow list and detail, question/notification answers and cancel."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.reviews import CancelOutcome, ReviewOutcome, answer_question, cancel_workflow
from app.db.models import (
    Action,
    Approval,
    EmailMessage,
    Job,
    Owner,
    Proposal,
    ProposalRevision,
    ReviewItem,
    WorkflowEvent,
    WorkflowRun,
)
from tests.api.conftest import SeededProposal

SeedItem = Callable[..., Awaitable[ReviewItem]]
SeedProposal = Callable[[], Awaitable[SeededProposal]]
WorkflowFactory = Callable[[], Awaitable[WorkflowRun]]


async def test_reviews_pagination_and_filter(authed: AsyncClient, seed_item: SeedItem) -> None:
    questions = [await seed_item("question", subject=f"Q{i}") for i in range(3)]
    await seed_item("notification", subject="N")
    await seed_item("question", subject="Done", status="resolved")

    first = await authed.get(
        "/api/v1/reviews", params={"status": "pending", "kind": "question", "limit": 2}
    )
    assert first.status_code == 200
    page1 = first.json()
    assert [i["subject"] for i in page1["items"]] == ["Q2", "Q1"]  # newest first
    item = page1["items"][0]
    assert item["id"] == str(questions[2].id)
    assert item["kind"] == "question" and item["status"] == "pending"
    assert item["sender"] == "alice@example.com"
    assert item["classification"] == "respond" and item["reason"] == "Reason for Q2"
    assert item["interrupt_id"] == questions[2].interrupt_id
    assert item["received_at"] and item["created_at"]
    assert page1["next_cursor"]

    second = await authed.get(
        "/api/v1/reviews",
        params={
            "status": "pending",
            "kind": "question",
            "limit": 2,
            "cursor": page1["next_cursor"],
        },
    )
    page2 = second.json()
    assert [i["subject"] for i in page2["items"]] == ["Q0"]
    assert page2["next_cursor"] is None

    pending = (await authed.get("/api/v1/reviews")).json()["items"]  # default: pending
    assert {i["subject"] for i in pending} == {"Q0", "Q1", "Q2", "N"}
    resolved = (await authed.get("/api/v1/reviews", params={"status": "resolved"})).json()
    assert [i["subject"] for i in resolved["items"]] == ["Done"]

    assert (
        await authed.get("/api/v1/reviews", params={"cursor": "not-a-cursor"})
    ).status_code == 422
    assert (await authed.get("/api/v1/reviews", params={"status": "bogus"})).status_code == 422


async def test_workflow_html_is_sanitized(
    authed: AsyncClient, session: AsyncSession, workflow_factory: WorkflowFactory
) -> None:
    run = await workflow_factory()
    message = await session.get_one(EmailMessage, run.message_id)
    message.body_text = "Plain text body"
    message.body_html = (
        '<p onclick="steal()">Hi <b>there</b></p><script>alert(1)</script>'
        '<img src="http://tracker.example/pixel.gif"><a href="javascript:alert(1)">x</a>'
    )
    session.add(
        WorkflowEvent(workflow_id=run.id, event_type="node_started", payload={"node": "triage"})
    )
    await session.commit()

    r = await authed.get(f"/api/v1/workflows/{run.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["run"]["id"] == str(run.id)
    [msg] = body["thread"]
    assert msg["body_text"] == "Plain text body"
    html = msg["body_html"]
    assert "<b>there</b>" in html
    for forbidden in ("<script", "onclick", "<img", "tracker.example", "javascript:"):
        assert forbidden not in html
    assert [e["type"] for e in body["events"]] == ["node_started"]
    assert body["review_items"] == [] and body["proposals"] == [] and body["actions"] == []


async def test_workflow_detail_lists_proposals_with_active_revision(
    authed: AsyncClient, seed_proposal: SeedProposal
) -> None:
    seeded = await seed_proposal()
    body = (await authed.get(f"/api/v1/workflows/{seeded.workflow_id}")).json()
    [proposal] = body["proposals"]
    assert proposal["id"] == str(seeded.proposal_id)
    assert proposal["revision"] == seeded.revision
    assert proposal["payload_hash"] == seeded.payload_hash
    assert proposal["payload"]["body_text"] == "Tuesday works."
    [item] = body["review_items"]
    assert item["interrupt_id"] == seeded.interrupt_id and item["kind"] == "proposal"


async def test_workflow_list_status_filter(
    authed: AsyncClient, session: AsyncSession, workflow_factory: WorkflowFactory
) -> None:
    runs = [await workflow_factory() for _ in range(4)]
    for run, status in zip(
        runs, ["failed", "done_no_action", "done_with_actions", "running"], strict=True
    ):
        run.status = status
    runs[0].error_code = "internal_error"
    await session.commit()

    failed = (await authed.get("/api/v1/workflows", params={"status": "failed"})).json()
    assert [w["id"] for w in failed["items"]] == [str(runs[0].id)]
    assert failed["items"][0]["error_code"] == "internal_error"

    done = await authed.get(
        "/api/v1/workflows", params=[("status", "done_no_action"), ("status", "done_with_actions")]
    )
    assert {w["id"] for w in done.json()["items"]} == {str(runs[1].id), str(runs[2].id)}

    paged = (await authed.get("/api/v1/workflows", params={"limit": 3})).json()
    assert len(paged["items"]) == 3 and paged["next_cursor"]
    rest = (
        await authed.get("/api/v1/workflows", params={"limit": 3, "cursor": paged["next_cursor"]})
    ).json()
    assert len(rest["items"]) == 1 and rest["next_cursor"] is None

    assert (await authed.get("/api/v1/workflows", params={"status": "bogus"})).status_code == 422


async def _job(session: AsyncSession, dedup_key: str) -> Job:
    return (await session.execute(select(Job).where(Job.dedup_key == dedup_key))).scalar_one()


async def test_question_answer_enqueues_resume(
    authed: AsyncClient, session: AsyncSession, seed_item: SeedItem
) -> None:
    item = await seed_item("question")
    url = f"/api/v1/questions/{item.id}/answer"
    body = {"answer": "Tuesday", "request_id": "req-1", "interrupt_id": item.interrupt_id}

    r = await authed.post(url, json=body)
    assert r.status_code == 202
    job = await _job(session, f"resume:{item.interrupt_id}")
    assert r.json() == {"status": "accepted", "review_item_id": str(item.id), "job_id": str(job.id)}
    assert job.kind == "resume_workflow"
    assert job.payload == {
        "workflow_id": str(item.workflow_id),
        "interrupt_id": item.interrupt_id,
        "review_item_id": str(item.id),
        "resume": {"decision": "answer", "answer": "Tuesday"},
    }
    await session.refresh(item)
    assert item.status == "resolved" and item.resolved_at is not None
    assert item.payload["decision_request_id"] == "req-1"

    again = await authed.post(url, json=body)  # same request id: same outcome
    assert again.status_code == 202 and again.json() == r.json()
    other = await authed.post(url, json={**body, "request_id": "req-2"})
    assert other.status_code == 409 and other.json() == {
        "error": "conflict",
        "reason": "not_pending",
    }


async def test_question_answer_wrong_interrupt_409(
    authed: AsyncClient, seed_item: SeedItem
) -> None:
    item = await seed_item("question")
    r = await authed.post(
        f"/api/v1/questions/{item.id}/answer",
        json={"answer": "Tuesday", "request_id": "req-1", "interrupt_id": "old-interrupt"},
    )
    assert r.status_code == 409 and r.json()["reason"] == "wrong_interrupt"
    empty = await authed.post(
        f"/api/v1/questions/{item.id}/answer",
        json={"answer": "  ", "request_id": "req-2", "interrupt_id": item.interrupt_id},
    )
    assert empty.status_code == 422


async def test_notification_decision_enqueues_resume(
    authed: AsyncClient, session: AsyncSession, seed_item: SeedItem
) -> None:
    item = await seed_item("notification")
    url = f"/api/v1/notifications/{item.id}/decision"
    r = await authed.post(
        url, json={"decision": "respond", "request_id": "n-1", "interrupt_id": item.interrupt_id}
    )
    assert r.status_code == 202
    job = await _job(session, f"resume:{item.interrupt_id}")
    assert job.payload["resume"] == {"decision": "respond"}
    assert job.payload["review_item_id"] == str(item.id)
    await session.refresh(item)
    assert item.status == "resolved"

    bad = await authed.post(
        url, json={"decision": "approve", "request_id": "n-2", "interrupt_id": item.interrupt_id}
    )
    assert bad.status_code == 422
    # A question id is not a notification.
    question = await seed_item("question")
    wrong_kind = await authed.post(
        f"/api/v1/notifications/{question.id}/decision",
        json={"decision": "dismiss", "request_id": "n-3", "interrupt_id": question.interrupt_id},
    )
    assert wrong_kind.status_code == 404


async def _add_executed_proposal(session: AsyncSession, workflow_id: uuid.UUID) -> Proposal:
    """A second proposal on the run that was approved and already has an Action row."""
    owner = (await session.execute(select(Owner))).scalar_one()  # created by the login
    proposal = Proposal(workflow_id=workflow_id, kind="send_reply", status="approved")
    session.add(proposal)
    await session.flush()
    revision = ProposalRevision(
        proposal_id=proposal.id, revision=1, payload={}, payload_hash="h", author="model"
    )
    session.add(revision)
    await session.flush()
    approval = Approval(
        proposal_revision_id=revision.id, owner_id=owner.id, decision="approve", request_id="a-1"
    )
    session.add(approval)
    await session.flush()
    session.add(
        Action(
            operation_id=f"op-{proposal.id}",
            approval_id=approval.id,
            kind="send_reply",
            request_fingerprint="h",
            status="succeeded",
            provider_ref="mock-123",
        )
    )
    await session.commit()
    return proposal


async def test_cancel_marks_items_and_emits_event(
    authed: AsyncClient, session: AsyncSession, seed_proposal: SeedProposal
) -> None:
    seeded = await seed_proposal()
    executed = await _add_executed_proposal(session, seeded.workflow_id)

    url = f"/api/v1/workflows/{seeded.workflow_id}/cancel"
    r = await authed.post(url, json={"request_id": "c-1"})
    assert r.status_code == 202
    out: dict[str, Any] = r.json()
    assert out["status"] == "accepted"
    assert out["cancelled_review_items"] == [str(seeded.review_item_id)]
    assert out["cancelled_proposals"] == [str(seeded.proposal_id)]
    [completed] = out["completed_actions"]
    assert completed["proposal_id"] == str(executed.id)
    assert completed["status"] == "succeeded" and completed["provider_ref"] == "mock-123"

    item = await session.get_one(ReviewItem, seeded.review_item_id, populate_existing=True)
    proposal = await session.get_one(Proposal, seeded.proposal_id, populate_existing=True)
    kept = await session.get_one(Proposal, executed.id, populate_existing=True)
    run = await session.get_one(WorkflowRun, seeded.workflow_id, populate_existing=True)
    assert item.status == "cancelled" and item.resolved_at is not None
    assert proposal.status == "cancelled_before_execution"
    assert kept.status == "approved"
    assert run.status == "cancelled"
    events = (
        (
            await session.execute(
                select(WorkflowEvent).where(WorkflowEvent.workflow_id == seeded.workflow_id)
            )
        )
        .scalars()
        .all()
    )
    assert [e.event_type for e in events if e.event_type == "run_cancelled"] == ["run_cancelled"]
    job = await _job(session, f"resume:{seeded.interrupt_id}")
    assert job.payload["resume"] == {"decision": "cancelled"}
    assert job.payload["review_item_id"] == str(seeded.review_item_id)

    again = await authed.post(url, json={"request_id": "c-1"})
    assert again.status_code == 202 and again.json() == out
    other = await authed.post(url, json={"request_id": "c-2"})
    assert other.status_code == 409 and other.json()["reason"] == "already_cancelled"


async def test_owner_scoped_404_for_unknown_ids(authed: AsyncClient) -> None:
    unknown = uuid.uuid4()
    decision = {
        "decision": "approve",
        "proposal_revision": 1,
        "payload_hash": "h",
        "interrupt_id": "i",
        "request_id": "r",
    }
    requests = [
        ("GET", f"/api/v1/workflows/{unknown}", None),
        ("POST", f"/api/v1/workflows/{unknown}/cancel", {"request_id": "r"}),
        ("POST", f"/api/v1/proposals/{unknown}/decision", decision),
        (
            "POST",
            f"/api/v1/questions/{unknown}/answer",
            {"answer": "a", "request_id": "r", "interrupt_id": "i"},
        ),
        (
            "POST",
            f"/api/v1/notifications/{unknown}/decision",
            {"decision": "dismiss", "request_id": "r", "interrupt_id": "i"},
        ),
    ]
    for method, url, body in requests:
        r = await authed.request(method, url, json=body)
        assert r.status_code == 404, url
        assert r.json() == {"error": "not_found"}
    assert (await authed.get("/api/v1/workflows/not-a-uuid")).status_code == 422


async def test_concurrent_duplicate_cancels_return_original_outcome(
    sessionmaker: async_sessionmaker[AsyncSession], seed_proposal: SeedProposal
) -> None:
    seeded = await seed_proposal()

    async def cancel() -> CancelOutcome:
        async with sessionmaker() as s:
            return await cancel_workflow(s, seeded.workflow_id, request_id="same")

    first, second = await asyncio.gather(cancel(), cancel())
    assert first == second
    assert first.cancelled_review_items == [seeded.review_item_id]
    async with sessionmaker() as s:
        events = await s.scalar(
            select(func.count())
            .select_from(WorkflowEvent)
            .where(WorkflowEvent.event_type == "run_cancelled")
        )
    assert events == 1


async def test_concurrent_duplicate_answers_return_original_outcome(
    sessionmaker: async_sessionmaker[AsyncSession], seed_item: SeedItem
) -> None:
    item = await seed_item("question")
    assert item.interrupt_id is not None
    interrupt_id = item.interrupt_id

    async def answer() -> ReviewOutcome:
        async with sessionmaker() as s:
            return await answer_question(
                s, item.id, answer="Tuesday", request_id="same", interrupt_id=interrupt_id
            )

    first, second = await asyncio.gather(answer(), answer())
    assert first == second
    async with sessionmaker() as s:
        jobs = await s.scalar(select(func.count()).select_from(Job))
    assert jobs == 1
