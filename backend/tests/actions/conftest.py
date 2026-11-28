"""Fixtures for proposal, decision and executor tests: a real thread, valid payloads, mocks."""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.proposals import create_proposal
from app.db.models import EmailMessage, EmailThread, Owner, ReviewItem, WorkflowRun
from app.graphs.parent import GRAPH_VERSION
from app.providers.mock_calendar import MockCalendarProvider
from app.providers.mock_mail import MockFaults, MockMailProvider


@dataclass(frozen=True)
class Seeded:
    """Plain values (not ORM rows), so they survive the rollbacks `decide_proposal` performs."""

    run_id: uuid.UUID
    proposal_id: uuid.UUID
    revision: int
    payload_hash: str
    payload: dict[str, Any]
    review_item_id: uuid.UUID
    interrupt_id: str


def reply_payload(thread_id: str, rfc_id: str) -> dict[str, Any]:
    return {
        "provider_thread_id": thread_id,
        "in_reply_to": rfc_id,
        "references": [rfc_id],
        "to": ["alice@example.com"],
        "cc": [],
        "subject": "Re: Lunch?",
        "body_text": "Tuesday works for me.",
        "evidence_refs": [f"thread:{thread_id}"],
    }


def event_payload() -> dict[str, Any]:
    return {
        "calendar_id": "primary",
        "title": "Lunch with Alice",
        "start": "2026-11-25T11:00:00+00:00",
        "end": "2026-11-25T12:00:00+00:00",
        "time_zone": "Europe/Berlin",
        "attendees": ["alice@example.com"],
        "location": None,
        "send_updates": "none",
        "evidence_refs": ["availability:x"],
    }


SeedFactory = Callable[..., Awaitable[Seeded]]


@pytest.fixture
def owner_id(owner: Owner) -> uuid.UUID:
    """The owner's id as a plain value (ORM rows expire when `decide_proposal` rolls back)."""
    return owner.id


@pytest.fixture
def faults() -> MockFaults:
    return MockFaults()


@pytest.fixture
def mail(sessionmaker: async_sessionmaker[AsyncSession], faults: MockFaults) -> MockMailProvider:
    return MockMailProvider(sessionmaker, faults)


@pytest.fixture
def calendar(
    sessionmaker: async_sessionmaker[AsyncSession], faults: MockFaults
) -> MockCalendarProvider:
    return MockCalendarProvider(sessionmaker, None, "Europe/Berlin", faults)


@pytest.fixture
def seed(session: AsyncSession) -> SeedFactory:
    """Thread + message + run, then `create_proposal`; the review item gets an interrupt id."""

    async def make(kind: Literal["send_reply", "create_event"] = "send_reply") -> Seeded:
        n = uuid.uuid4().hex[:8]
        thread_id, rfc_id = f"t-{n}", f"<m-{n}@example.com>"
        thread = EmailThread(account_id="mock-account", provider_thread_id=thread_id)
        session.add(thread)
        await session.flush()
        message = EmailMessage(
            thread_id=thread.id,
            account_id="mock-account",
            provider_message_id=f"m-{n}",
            rfc_message_id=rfc_id,
            sender="alice@example.com",
            to_addrs=["owner@example.com"],
            subject="Lunch?",
            body_text="Lunch on Tuesday?",
            received_at=datetime(2026, 11, 20, 9, 0, tzinfo=UTC),
            content_hash=f"h-{n}",
        )
        session.add(message)
        await session.flush()
        run = WorkflowRun(
            graph_thread_id=f"mock-account:m-{n}:{GRAPH_VERSION}",
            message_id=message.id,
            provider_thread_id=thread_id,
            graph_version="test",
        )
        session.add(run)
        await session.flush()
        payload = reply_payload(thread_id, rfc_id) if kind == "send_reply" else event_payload()
        proposal, revision = await create_proposal(
            session,
            run.id,
            kind,
            payload,
            source_version="sv1",
            preference_version="pv1",
            idempotency_key=f"{run.id}:1",
        )
        item = (
            await session.execute(select(ReviewItem).where(ReviewItem.proposal_id == proposal.id))
        ).scalar_one()
        interrupt_id = f"int-{n}"
        item.interrupt_id = interrupt_id
        await session.commit()
        return Seeded(
            run_id=run.id,
            proposal_id=proposal.id,
            revision=revision.revision,
            payload_hash=revision.payload_hash,
            payload=dict(revision.payload),
            review_item_id=item.id,
            interrupt_id=interrupt_id,
        )

    return make
