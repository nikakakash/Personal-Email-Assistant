"""API test fixtures: an app on the test DB, anonymous and logged-in clients, seeded rows."""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.proposals import create_proposal
from app.api.auth import CSRF_HEADER
from app.config import Settings
from app.db.models import EmailMessage, ReviewItem, WorkflowRun
from app.main import create_app

PASSPHRASE = "a-real-passphrase"  # matches `base_env`


@pytest.fixture
def settings(base_env: None) -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def app(settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]) -> FastAPI:
    return create_app(settings, sessionmaker=sessionmaker)


@pytest.fixture
async def anon(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def authed(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Logged in, sending the CSRF header on every request."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/auth/login", json={"passphrase": PASSPHRASE})
        assert r.status_code == 200, r.text
        c.headers[CSRF_HEADER] = r.json()["csrf_token"]
        yield c


@dataclass(frozen=True)
class SeededProposal:
    workflow_id: uuid.UUID
    proposal_id: uuid.UUID
    review_item_id: uuid.UUID
    revision: int
    payload_hash: str
    interrupt_id: str


WorkflowFactory = Callable[[], Awaitable[WorkflowRun]]


@pytest.fixture
def seed_proposal(
    session: AsyncSession, workflow_factory: WorkflowFactory
) -> Callable[[], Awaitable[SeededProposal]]:
    """A run with a pending send_reply proposal whose review item carries an interrupt id."""

    async def make() -> SeededProposal:
        run = await workflow_factory()
        payload: dict[str, Any] = {
            "provider_thread_id": run.provider_thread_id,
            "to": ["alice@example.com"],
            "cc": [],
            "subject": "Re: Lunch?",
            "body_text": "Tuesday works.",
        }
        proposal, revision = await create_proposal(
            session, run.id, "send_reply", payload, None, None, idempotency_key=f"{run.id}:1"
        )
        item = (
            await session.execute(select(ReviewItem).where(ReviewItem.proposal_id == proposal.id))
        ).scalar_one()
        interrupt_id = f"int-{uuid.uuid4().hex[:8]}"
        item.interrupt_id = interrupt_id
        await session.commit()
        return SeededProposal(
            run.id, proposal.id, item.id, revision.revision, revision.payload_hash, interrupt_id
        )

    return make


@pytest.fixture
def seed_item(
    session: AsyncSession, workflow_factory: WorkflowFactory
) -> Callable[..., Awaitable[ReviewItem]]:
    """A run plus one pending question or notification review item with an interrupt id."""

    async def make(
        kind: str = "question", *, subject: str = "Hello", status: str = "pending"
    ) -> ReviewItem:
        run = await workflow_factory()
        run.classification = "notify" if kind == "notification" else "respond"
        run.triage_reason = f"Reason for {subject}"
        message = await session.get_one(EmailMessage, run.message_id)
        message.subject = subject
        item = ReviewItem(
            workflow_id=run.id,
            kind=kind,
            status=status,
            interrupt_id=f"int-{uuid.uuid4().hex[:8]}",
            payload={"question": "Which day?"} if kind == "question" else {},
        )
        session.add(item)
        await session.commit()
        return item

    return make
