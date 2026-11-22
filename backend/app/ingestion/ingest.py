"""Ingestion: store an incoming message, filter it, supersede stale proposals, queue a workflow.

The caller owns the transaction: this flushes but never commits.
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.hashing import payload_hash
from app.db import repo
from app.db.models import EmailThread, Proposal
from app.jobs.queue import enqueue
from app.mail.addresses import is_owner_address, normalize_address
from app.providers.interfaces import NormalizedMessage

_AUTOMATED_PREFIXES = ("mailer-daemon@", "no-reply@", "noreply@")


class IngestResult(BaseModel):
    status: Literal["queued", "duplicate", "skipped_owner", "skipped_automated", "paused"]
    workflow_job_id: UUID | None = None


def _is_automated(msg: NormalizedMessage) -> bool:
    try:
        sender = normalize_address(msg.sender)
    except ValueError:
        sender = msg.sender.strip().lower()
    if sender.startswith(_AUTOMATED_PREFIXES):
        return True
    headers = {k.lower(): v for k, v in msg.headers.items()}
    auto = headers.get("auto-submitted")
    if auto is not None and auto.strip().lower() != "no":
        return True
    return "list-id" in headers


async def _mark_superseded(session: AsyncSession, provider_thread_id: str) -> None:
    for item in await repo.pending_proposal_items(session, provider_thread_id):
        item.status = "stale"
        if item.proposal_id is not None:
            await session.execute(
                update(Proposal).where(Proposal.id == item.proposal_id).values(status="stale")
            )
        await enqueue(
            session,
            "resume_workflow",
            f"resume:superseded:{item.id}",
            {
                "workflow_id": str(item.workflow_id),
                "review_item_id": str(item.id),
                "interrupt_id": item.interrupt_id,
                "resume": {"decision": "superseded"},
            },
            serial_key=provider_thread_id,
        )


async def ingest_message(
    session: AsyncSession,
    msg: NormalizedMessage,
    owner_identities: list[str],
    *,
    account_id: str = "mock-account",
) -> IngestResult:
    if await repo.ingestion_paused(session):
        return IngestResult(status="paused")

    thread = await repo.upsert_thread(session, account_id, msg.provider_thread_id)
    content_hash = payload_hash(
        {"subject": msg.subject, "text": msg.body_text, "html": msg.body_html}
    )
    if not await repo.insert_message(session, thread, msg, account_id, content_hash):
        return IngestResult(status="duplicate")

    ids = await repo.thread_message_ids(session, thread.id)
    await session.execute(
        update(EmailThread)
        .where(EmailThread.id == thread.id)
        .values(
            latest_message_id=ids[-1],
            source_version=payload_hash({"ids": ids}),
            updated_at=datetime.now(UTC),
        )
    )

    if is_owner_address(msg.sender, owner_identities):
        return IngestResult(status="skipped_owner")
    if _is_automated(msg):
        return IngestResult(status="skipped_automated")

    await _mark_superseded(session, msg.provider_thread_id)
    job = await enqueue(
        session,
        "start_workflow",
        f"{account_id}:{msg.provider_message_id}:start_workflow",
        {"account_id": account_id, "provider_message_id": msg.provider_message_id},
        serial_key=msg.provider_thread_id,
    )
    return IngestResult(status="queued", workflow_job_id=job.id)
