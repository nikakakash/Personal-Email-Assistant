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

# Loop risks, dropped before triage: bounces and auto-replies (spec: suppress reply loops and
# bounces). Other automated mail (no-reply senders, generated alerts, lists) is still triaged so
# the owner can be notified, but is flagged `automated` and never answered (see triage).
_BOUNCE_PREFIXES = ("mailer-daemon@", "postmaster@")
_NO_REPLY_PREFIXES = ("no-reply@", "noreply@", "donotreply@", "do-not-reply@")


class IngestResult(BaseModel):
    status: Literal["queued", "duplicate", "skipped_owner", "skipped_automated", "paused"]
    workflow_job_id: UUID | None = None


def _sender(msg: NormalizedMessage) -> str:
    try:
        return normalize_address(msg.sender)
    except ValueError:
        return msg.sender.strip().lower()


def _headers(msg: NormalizedMessage) -> dict[str, str]:
    return {k.lower(): v.strip().lower() for k, v in msg.headers.items()}


def _is_loop_risk(msg: NormalizedMessage) -> bool:
    """A bounce or an auto-reply: answering or even notifying would only feed a loop."""
    return (
        _sender(msg).startswith(_BOUNCE_PREFIXES)
        or _headers(msg).get("auto-submitted") == "auto-replied"
    )


def _is_automated(msg: NormalizedMessage) -> bool:
    """Machine-sent mail that must never get a reply (it may still deserve a notification)."""
    headers = _headers(msg)
    auto = headers.get("auto-submitted")
    return (
        _sender(msg).startswith(_NO_REPLY_PREFIXES)
        or (auto is not None and auto != "no")
        or "list-id" in headers
    )


async def _mark_superseded(session: AsyncSession, provider_thread_id: str) -> None:
    for item in await repo.pending_proposal_items(session, provider_thread_id):
        if item.proposal_id is not None:
            proposal = await session.get(Proposal, item.proposal_id, with_for_update=True)
            # Only a proposal still awaiting approval can be superseded; never overwrite
            # approved, executed or any other status.
            if proposal is None or proposal.status != "awaiting_approval":
                continue
            proposal.status = "stale"
        item.status = "stale"
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
    index: bool = False,
) -> IngestResult:
    """Store `msg` and queue its workflow. With `index`, also queue `index_message` for the
    retrieval index (owner and ordinary inbound mail; never bounces or automated mail)."""
    if await repo.ingestion_paused(session):
        return IngestResult(status="paused")

    thread = await repo.upsert_thread(session, account_id, msg.provider_thread_id)
    content_hash = payload_hash(
        {"subject": msg.subject, "text": msg.body_text, "html": msg.body_html}
    )
    message_id = await repo.insert_message(session, thread, msg, account_id, content_hash)
    if message_id is None:
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

    loop_risk, automated = _is_loop_risk(msg), _is_automated(msg)
    if index and not loop_risk and not automated:
        await enqueue(
            session,
            "index_message",
            f"{account_id}:{msg.provider_message_id}:index_message",
            # Only the row id: the worker must not mistake this for a workflow job.
            {"message_id": str(message_id)},
        )

    if is_owner_address(msg.sender, owner_identities):
        # A manual reply from the owner (e.g. in Gmail) invalidates pending proposals,
        # but never starts a workflow.
        await _mark_superseded(session, msg.provider_thread_id)
        return IngestResult(status="skipped_owner")
    if loop_risk:
        return IngestResult(status="skipped_automated")

    if not automated:  # a list post or an alert does not invalidate a pending reply
        await _mark_superseded(session, msg.provider_thread_id)
    job = await enqueue(
        session,
        "start_workflow",
        f"{account_id}:{msg.provider_message_id}:start_workflow",
        {
            "account_id": account_id,
            "provider_message_id": msg.provider_message_id,
            "automated": automated,
        },
        serial_key=msg.provider_thread_id,
    )
    return IngestResult(status="queued", workflow_job_id=job.id)
