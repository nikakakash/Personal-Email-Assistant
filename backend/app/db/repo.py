"""Small query helpers shared by ingestion and the mock providers.

None of these commit; the caller owns the transaction.
"""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AppFlags, EmailMessage, EmailThread, ReviewItem, WorkflowRun
from app.providers.interfaces import NormalizedMessage


async def ingestion_paused(session: AsyncSession) -> bool:
    """Read app_flags id=1; a missing row means not paused."""
    flags = await session.get(AppFlags, 1)
    return bool(flags and flags.ingestion_paused)


async def upsert_thread(
    session: AsyncSession, account_id: str, provider_thread_id: str
) -> EmailThread:
    stmt = (
        insert(EmailThread)
        .values(account_id=account_id, provider_thread_id=provider_thread_id)
        .on_conflict_do_nothing(index_elements=["account_id", "provider_thread_id"])
    )
    await session.execute(stmt)
    return (
        await session.execute(
            select(EmailThread).where(
                EmailThread.account_id == account_id,
                EmailThread.provider_thread_id == provider_thread_id,
            )
        )
    ).scalar_one()


async def insert_message(
    session: AsyncSession,
    thread: EmailThread,
    msg: NormalizedMessage,
    account_id: str,
    content_hash: str,
) -> bool:
    """Insert the message; return False when (account, provider_message_id) already exists."""
    stmt = (
        insert(EmailMessage)
        .values(
            thread_id=thread.id,
            account_id=account_id,
            provider_message_id=msg.provider_message_id,
            rfc_message_id=msg.rfc_message_id,
            sender=msg.sender,
            to_addrs=msg.to,
            cc_addrs=msg.cc,
            subject=msg.subject,
            body_text=msg.body_text or "",
            body_html=msg.body_html,
            received_at=msg.received_at,
            content_hash=content_hash,
        )
        .on_conflict_do_nothing(index_elements=["account_id", "provider_message_id"])
        .returning(EmailMessage.id)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def thread_message_ids(session: AsyncSession, thread_id: object) -> list[str]:
    """Provider message ids of a thread in chronological order."""
    rows = await session.execute(
        select(EmailMessage.provider_message_id)
        .where(EmailMessage.thread_id == thread_id)
        .order_by(EmailMessage.received_at, EmailMessage.provider_message_id)
    )
    return list(rows.scalars())


async def pending_proposal_items(
    session: AsyncSession, provider_thread_id: str
) -> list[ReviewItem]:
    """Pending proposal review items on this provider thread, row-locked.

    The lock serializes stale-marking with a concurrent approval of the same item.
    """
    rows = await session.execute(
        select(ReviewItem)
        .join(WorkflowRun, WorkflowRun.id == ReviewItem.workflow_id)
        .where(
            WorkflowRun.provider_thread_id == provider_thread_id,
            ReviewItem.kind == "proposal",
            ReviewItem.status == "pending",
        )
        .order_by(ReviewItem.created_at)
        .with_for_update(of=ReviewItem)
    )
    return list(rows.scalars())
