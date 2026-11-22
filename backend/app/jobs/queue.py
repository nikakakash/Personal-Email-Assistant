"""Job queue writes. Only `enqueue` exists so far; claiming and leasing arrive with the worker."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job


async def enqueue(
    session: AsyncSession,
    kind: str,
    dedup_key: str,
    payload: dict[str, Any],
    serial_key: str | None = None,
) -> Job:
    """Insert a queued job, or return the existing row when `dedup_key` already exists.

    Flushes but does not commit; the caller owns the transaction.
    """
    stmt = (
        insert(Job)
        .values(kind=kind, dedup_key=dedup_key, payload=payload, serial_key=serial_key)
        .on_conflict_do_nothing(index_elements=[Job.dedup_key])
        .returning(Job)
    )
    job = (await session.execute(stmt)).scalar_one_or_none()
    if job is None:
        job = (await session.execute(select(Job).where(Job.dedup_key == dedup_key))).scalar_one()
    return job
