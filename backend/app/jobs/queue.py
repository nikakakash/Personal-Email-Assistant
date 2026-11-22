"""The Postgres job queue: enqueue, claim with a lease, heartbeat, complete, fail, reschedule.

A job is a row in `jobs`. Its life: `queued` → (claim) `running` → `done` or `failed`, or back
to `queued` with a later `scheduled_at` for a retry. A worker that claims a job holds a *lease*
(`lease_owner`, `lease_until`) and renews it with heartbeats while it works. If the worker dies,
the lease runs out and any worker may claim the job again: that is crash recovery, and it is
why every job handler must be safe to run twice.

Leases use the database clock (`now()`) only, never the worker's clock, so workers on machines
with skewed clocks still agree on whether a lease is live.

`complete`, `fail`, `reschedule` and `heartbeat` are compare-and-set on `lease_owner`: a worker
whose lease expired (and whose job another worker has since claimed) changes nothing and gets
`False`.

None of these functions commit; the caller owns the transaction (as with `enqueue`). Commit a
claim right away, so the row lock is held only for the claim itself.
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, exists, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import Job

# Key of the transaction-level advisory lock that serializes claims (see `claim`).
_CLAIM_LOCK_KEY = 0x6A6F6273  # "jobs"
# A job claimed this many times without finishing is failed instead of claimed again.
MAX_JOB_ATTEMPTS = 5

Exhausted = Callable[[AsyncSession, Job], Awaitable[None]]


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


async def claim(
    session: AsyncSession,
    worker_id: str,
    lease: timedelta,
    *,
    on_exhausted: Exhausted | None = None,
) -> Job | None:
    """Claim the next runnable job for `worker_id`, or return None.

    Runnable: `queued` and due (`scheduled_at <= now()`), or `running` with an expired lease
    (its worker died). A job is skipped while another job with the same `serial_key` (the
    conversation) is `running` under a live lease, so one thread's jobs never run at once.

    A runnable job already claimed `MAX_JOB_ATTEMPTS` times is not claimed again: it is marked
    `failed` (`max_attempts`), `on_exhausted(session, job)` runs in the same transaction (the
    worker fails the job's run there), and the next runnable job is tried. This stops a job
    that crashes its worker every time from being retried forever.

    `FOR UPDATE SKIP LOCKED` makes concurrent claimers skip a row another transaction is
    claiming instead of waiting for it. The serial-key check reads *other* rows, though, and
    two transactions could each see "nothing running" for the same thread before either
    commits; a transaction-level advisory lock taken first makes claims one at a time, so
    each sees the previous claim committed. Claims are a single short query, so this costs
    nothing for a single-owner app.
    """
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _CLAIM_LOCK_KEY})
    other = aliased(Job)
    thread_busy = exists().where(
        other.serial_key == Job.serial_key,
        other.id != Job.id,
        other.state == "running",
        other.lease_until >= func.now(),
    )
    candidate = (
        select(Job.id, Job.attempts)
        .where(
            or_(
                and_(Job.state == "queued", Job.scheduled_at <= func.now()),
                and_(Job.state == "running", Job.lease_until < func.now()),
            ),
            or_(Job.serial_key.is_(None), ~thread_busy),
        )
        .order_by(Job.scheduled_at, Job.id)
        .limit(1)
        .with_for_update(skip_locked=True, of=Job)
    )
    while True:
        row = (await session.execute(candidate)).one_or_none()
        if row is None:
            return None
        job_id, attempts = row
        if attempts < MAX_JOB_ATTEMPTS:
            break
        exhausted = (
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(state="failed", error="max_attempts", lease_owner=None, lease_until=None)
                .returning(Job)
                .execution_options(populate_existing=True)
            )
        ).scalar_one()
        if on_exhausted is not None:
            await on_exhausted(session, exhausted)
    stmt = (
        update(Job)
        .where(Job.id == job_id)
        .values(
            state="running",
            lease_owner=worker_id,
            lease_until=func.now() + lease,
            heartbeat_at=func.now(),
            attempts=Job.attempts + 1,
        )
        .returning(Job)
        .execution_options(populate_existing=True)
    )
    return (await session.execute(stmt)).scalar_one()


def _owned(job_id: uuid.UUID, worker_id: str) -> Any:
    """SQL condition: the job is running under this worker's lease (the compare-and-set)."""
    return and_(Job.id == job_id, Job.state == "running", Job.lease_owner == worker_id)


async def heartbeat(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, lease: timedelta
) -> bool:
    """Extend the lease to `now() + lease`; False if this worker no longer owns the job."""
    result = await session.execute(
        update(Job)
        .where(_owned(job_id, worker_id))
        .values(lease_until=func.now() + lease, heartbeat_at=func.now())
        .returning(Job.id)
    )
    return result.scalar_one_or_none() is not None


async def complete(session: AsyncSession, job_id: uuid.UUID, worker_id: str) -> bool:
    """Mark the job `done`; False (and nothing changed) if the lease was lost."""
    result = await session.execute(
        update(Job)
        .where(_owned(job_id, worker_id))
        .values(state="done", lease_owner=None, lease_until=None, error=None)
        .returning(Job.id)
    )
    return result.scalar_one_or_none() is not None


async def fail(
    session: AsyncSession,
    job_id: uuid.UUID,
    worker_id: str,
    error: str,
    retry_in: timedelta | None = None,
) -> bool:
    """Record `error` (a fixed code, never exception text). With `retry_in` the job is queued
    again that much later; without it the job is `failed` for good. False if the lease was lost.
    """
    values: dict[str, Any] = {"error": error, "lease_owner": None, "lease_until": None}
    if retry_in is None:
        values["state"] = "failed"
    else:
        values |= {"state": "queued", "scheduled_at": func.now() + retry_in}
    result = await session.execute(
        update(Job).where(_owned(job_id, worker_id)).values(**values).returning(Job.id)
    )
    return result.scalar_one_or_none() is not None


async def reschedule(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, delay: timedelta
) -> bool:
    """Queue the job again `delay` from now without recording a failure.

    `attempts` keeps the count the claim set. False if the lease was lost.
    """
    result = await session.execute(
        update(Job)
        .where(_owned(job_id, worker_id))
        .values(
            state="queued",
            scheduled_at=func.now() + delay,
            lease_owner=None,
            lease_until=None,
            error=None,
        )
        .returning(Job.id)
    )
    return result.scalar_one_or_none() is not None
