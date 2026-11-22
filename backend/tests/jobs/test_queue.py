"""The Postgres job queue: claims, leases, serial keys and compare-and-set completion."""

import asyncio
import uuid
from datetime import timedelta

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Job
from app.jobs.queue import claim, complete, enqueue, fail, heartbeat

Session = async_sessionmaker[AsyncSession]
LEASE = timedelta(seconds=60)


async def _enqueue(sessionmaker: Session, key: str, serial_key: str | None = None) -> uuid.UUID:
    async with sessionmaker() as s, s.begin():
        job = await enqueue(s, "start_workflow", key, {"n": key}, serial_key=serial_key)
        return job.id


async def _claim(sessionmaker: Session, worker_id: str) -> Job | None:
    async with sessionmaker() as s, s.begin():
        return await claim(s, worker_id, LEASE)


async def _expire_lease(sessionmaker: Session, job_id: uuid.UUID) -> None:
    async with sessionmaker() as s, s.begin():
        await s.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(lease_until=func.now() - timedelta(seconds=1))
        )


async def _job(sessionmaker: Session, job_id: uuid.UUID) -> Job:
    async with sessionmaker() as s:
        return await s.get_one(Job, job_id)


async def test_two_workers_cannot_claim_same_job(sessionmaker: Session) -> None:
    job_id = await _enqueue(sessionmaker, "only")

    results = await asyncio.gather(_claim(sessionmaker, "w1"), _claim(sessionmaker, "w2"))

    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1
    job = claimed[0]
    assert job.id == job_id
    assert (job.state, job.attempts) == ("running", 1)
    assert job.lease_owner in ("w1", "w2")
    assert job.lease_until is not None and job.heartbeat_at is not None
    assert job.lease_until - job.heartbeat_at == LEASE  # both from the DB clock


async def test_expired_lease_is_reclaimable(sessionmaker: Session) -> None:
    job_id = await _enqueue(sessionmaker, "crashy")
    assert (await _claim(sessionmaker, "w1")) is not None
    assert await _claim(sessionmaker, "w2") is None  # the lease is live

    await _expire_lease(sessionmaker, job_id)  # w1 died without a heartbeat
    job = await _claim(sessionmaker, "w2")

    assert job is not None and job.id == job_id
    assert (job.lease_owner, job.attempts) == ("w2", 2)


async def test_serial_key_blocks_second_job_on_thread(sessionmaker: Session) -> None:
    first = await _enqueue(sessionmaker, "a1", serial_key="thread-a")
    second = await _enqueue(sessionmaker, "a2", serial_key="thread-a")
    other = await _enqueue(sessionmaker, "b1", serial_key="thread-b")

    claimed_first = await _claim(sessionmaker, "w1")
    claimed_other = await _claim(sessionmaker, "w2")

    assert claimed_first is not None and claimed_first.id == first
    assert claimed_other is not None and claimed_other.id == other  # thread-a is busy
    assert await _claim(sessionmaker, "w3") is None

    async with sessionmaker() as s, s.begin():
        assert await complete(s, first, "w1")
    claimed_second = await _claim(sessionmaker, "w3")
    assert claimed_second is not None and claimed_second.id == second


async def test_lost_lease_does_not_overwrite(sessionmaker: Session) -> None:
    job_id = await _enqueue(sessionmaker, "contested")
    await _claim(sessionmaker, "w1")
    await _expire_lease(sessionmaker, job_id)
    assert await _claim(sessionmaker, "w2") is not None  # w2 now owns the job

    async with sessionmaker() as s, s.begin():
        assert not await heartbeat(s, job_id, "w1", LEASE)
        assert not await complete(s, job_id, "w1")
        assert not await fail(s, job_id, "w1", "internal_error")

    job = await _job(sessionmaker, job_id)
    assert (job.state, job.lease_owner, job.error) == ("running", "w2", None)


async def test_fail_with_retry_requeues_later(sessionmaker: Session) -> None:
    job_id = await _enqueue(sessionmaker, "retry")
    await _claim(sessionmaker, "w1")

    async with sessionmaker() as s, s.begin():
        assert await fail(s, job_id, "w1", "internal_error", retry_in=timedelta(minutes=5))

    job = await _job(sessionmaker, job_id)
    assert (job.state, job.error, job.lease_owner) == ("queued", "internal_error", None)
    assert await _claim(sessionmaker, "w1") is None  # not due yet
