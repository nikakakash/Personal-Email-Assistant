"""MockMailProvider: thread snapshots, send faults and idempotent outbox lookups."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import EmailMessage, EmailThread, MockOutbox
from app.providers.interfaces import OutgoingReply, ProviderTimeout
from app.providers.mock_mail import MockFaults, MockMailProvider

BASE = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)


async def _add_message(session: AsyncSession, thread_id: str, n: int) -> None:
    thread = (
        await session.execute(
            select(EmailThread).where(EmailThread.provider_thread_id == thread_id)
        )
    ).scalar_one_or_none()
    if thread is None:
        thread = EmailThread(account_id="mock-account", provider_thread_id=thread_id)
        session.add(thread)
        await session.flush()
    session.add(
        EmailMessage(
            thread_id=thread.id,
            account_id="mock-account",
            provider_message_id=f"{thread_id}-m{n}",
            sender="alice@example.com",
            to_addrs=["owner@example.com"],
            subject="Hi",
            body_text=f"message {n}",
            received_at=BASE + timedelta(minutes=n),
            content_hash=f"h{n}",
        )
    )
    await session.commit()


def _reply() -> OutgoingReply:
    return OutgoingReply(
        provider_thread_id="t1",
        in_reply_to="<a@example.com>",
        references=["<a@example.com>"],
        to=["alice@example.com"],
        cc=[],
        subject="Re: Hi",
        body_text="Thanks",
    )


async def _outbox_count(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(MockOutbox)) or 0)


async def test_get_thread_version_changes_when_message_added(
    session: AsyncSession, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    provider = MockMailProvider(sessionmaker, MockFaults())
    await _add_message(session, "t1", 1)
    first = await provider.get_thread("t1")
    await _add_message(session, "t1", 2)
    second = await provider.get_thread("t1")
    assert [m.provider_message_id for m in first.messages] == ["t1-m1"]
    assert [m.provider_message_id for m in second.messages] == ["t1-m1", "t1-m2"]
    assert first.version != second.version


async def test_get_thread_limit_keeps_most_recent_in_order(
    session: AsyncSession, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    provider = MockMailProvider(sessionmaker, MockFaults())
    for n in range(1, 5):
        await _add_message(session, "t1", n)
    snap = await provider.get_thread("t1", limit=2)
    assert [m.provider_message_id for m in snap.messages] == ["t1-m3", "t1-m4"]


async def test_get_thread_unknown_raises_key_error(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(KeyError):
        await MockMailProvider(sessionmaker, MockFaults()).get_thread("nope")


async def test_send_ok_confirms_and_is_idempotent(
    session: AsyncSession, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    provider = MockMailProvider(sessionmaker, MockFaults())
    first = await provider.send_reply(_reply(), "op-1")
    second = await provider.send_reply(_reply(), "op-1")
    assert first.status == second.status == "confirmed"
    assert await _outbox_count(session) == 1


async def test_send_fail_is_definite_and_writes_nothing(
    session: AsyncSession, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    provider = MockMailProvider(sessionmaker, MockFaults(send="fail"))
    result = await provider.send_reply(_reply(), "op-1")
    assert result.status == "failed_definite"
    assert await _outbox_count(session) == 0


async def test_send_timeout_before_accept_writes_nothing(
    session: AsyncSession, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    provider = MockMailProvider(sessionmaker, MockFaults(send="timeout_before_accept"))
    with pytest.raises(ProviderTimeout):
        await provider.send_reply(_reply(), "op-1")
    assert await _outbox_count(session) == 0
    assert await provider.find_sent("op-1") is None


async def test_send_timeout_after_accept_leaves_outbox_row_and_raises(
    session: AsyncSession, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    provider = MockMailProvider(sessionmaker, MockFaults(send="timeout_after_accept"))
    with pytest.raises(ProviderTimeout):
        await provider.send_reply(_reply(), "op-1")
    assert await _outbox_count(session) == 1


async def test_find_sent_returns_confirmed_after_timeout_after_accept(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    provider = MockMailProvider(sessionmaker, MockFaults(send="timeout_after_accept"))
    with pytest.raises(ProviderTimeout):
        await provider.send_reply(_reply(), "op-1")
    found = await provider.find_sent("op-1")
    assert found is not None and found.status == "confirmed"
    assert await provider.find_sent("other") is None


async def test_faults_are_held_by_reference(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    faults = MockFaults()
    provider = MockMailProvider(sessionmaker, faults)
    faults.send = "fail"
    assert (await provider.send_reply(_reply(), "op-1")).status == "failed_definite"
