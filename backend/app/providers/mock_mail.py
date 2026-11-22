"""Mock mail provider backed by Postgres (`email_messages` read, `mock_outbox` written).

`MockFaults` is a plain mutable dataclass held by reference, so a dev endpoint can flip a
fault at runtime and the provider sees it on the next call.
"""

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.hashing import payload_hash
from app.db.models import EmailMessage, EmailThread, MockOutbox
from app.providers.interfaces import (
    NormalizedMessage,
    OutgoingReply,
    ProviderResult,
    ProviderTimeout,
    ThreadSnapshot,
)

FaultMode = Literal["ok", "fail", "timeout_before_accept", "timeout_after_accept"]


@dataclass
class MockFaults:
    send: FaultMode = "ok"
    calendar_create: FaultMode = "ok"


class MockMailProvider:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        faults: MockFaults,
        account_id: str = "mock-account",
    ) -> None:
        self._sessionmaker = sessionmaker
        self.faults = faults
        self._account_id = account_id

    async def get_thread(self, provider_thread_id: str, limit: int = 20) -> ThreadSnapshot:
        async with self._sessionmaker() as session:
            thread = (
                await session.execute(
                    select(EmailThread).where(
                        EmailThread.account_id == self._account_id,
                        EmailThread.provider_thread_id == provider_thread_id,
                    )
                )
            ).scalar_one_or_none()
            if thread is None:
                raise KeyError(provider_thread_id)
            rows = (
                (
                    await session.execute(
                        select(EmailMessage)
                        .where(EmailMessage.thread_id == thread.id)
                        .order_by(
                            EmailMessage.received_at.desc(),
                            EmailMessage.provider_message_id.desc(),
                        )
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        messages = [
            NormalizedMessage(
                provider_message_id=r.provider_message_id,
                provider_thread_id=provider_thread_id,
                rfc_message_id=r.rfc_message_id,
                sender=r.sender,
                to=list(r.to_addrs),
                cc=list(r.cc_addrs),
                subject=r.subject,
                body_text=r.body_text or None,
                body_html=r.body_html,
                received_at=r.received_at,
            )
            for r in reversed(rows)
        ]
        version = payload_hash({"ids": [m.provider_message_id for m in messages]})
        return ThreadSnapshot(
            provider_thread_id=provider_thread_id, messages=messages, version=version
        )

    async def send_reply(self, req: OutgoingReply, operation_id: str) -> ProviderResult:
        mode = self.faults.send
        if mode == "timeout_before_accept":
            raise ProviderTimeout("send timed out before the provider accepted it")
        if mode == "fail":
            return ProviderResult(status="failed_definite", error="mock send failure")
        async with self._sessionmaker() as session:
            await session.execute(
                insert(MockOutbox)
                .values(operation_id=operation_id, payload=req.model_dump(mode="json"))
                .on_conflict_do_nothing(index_elements=["operation_id"])
            )
            await session.commit()
        if mode == "timeout_after_accept":
            raise ProviderTimeout("send timed out after the provider accepted it")
        return ProviderResult(status="confirmed", provider_ref=operation_id)

    async def find_sent(self, operation_id: str) -> ProviderResult | None:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(MockOutbox.operation_id).where(MockOutbox.operation_id == operation_id)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return ProviderResult(status="confirmed", provider_ref=operation_id)
