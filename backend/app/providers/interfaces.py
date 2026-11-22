"""Provider contracts: normalized data models and the mail/calendar Protocols.

The workflow and executor only ever talk to these Protocols. The mock providers (Postgres
backed) implement them now; Gmail and Google Calendar adapters implement them later.
"""

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, Field

from app.scheduling.slots import Interval


class ProviderTimeout(Exception):  # noqa: N818 - name fixed by the contract
    """The provider call timed out; the outcome of the operation is unknown."""


class ThreadNotFound(KeyError):
    """`get_thread` found no such thread. A KeyError subclass, so older `except KeyError`
    callers keep working while new code can catch exactly this."""


class NormalizedMessage(BaseModel):
    provider_message_id: str
    provider_thread_id: str
    rfc_message_id: str | None = None
    sender: str
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str = ""
    body_text: str | None = None
    body_html: str | None = None
    received_at: AwareDatetime
    # Auto-Submitted, List-Id and List-Unsubscribe, when present.
    headers: dict[str, str] = Field(default_factory=dict)


class ThreadSnapshot(BaseModel):
    provider_thread_id: str
    messages: list[NormalizedMessage]
    version: str


class OutgoingReply(BaseModel):
    provider_thread_id: str
    in_reply_to: str
    references: list[str] = Field(default_factory=list)
    to: list[str]
    cc: list[str] = Field(default_factory=list)
    subject: str
    body_text: str


class EventRequest(BaseModel):
    calendar_id: str = "primary"
    title: str
    start: AwareDatetime
    end: AwareDatetime
    time_zone: str
    attendees: list[str] = Field(default_factory=list)
    location: str | None = None
    send_updates: Literal["all", "none"] = "none"


class ProviderResult(BaseModel):
    status: Literal["confirmed", "failed_definite", "unknown"]
    provider_ref: str | None = None
    error: str | None = None


class FreeBusy(BaseModel):
    busy: list[Interval]
    retrieved_at: datetime

    model_config = {"arbitrary_types_allowed": True}


class MailProvider(Protocol):
    async def get_thread(self, provider_thread_id: str, limit: int = 20) -> ThreadSnapshot: ...

    async def send_reply(self, req: OutgoingReply, operation_id: str) -> ProviderResult: ...

    async def find_sent(self, operation_id: str) -> ProviderResult | None: ...


class CalendarProvider(Protocol):
    async def get_freebusy(self, window: Interval, time_zone: str) -> FreeBusy: ...

    async def create_event(self, req: EventRequest, event_id: str) -> ProviderResult: ...

    async def get_event(self, event_id: str) -> dict[str, Any] | None: ...
