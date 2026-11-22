"""Mock calendar provider backed by Postgres (`mock_calendar_events`).

Busy time is deterministic: Monday to Friday 12:00-13:00 and 15:00-16:00 local time, unless
`busy_by_date` gives explicit ranges for that local date (which then replace the default
pattern for it). Events created through `create_event` are busy time too.
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import MockCalendarEvent
from app.providers.interfaces import EventRequest, FreeBusy, ProviderResult, ProviderTimeout
from app.providers.mock_mail import MockFaults
from app.scheduling.slots import Interval

DEFAULT_BUSY: list[tuple[time, time]] = [(time(12), time(13)), (time(15), time(16))]


class MockCalendarProvider:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        busy_by_date: dict[date, list[tuple[time, time]]] | None,
        time_zone: str,
        faults: MockFaults,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._busy_by_date = busy_by_date or {}
        self._time_zone = time_zone
        self.faults = faults

    async def get_freebusy(self, window: Interval, time_zone: str | None = None) -> FreeBusy:
        zone = ZoneInfo(time_zone or self._time_zone)
        busy: list[Interval] = []
        day = window.start.astimezone(zone).date()
        last = window.end.astimezone(zone).date()
        while day <= last:
            ranges = self._busy_by_date.get(day)
            if ranges is None:
                ranges = DEFAULT_BUSY if day.weekday() < 5 else []
            for start, end in ranges:
                lo = datetime.combine(day, start, tzinfo=zone).astimezone(UTC)
                hi = datetime.combine(day, end, tzinfo=zone).astimezone(UTC)
                if lo < window.end and hi > window.start:
                    busy.append(Interval(lo, hi))
            day += timedelta(days=1)

        async with self._sessionmaker() as session:
            payloads = (await session.execute(select(MockCalendarEvent.payload))).scalars().all()
        for payload in payloads:
            lo = datetime.fromisoformat(payload["start"])
            hi = datetime.fromisoformat(payload["end"])
            if lo < window.end and hi > window.start:
                busy.append(Interval(lo, hi))
        busy.sort(key=lambda i: (i.start, i.end))
        return FreeBusy(busy=busy, retrieved_at=datetime.now(UTC))

    async def create_event(self, req: EventRequest, event_id: str) -> ProviderResult:
        mode = self.faults.calendar_create
        if mode == "timeout_before_accept":
            raise ProviderTimeout("event create timed out before the provider accepted it")
        if mode == "fail":
            return ProviderResult(status="failed_definite", error="mock calendar failure")
        async with self._sessionmaker() as session:
            await session.execute(
                insert(MockCalendarEvent)
                .values(event_id=event_id, payload=req.model_dump(mode="json"))
                .on_conflict_do_nothing(index_elements=["event_id"])
            )
            await session.commit()
        if mode == "timeout_after_accept":
            raise ProviderTimeout("event create timed out after the provider accepted it")
        return ProviderResult(status="confirmed", provider_ref=event_id)

    async def get_event(self, event_id: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            row = await session.get(MockCalendarEvent, event_id)
            return dict(row.payload) if row else None
