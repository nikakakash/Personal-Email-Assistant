"""MockCalendarProvider: deterministic busy pattern, stored events and faults."""

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.providers.interfaces import EventRequest, ProviderTimeout
from app.providers.mock_calendar import MockCalendarProvider
from app.providers.mock_mail import MockFaults
from app.scheduling.slots import Interval

BERLIN = ZoneInfo("Europe/Berlin")


def _local(d: int, h: int, m: int = 0) -> datetime:
    return datetime(2026, 11, d, h, m, tzinfo=BERLIN).astimezone(UTC)


def _event(start: datetime, end: datetime) -> EventRequest:
    return EventRequest(
        title="Call",
        start=start,
        end=end,
        time_zone="Europe/Berlin",
        attendees=["alice@example.com"],
        location=None,
        send_updates="none",
    )


async def test_freebusy_default_pattern(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    provider = MockCalendarProvider(sessionmaker, {}, "Europe/Berlin", MockFaults())
    # Thu 2026-11-19 through Mon 2026-11-23 (weekend in between).
    window = Interval(_local(19, 0), _local(23, 23))
    fb = await provider.get_freebusy(window, "Europe/Berlin")
    expected = [
        Interval(_local(19, 12), _local(19, 13)),
        Interval(_local(19, 15), _local(19, 16)),
        Interval(_local(20, 12), _local(20, 13)),
        Interval(_local(20, 15), _local(20, 16)),
        Interval(_local(23, 12), _local(23, 13)),
        Interval(_local(23, 15), _local(23, 16)),
    ]
    assert fb.busy == expected
    assert fb.retrieved_at.tzinfo is not None


async def test_freebusy_busy_times_do_not_depend_on_the_requested_zone(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """Like a real calendar: busy blocks live in the calendar's own zone; `time_zone` only
    names the caller's display zone and must not move them (a meeting in another zone
    would otherwise collide with phantom busy time at preflight)."""
    provider = MockCalendarProvider(sessionmaker, {}, "Europe/Berlin", MockFaults())
    window = Interval(_local(19, 0), _local(19, 23))
    berlin = await provider.get_freebusy(window, "Europe/Berlin")
    kolkata = await provider.get_freebusy(window, "Asia/Kolkata")
    assert (
        kolkata.busy
        == berlin.busy
        == [
            Interval(_local(19, 12), _local(19, 13)),
            Interval(_local(19, 15), _local(19, 16)),
        ]
    )


async def test_freebusy_busy_by_date_overrides_default_for_that_date(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    provider = MockCalendarProvider(
        sessionmaker, {date(2026, 11, 19): [(time(9), time(10))]}, "Europe/Berlin", MockFaults()
    )
    fb = await provider.get_freebusy(Interval(_local(19, 0), _local(20, 0)), "Europe/Berlin")
    assert fb.busy == [Interval(_local(19, 9), _local(19, 10))]


async def test_created_event_shows_in_freebusy_and_get_event(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    provider = MockCalendarProvider(sessionmaker, {}, "Europe/Berlin", MockFaults())
    res = await provider.create_event(_event(_local(20, 10), _local(20, 10, 30)), "ev-1")
    assert res.status == "confirmed" and res.provider_ref == "ev-1"
    fb = await provider.get_freebusy(Interval(_local(20, 0), _local(21, 0)), "Europe/Berlin")
    assert Interval(_local(20, 10), _local(20, 10, 30)) in fb.busy
    stored = await provider.get_event("ev-1")
    assert stored is not None and stored["title"] == "Call"
    assert await provider.get_event("missing") is None


async def test_create_event_is_idempotent_per_event_id(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    provider = MockCalendarProvider(sessionmaker, {}, "Europe/Berlin", MockFaults())
    req = _event(_local(20, 10), _local(20, 11))
    await provider.create_event(req, "ev-1")
    again = await provider.create_event(req, "ev-1")
    assert again.status == "confirmed"


async def test_calendar_faults(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    faults = MockFaults(calendar_create="fail")
    provider = MockCalendarProvider(sessionmaker, {}, "Europe/Berlin", faults)
    req = _event(_local(20, 10), _local(20, 11))
    assert (await provider.create_event(req, "ev-1")).status == "failed_definite"
    assert await provider.get_event("ev-1") is None

    faults.calendar_create = "timeout_before_accept"
    with pytest.raises(ProviderTimeout):
        await provider.create_event(req, "ev-1")
    assert await provider.get_event("ev-1") is None

    faults.calendar_create = "timeout_after_accept"
    with pytest.raises(ProviderTimeout):
        await provider.create_event(req, "ev-1")
    assert await provider.get_event("ev-1") is not None
