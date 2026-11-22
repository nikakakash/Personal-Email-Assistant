"""Tests for the working-hours / DST-aware slot finder."""

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.scheduling.slots import Interval, SchedulingPrefs, find_slots

PARIS = ZoneInfo("Europe/Paris")
PREFS = SchedulingPrefs(
    time_zone="Europe/Paris",
    work_start=time(9, 0),
    work_end=time(17, 0),
    work_days=frozenset({0, 1, 2, 3, 4}),
)
HALF_HOUR = timedelta(minutes=30)


def day_window(y: int, m: int, d: int) -> Interval:
    start = datetime(y, m, d, tzinfo=PARIS)
    return Interval(start, start + timedelta(days=1))


def test_slots_respect_working_hours() -> None:
    slots = find_slots([], day_window(2026, 11, 19), HALF_HOUR, PREFS)
    assert slots[0].start == datetime(2026, 11, 19, 8, 0, tzinfo=UTC)
    assert slots[0].end == datetime(2026, 11, 19, 8, 30, tzinfo=UTC)
    assert all(s.start.tzinfo is not None and s.start.utcoffset() == timedelta(0) for s in slots)


def test_slots_at_most_one_per_hour() -> None:
    slots = find_slots([], day_window(2026, 11, 19), HALF_HOUR, PREFS, max_slots=5)
    assert [s.start.hour for s in slots] == [8, 9, 10, 11, 12]


def test_slots_skip_busy_with_buffer() -> None:
    busy = [
        Interval(
            datetime(2026, 11, 19, 9, 0, tzinfo=PARIS),
            datetime(2026, 11, 19, 10, 0, tzinfo=PARIS),
        )
    ]
    slots = find_slots(busy, day_window(2026, 11, 19), HALF_HOUR, PREFS)
    assert slots[0].start == datetime(2026, 11, 19, 10, 30, tzinfo=PARIS)


def test_slots_never_overlap_work_end() -> None:
    slots = find_slots([], day_window(2026, 11, 19), timedelta(minutes=45), PREFS, max_slots=20)
    assert all(s.end.astimezone(PARIS).time() <= time(17, 0) for s in slots)


def test_slots_across_dst_change() -> None:
    slots = find_slots([], day_window(2026, 3, 30), HALF_HOUR, PREFS)
    assert slots[0].start == datetime(2026, 3, 30, 7, 0, tzinfo=UTC)


def test_weekend_excluded() -> None:
    assert find_slots([], day_window(2026, 11, 21), HALF_HOUR, PREFS) == []


def test_not_before_excludes_earlier_candidates() -> None:
    not_before = datetime(2026, 11, 19, 11, 15, tzinfo=PARIS)
    slots = find_slots([], day_window(2026, 11, 19), HALF_HOUR, PREFS, not_before=not_before)
    assert slots[0].start == datetime(2026, 11, 19, 11, 30, tzinfo=PARIS)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError):
        Interval(datetime(2026, 1, 1), datetime(2026, 1, 2, tzinfo=UTC))


def test_end_must_follow_start() -> None:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        Interval(t, t)


def test_invalid_time_zone_rejected() -> None:
    with pytest.raises(ValueError):
        SchedulingPrefs("Nowhere/Land", time(9), time(17), frozenset({0}))
