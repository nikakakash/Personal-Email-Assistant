"""Slot finder: proposes meeting times inside working hours, avoiding busy time.

Candidates are generated on the owner's local wall clock (so DST shifts are
respected) and returned as UTC intervals.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

MIN_SLOT_SPACING = timedelta(hours=1)


@dataclass(frozen=True)
class Interval:
    """A half-open time range; both ends must be timezone-aware and end > start."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Interval datetimes must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("Interval end must be after start")


@dataclass(frozen=True)
class SchedulingPrefs:
    """Owner scheduling preferences. ``work_days`` uses 0 for Monday."""

    time_zone: str
    work_start: time
    work_end: time
    work_days: frozenset[int]
    buffer: timedelta = timedelta(minutes=10)
    step: timedelta = timedelta(minutes=30)
    _zone: ZoneInfo = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            zone = ZoneInfo(self.time_zone)
        except Exception as exc:  # ZoneInfoNotFoundError, ValueError for bad keys
            raise ValueError(f"unknown time zone: {self.time_zone!r}") from exc
        object.__setattr__(self, "_zone", zone)
        if self.step <= timedelta(0):
            raise ValueError("step must be positive")


def _overlaps_busy(start: datetime, end: datetime, busy: Sequence[Interval]) -> bool:
    return any(start < b.end and end > b.start for b in busy)


def find_slots(
    busy: Sequence[Interval],
    window: Interval,
    duration: timedelta,
    prefs: SchedulingPrefs,
    max_slots: int = 3,
    not_before: datetime | None = None,
) -> list[Interval]:
    """Return up to ``max_slots`` UTC slots, at least an hour apart, sorted by start."""
    if duration <= timedelta(0):
        raise ValueError("duration must be positive")
    if not_before is not None and not_before.tzinfo is None:
        raise ValueError("not_before must be timezone-aware")
    zone = prefs._zone
    earliest = window.start if not_before is None else max(window.start, not_before)
    day = window.start.astimezone(zone).date()
    last_day = window.end.astimezone(zone).date()
    slots: list[Interval] = []

    while day <= last_day and len(slots) < max_slots:
        if day.weekday() in prefs.work_days:
            wall = datetime.combine(day, prefs.work_start)
            wall_limit = datetime.combine(day, prefs.work_end)
            while wall + duration <= wall_limit and len(slots) < max_slots:
                local = wall.replace(tzinfo=zone)
                start = local.astimezone(UTC)
                wall += prefs.step
                # Skip wall times that do not exist (spring-forward gap).
                if start.astimezone(zone).replace(tzinfo=None) != local.replace(tzinfo=None):
                    continue
                end = start + duration
                if start < earliest or end > window.end:
                    continue
                if slots and start < slots[-1].start + MIN_SLOT_SPACING:
                    continue
                if _overlaps_busy(start - prefs.buffer, end + prefs.buffer, busy):
                    continue
                slots.append(Interval(start, end))
        day += timedelta(days=1)
    return slots
