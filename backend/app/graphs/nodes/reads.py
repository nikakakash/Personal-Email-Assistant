"""`read_tools` node: runs the read-only tool the planner selected (`read_thread`, `get_freebusy`).

Node: it receives the state and returns a partial update. Results are appended to
`read_findings`, whose `merge_by_key` reducer (see `state.py`) merges by `"key"`, so re-running
the same read replaces its finding instead of duplicating it.

Runtime context: providers and the clock come from `runtime.context` (`GraphDeps`), never from
state, so nothing live is checkpointed.

Every provider call is bounded by `TOOL_TIMEOUT_SECONDS`. A timeout or provider failure becomes a
`tool_error` finding rather than an exception: the planner sees it and can retry or ask the
owner, and never has to invent availability.
"""

import asyncio
import logging
from collections.abc import Awaitable
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.graphs.deps import GraphDeps
from app.graphs.schemas import FreeBusyArgs
from app.graphs.state import EmailState
from app.providers.interfaces import ProviderTimeout
from app.scheduling.slots import Interval, SchedulingPrefs, find_slots

TOOL_TIMEOUT_SECONDS: float = 60
THREAD_MESSAGE_LIMIT = 20

# Used until onboarding stores the owner's real scheduling preferences.
DEFAULT_SCHEDULING: dict[str, Any] = {
    "time_zone": "UTC",
    "work_start": "09:00",
    "work_end": "17:00",
    "work_days": [0, 1, 2, 3, 4],
    "buffer_minutes": 10,
    "default_duration_minutes": 30,
}

logger = logging.getLogger(__name__)

_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class _ToolFailed(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def scheduling_preferences(state: EmailState) -> dict[str, Any]:
    """The owner's scheduling preferences, with defaults for anything unset."""
    return {**DEFAULT_SCHEDULING, **((state.get("preferences") or {}).get("scheduling") or {})}


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def render_local(slot: Interval, zone: ZoneInfo, zone_name: str) -> str:
    """E.g. ``Thu 19 Nov 2026 15:00–15:30 Europe/Paris (UTC+01:00)`` (locale independent)."""
    start, end = slot.start.astimezone(zone), slot.end.astimezone(zone)
    offset = start.strftime("%z")
    return (
        f"{_DAYS[start.weekday()]} {start.day} {_MONTHS[start.month - 1]} {start.year} "
        f"{start:%H:%M}–{end:%H:%M} {zone_name} (UTC{offset[:3]}:{offset[3:]})"
    )


async def _bounded[R](call: Awaitable[R]) -> R:
    try:
        return await asyncio.wait_for(call, TOOL_TIMEOUT_SECONDS)
    except (TimeoutError, ProviderTimeout) as exc:
        raise _ToolFailed("timeout") from exc
    except KeyError as exc:
        raise _ToolFailed("not_found") from exc
    except Exception as exc:  # any provider failure is reported, never raised into the graph
        raise _ToolFailed("provider_error") from exc


async def _read_thread(state: EmailState, deps: GraphDeps) -> dict[str, Any]:
    snapshot = await _bounded(
        deps.mail.get_thread(state["provider_thread_id"], limit=THREAD_MESSAGE_LIMIT)
    )
    return {
        "thread": snapshot.model_dump(mode="json"),
        "source_version": snapshot.version,
        "read_findings": [
            {
                "key": f"thread:{snapshot.version}",
                "kind": "thread",
                "message_count": len(snapshot.messages),
            }
        ],
    }


async def _get_freebusy(state: EmailState, deps: GraphDeps) -> dict[str, Any]:
    try:
        args = FreeBusyArgs.model_validate(state.get("pending_freebusy"))
    except ValidationError as exc:
        raise _ToolFailed("invalid_args") from exc
    prefs = scheduling_preferences(state)
    try:
        tz = str(prefs["time_zone"])
        zone = ZoneInfo(tz)
        slot_prefs = SchedulingPrefs(
            time_zone=tz,
            work_start=time.fromisoformat(prefs["work_start"]),
            work_end=time.fromisoformat(prefs["work_end"]),
            work_days=frozenset(int(d) for d in prefs["work_days"]),
            buffer=timedelta(minutes=int(prefs["buffer_minutes"])),
        )
        duration = timedelta(minutes=int(prefs["default_duration_minutes"]))
        if duration <= timedelta(0):
            raise ValueError("default_duration_minutes must be positive")
    except (KeyError, TypeError, ValueError) as exc:  # ZoneInfoNotFoundError is a KeyError
        raise _ToolFailed("invalid_preferences") from exc

    # Whole owner-local days: start_date 00:00 up to (end_date + 1) 00:00.
    window = Interval(
        datetime.combine(args.start_date, time(0), tzinfo=zone).astimezone(UTC),
        datetime.combine(args.end_date + timedelta(days=1), time(0), tzinfo=zone).astimezone(UTC),
    )
    freebusy = await _bounded(deps.calendar.get_freebusy(window, tz))
    slots = find_slots(freebusy.busy, window, duration, slot_prefs, not_before=deps.now())
    finding = {
        "key": f"availability:{args.start_date.isoformat()}:{args.end_date.isoformat()}",
        "kind": "availability",
        "busy": [{"start": iso_utc(b.start), "end": iso_utc(b.end)} for b in freebusy.busy],
        "candidate_slots": [
            {"start": iso_utc(s.start), "end": iso_utc(s.end), "local": render_local(s, zone, tz)}
            for s in slots
        ],
        "time_zone": tz,
        "retrieved_at": freebusy.retrieved_at.isoformat(),
    }
    return {"read_findings": [finding], "pending_freebusy": None}


async def read_tools(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Run the read tool named by `next_action`, then clear `next_action` (it is consumed)."""
    tool = state.get("next_action")
    if tool == "read_thread":
        run = _read_thread
    elif tool == "get_freebusy":
        run = _get_freebusy
    else:
        raise ValueError(f"read_tools cannot run {tool!r}; route_after_plan should prevent this")
    try:
        update = await run(state, runtime.context)
    except _ToolFailed as failure:
        # The traceback (with the provider's exception as its cause) stays in server logs;
        # the finding, which the model reads, carries only the fixed code.
        logger.exception("read tool %s failed: %s", tool, failure.code)
        update = {
            "read_findings": [
                {
                    "key": f"error:{tool}:{state.get('step_count', 0)}",
                    "kind": "tool_error",
                    "tool": tool,
                    "error": failure.code,
                }
            ]
        }
    return {**update, "next_action": None}
