"""`draft_reply` and `draft_meeting` nodes: turn model output into an action payload.

Node: each receives the state and returns the partial update
`{"draft": payload, "draft_kind": ..., "draft_errors": ...}`. The model only supplies the
creative fields (`ReplyDraft` / `MeetingDraft`); code derives everything that must be exact:
thread id, `In-Reply-To`, `References` and subject for replies; the calendar, and UTC instants
(localized from the model's wall-clock times with `ZoneInfo`) for events. The payload is checked
next by `validate_proposal`; nothing here is sent or written anywhere.

Runtime context: the LLM client and settings come from `runtime.context` (`GraphDeps`).

If the model output cannot become a payload (no inbound message to answer, a local time that
does not exist because of a DST jump), the node returns `draft=None` plus `draft_errors`, and
validation reports those errors so the planner gets one chance to fix them.
"""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from langgraph.runtime import Runtime

from app.graphs.deps import GraphDeps
from app.graphs.nodes.reads import scheduling_preferences
from app.graphs.nodes.shared import (
    availability_findings,
    is_from_owner,
    llm_failure_update,
    owner_identities,
    preferences_context,
    thread_of,
    thread_participants,
    user_content,
)
from app.graphs.schemas import MeetingDraft, ReplyDraft
from app.graphs.state import EmailState
from app.llm.client import BudgetExceeded, LLMCall, LLMMalformedOutput, LLMRefusal, T
from app.mail.addresses import normalize_address
from app.mail.headers import reply_subject
from app.prompts import load_prompt


class _NonexistentLocalTime(ValueError):
    pass


def _choices(state: EmailState, identities: list[str]) -> list[tuple[str, str]]:
    """Sections telling the model exactly which recipients and times it may use."""
    participants = thread_participants(thread_of(state), identities)
    slot_lines: list[str] = []
    for finding in availability_findings(state.get("read_findings") or []):
        slot_lines.append(f"{finding['key']} (time zone {finding['time_zone']}):")
        slot_lines += [
            f"- {s['local']}  [UTC {s['start']} to {s['end']}]" for s in finding["candidate_slots"]
        ] or ["- no free slot in this range"]
    tz = scheduling_preferences(state)["time_zone"]
    return [
        (
            "Allowed recipients (thread participants other than the owner; use only these)",
            "\n".join(f"- {p}" for p in participants),
        ),
        (
            "Candidate slots (offer or book only these times; cite the availability key)",
            "\n".join(slot_lines),
        ),
        ("Owner time zone", str(tz)),
    ]


async def _draft(
    state: EmailState,
    runtime: Runtime[GraphDeps],
    purpose: Literal["draft_reply", "draft_meeting"],
    output_model: type[T],
) -> T | dict[str, Any]:
    """Call the model; return its output, or the error update for a handled LLM failure."""
    deps = runtime.context
    call = LLMCall(
        purpose=purpose,
        system=load_prompt(purpose).system,
        cached_context=preferences_context(state),
        user_content=user_content(state, _choices(state, owner_identities(deps.settings))),
        output_model=output_model,
        workflow_id=UUID(state["workflow_id"]),
        script_key=state["provider_message_id"],
    )
    try:
        return (await deps.llm.structured(call)).output
    except (LLMRefusal, LLMMalformedOutput, BudgetExceeded) as exc:
        return llm_failure_update(exc)


def _normalized(addresses: list[str]) -> list[str]:
    """Normalize what parses; keep the rest verbatim so validation can name it."""
    out: list[str] = []
    for raw in addresses:
        try:
            out.append(normalize_address(raw))
        except ValueError:
            out.append(raw)
    return out


def _localize(local: datetime, zone: ZoneInfo) -> datetime:
    """Attach `zone` to a wall-clock time; reject times skipped by a DST jump."""
    aware = local.replace(tzinfo=zone)
    if aware.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != local:
        raise _NonexistentLocalTime(local.isoformat())
    return aware


def _failed(kind: str, *errors: str) -> dict[str, Any]:
    return {"draft": None, "draft_kind": kind, "draft_errors": list(errors)}


async def draft_reply(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Draft a reply; code adds the threading headers and subject."""
    result = await _draft(state, runtime, "draft_reply", ReplyDraft)
    if isinstance(result, dict):
        return result
    thread = thread_of(state)
    identities = owner_identities(runtime.context.settings)
    inbound = [m for m in thread.messages if not is_from_owner(m, identities)]
    if not inbound:
        return _failed("send_reply", "no_inbound_message")
    latest = inbound[-1]
    if not latest.rfc_message_id:
        return _failed("send_reply", "missing_rfc_message_id")
    references = list(dict.fromkeys(m.rfc_message_id for m in thread.messages if m.rfc_message_id))
    payload = {
        "provider_thread_id": thread.provider_thread_id,
        "in_reply_to": latest.rfc_message_id,
        "references": references,
        "to": _normalized(result.to),
        "cc": _normalized(result.cc),
        "subject": reply_subject(latest.subject),
        "body_text": result.body_text,
        "evidence_refs": list(result.evidence_refs),
    }
    return {"draft": payload, "draft_kind": "send_reply", "draft_errors": None}


async def draft_meeting(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Draft a calendar event; code converts the local times to UTC instants."""
    result = await _draft(state, runtime, "draft_meeting", MeetingDraft)
    if isinstance(result, dict):
        return result
    zone = ZoneInfo(result.time_zone)  # MeetingDraft already checked the zone exists
    try:
        start = _localize(result.start_local, zone)
        end = _localize(result.end_local, zone)
    except _NonexistentLocalTime:
        return _failed("create_event", "invalid_local_time")
    payload = {
        "calendar_id": "primary",
        "title": result.title,
        "start": start.astimezone(UTC).isoformat(),
        "end": end.astimezone(UTC).isoformat(),
        "time_zone": result.time_zone,
        "attendees": _normalized(result.attendees),
        "location": result.location,
        "send_updates": result.send_updates,
        "evidence_refs": list(result.evidence_refs),
    }
    return {"draft": payload, "draft_kind": "create_event", "draft_errors": None}
