"""Deterministic proposal validation: `validate_payload`, the `validate_proposal` node and the
`route_after_validate` router.

`validate_payload` is ordinary code with no model in the loop (spec §10: facts such as allowed
recipients, durations and required fields are checked by code). It returns error codes; an empty
list means valid. It is shared by two callers:
- `validate_proposal` (this module), for model-authored drafts (`author="model"`);
- the edit-and-approve path (Task 9), for owner-edited payloads (`author="owner"`). The owner may
  add recipients, pick another time and drop evidence, so those checks are skipped; every
  safety check (field shape, canonical address parsing, header injection, body size, time
  zone, end after start, duration bounds) still applies.

For a model-drafted meeting the time must come from an availability read: `start` equals a
candidate slot's start exactly, `[start - buffer, end + buffer)` misses that finding's busy
intervals, and start and end lie inside working hours on a working day (owner's zone).

Node: `validate_proposal(state, runtime)` returns a partial update with `validation` and, on
failure, an owner-invisible `validation:<n>` finding (merged into `read_findings` by its reducer)
so the planner can see what went wrong. Runtime context supplies only the owner's identities.

Conditional edge: `route_after_validate(state)` is a pure function of state, as every router must
be, so the same state always takes the same path. It sends a valid draft to
`persist_proposal`, the first invalid one back to `plan_next_step`, and a second one to `error`.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, time, timedelta
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from langgraph.runtime import Runtime

from app.graphs.deps import GraphDeps
from app.graphs.nodes.reads import DEFAULT_SCHEDULING, scheduling_preferences
from app.graphs.nodes.shared import availability_findings, owner_identities, thread_participants
from app.graphs.state import EmailState
from app.mail.addresses import normalize_address
from app.mail.content import MAX_BODY_CHARS
from app.mail.headers import HeaderInjectionError, assert_safe_header
from app.providers.interfaces import ThreadSnapshot

Kind = Literal["send_reply", "create_event"]
Author = Literal["model", "owner"]

MIN_EVENT_DURATION = timedelta(minutes=5)
MAX_EVENT_DURATION = timedelta(hours=8)
# Invalid drafts loop back to planning this many times before the workflow errors.
MAX_VALIDATION_RETRIES = 1

_STR, _OPT_STR, _STR_LIST = "str", "str|None", "list[str]"
_FIELDS: dict[str, dict[str, str]] = {
    "send_reply": {
        "provider_thread_id": _STR,
        "in_reply_to": _STR,
        "references": _STR_LIST,
        "to": _STR_LIST,
        "cc": _STR_LIST,
        "subject": _STR,
        "body_text": _STR,
        "evidence_refs": _STR_LIST,
    },
    "create_event": {
        "calendar_id": _STR,
        "title": _STR,
        "start": _STR,
        "end": _STR,
        "time_zone": _STR,
        "attendees": _STR_LIST,
        "location": _OPT_STR,
        "send_updates": _STR,
        "evidence_refs": _STR_LIST,
    },
}


def _show(value: str) -> str:
    """Render untrusted text inside an error code: escaped, single line, bounded."""
    return value.encode("unicode_escape").decode("ascii")[:120]


def _type_ok(value: Any, expected: str) -> bool:
    if expected == _STR:
        return isinstance(value, str)
    if expected == _OPT_STR:
        return value is None or isinstance(value, str)
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _shape_errors(kind: Kind, payload: dict[str, Any]) -> list[str]:
    spec = _FIELDS[kind]
    errors = [f"unexpected_field:{_show(k)}" for k in payload if k not in spec]
    for name, expected in spec.items():
        if name not in payload:
            errors.append(f"missing_field:{name}")
        elif not _type_ok(payload[name], expected):
            errors.append(f"invalid_field:{name}")
    return errors


def _header_ok(value: str) -> bool:
    try:
        assert_safe_header(value)
    except HeaderInjectionError:
        return False
    return True


def _address_errors(
    field: str,
    addresses: list[str],
    allowed: set[str] | None,
    not_allowed_code: str,
) -> list[str]:
    """Check one address field. `allowed=None` means any parseable address is fine."""
    errors: list[str] = []
    if not all(_header_ok(a) for a in addresses):
        errors.append(f"unsafe_header:{field}")
    for raw in addresses:
        if not _header_ok(raw):
            continue
        try:
            addr = normalize_address(raw)
        except ValueError:
            errors.append(f"invalid_address:{_show(raw)}")
            continue
        if raw != addr:
            # Only bare lowercase addresses are sent; a display name or odd casing could hide
            # a different recipient from the owner reviewing the proposal.
            errors.append(f"non_canonical_address:{_show(raw)}")
            continue
        if allowed is not None and addr not in allowed:
            errors.append(f"{not_allowed_code}:{_show(addr)}")
    return errors


def _known_evidence(
    thread: ThreadSnapshot,
    findings: Sequence[dict[str, Any]],
    owner_inputs: Sequence[Any],
    history_refs: Iterable[str] = (),
) -> set[str]:
    known = {f"message:{m.provider_message_id}" for m in thread.messages}
    # Only past emails actually retrieved for this run may be cited (no invented history).
    known |= {ref for ref in history_refs if ref.startswith("history:")}
    known.add(f"thread:{thread.provider_thread_id}")
    known |= {f"owner_input:{n}" for n in range(len(owner_inputs))}
    # Finding keys the model sees verbatim: `availability:<start>:<end>` and `thread:<version>`.
    known |= {str(f["key"]) for f in findings if f.get("kind") in ("availability", "thread")}
    return known


def _evidence_errors(
    refs: list[str],
    thread: ThreadSnapshot,
    findings: Sequence[dict[str, Any]],
    owner_inputs: Sequence[Any],
    history_refs: Iterable[str] = (),
) -> list[str]:
    if not refs:
        return ["missing_evidence"]
    known = _known_evidence(thread, findings, owner_inputs, history_refs)
    return [f"unknown_evidence:{_show(ref)}" for ref in refs if ref not in known]


def _reply_errors(p: dict[str, Any], thread: ThreadSnapshot, allowed: set[str] | None) -> list[str]:
    errors: list[str] = []
    rfc_ids = {m.rfc_message_id for m in thread.messages if m.rfc_message_id}
    if p["provider_thread_id"] != thread.provider_thread_id:
        errors.append("thread_mismatch")
    if p["in_reply_to"] not in rfc_ids:
        errors.append("invalid_in_reply_to")
    if not set(p["references"]) <= rfc_ids:
        errors.append("invalid_references")
    if not p["to"] and not p["cc"]:
        errors.append("no_recipients")
    errors += _address_errors("to", p["to"], allowed, "recipient_not_in_thread")
    errors += _address_errors("cc", p["cc"], allowed, "recipient_not_in_thread")
    if not _header_ok(p["subject"]):
        errors.append("unsafe_header:subject")
    elif not p["subject"].strip():
        errors.append("empty_subject")
    if not p["body_text"].strip():
        errors.append("empty_body")
    elif len(p["body_text"]) > MAX_BODY_CHARS:
        errors.append("body_too_long")
    return errors


def _parse_aware(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _slot_finding(start: datetime, findings: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The availability finding that offered a candidate slot starting exactly at `start`."""
    for finding in availability_findings(findings):
        for slot in finding.get("candidate_slots", []):
            if datetime.fromisoformat(slot["start"]) == start:
                return finding
    return None


def _overlaps_busy(
    start: datetime, end: datetime, finding: dict[str, Any], buffer: timedelta
) -> bool:
    lo, hi = start - buffer, end + buffer
    return any(
        lo < datetime.fromisoformat(b["end"]) and hi > datetime.fromisoformat(b["start"])
        for b in finding.get("busy", [])
    )


def _within_working_hours(start: datetime, end: datetime, prefs: Mapping[str, Any]) -> bool:
    """Start and end on the same working day, inside [work_start, work_end], owner-local."""
    zone = ZoneInfo(str(prefs["time_zone"]))
    lo, hi = start.astimezone(zone), end.astimezone(zone)
    work_days = {int(d) for d in prefs["work_days"]}
    return (
        lo.date() == hi.date()
        and lo.weekday() in work_days
        and time.fromisoformat(prefs["work_start"]) <= lo.time()
        and hi.time() <= time.fromisoformat(prefs["work_end"])
    )


def _availability_errors(
    start: datetime | None,
    end: datetime | None,
    findings: Sequence[dict[str, Any]],
    scheduling: Mapping[str, Any],
) -> list[str]:
    """Model drafts: the time must be one the availability read actually offered."""
    if start is None:
        return []
    finding = _slot_finding(start, findings)
    errors = [] if finding else ["time_not_from_availability"]
    if end is None or end <= start:
        return errors
    prefs = {**DEFAULT_SCHEDULING, **scheduling}
    try:
        buffer = timedelta(minutes=int(prefs["buffer_minutes"]))
        in_hours = _within_working_hours(start, end, prefs)
    except (KeyError, TypeError, ValueError):  # ZoneInfoNotFoundError is a KeyError
        return [*errors, "invalid_preferences"]
    if finding and _overlaps_busy(start, end, finding, buffer):
        errors.append("time_overlaps_busy")
    if not in_hours:
        errors.append("outside_working_hours")
    return errors


def _event_errors(
    p: dict[str, Any],
    findings: Sequence[dict[str, Any]],
    allowed: set[str] | None,
    scheduling: Mapping[str, Any] | None,
) -> list[str]:
    """`scheduling=None` skips the availability checks (owner-authored payloads)."""
    errors: list[str] = []
    if p["calendar_id"] != "primary":
        errors.append("invalid_calendar")
    if not _header_ok(p["title"]):
        errors.append("unsafe_header:title")
    elif not p["title"].strip():
        errors.append("empty_title")
    if p["location"] is not None and not _header_ok(p["location"]):
        errors.append("unsafe_header:location")
    try:
        ZoneInfo(p["time_zone"])
    except (KeyError, ValueError, OSError):  # ZoneInfoNotFoundError is a KeyError
        errors.append("invalid_time_zone")

    start, end = _parse_aware(p["start"]), _parse_aware(p["end"])
    if start is None:
        errors.append("invalid_time:start")
    if end is None:
        errors.append("invalid_time:end")
    if start is not None and end is not None:
        if end <= start:
            errors.append("end_not_after_start")
        elif not MIN_EVENT_DURATION <= end - start <= MAX_EVENT_DURATION:
            errors.append("duration_out_of_bounds")
    if scheduling is not None:
        errors += _availability_errors(start, end, findings, scheduling)

    if p["send_updates"] not in ("all", "none"):
        errors.append("invalid_field:send_updates")
    errors += _address_errors("attendees", p["attendees"], allowed, "attendee_not_in_thread")
    return errors


def validate_payload(
    kind: Kind,
    payload: dict[str, Any],
    *,
    thread: ThreadSnapshot,
    findings: list[dict[str, Any]],
    owner_identities: Sequence[str],
    author: Author,
    owner_inputs: Sequence[Any] = (),
    scheduling: Mapping[str, Any] | None = None,
    history_refs: Iterable[str] = (),
) -> list[str]:
    """Return error codes for `payload` (empty means valid). See the module docstring.

    `owner_inputs` resolves `owner_input:<n>` evidence in model drafts. `scheduling` is the
    owner's scheduling preferences (missing keys fall back to `DEFAULT_SCHEDULING`);
    model-drafted meetings must fit its working hours and buffers.
    """
    if kind not in _FIELDS:
        raise ValueError(f"unknown proposal kind {kind!r}")
    errors = _shape_errors(kind, payload)
    if errors:
        return errors  # the rest assumes every field is present with the right type

    by_model = author == "model"
    # Model drafts may only address people already on the thread, never the owner.
    allowed = set(thread_participants(thread, owner_identities)) if by_model else None
    if kind == "send_reply":
        errors = _reply_errors(payload, thread, allowed)
    else:
        errors = _event_errors(
            payload, findings, allowed, scheduling=(scheduling or {}) if by_model else None
        )
    if by_model:
        errors += _evidence_errors(
            payload["evidence_refs"], thread, findings, owner_inputs, history_refs
        )
    return errors


async def validate_proposal(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Validate the current draft (pure code; no model, no writes)."""
    if state.get("next_action") in ("error", "budget_stop"):
        return {}  # drafting already failed; route_after_validate forwards that outcome

    draft, kind = state.get("draft"), state.get("draft_kind")
    if state.get("draft_errors"):
        errors = list(state.get("draft_errors") or [])
    elif draft is None or kind not in _FIELDS:
        errors = ["missing_draft"]
    else:
        errors = validate_payload(
            cast(Kind, kind),  # checked against _FIELDS above
            draft,
            thread=ThreadSnapshot.model_validate(state["thread"]),
            findings=list(state.get("read_findings") or []),
            owner_identities=owner_identities(runtime.context.settings),
            author="model",
            owner_inputs=list(state.get("owner_inputs") or []),
            scheduling=scheduling_preferences(state),
            history_refs=[str(h["ref"]) for h in state.get("retrieved_context") or []],
        )

    if not errors:
        return {"validation": {"ok": True, "errors": []}, "validation_attempts": 0}

    attempts = state.get("validation_attempts", 0) + 1
    update: dict[str, Any] = {
        "validation": {"ok": False, "errors": errors},
        "validation_attempts": attempts,
        "read_findings": [
            {"key": f"validation:{attempts}", "kind": "validation", "errors": errors}
        ],
    }
    if attempts > MAX_VALIDATION_RETRIES:
        update["error_code"] = "validation_failed"
    return update


def route_after_validate(
    state: EmailState,
) -> Literal["persist_proposal", "plan_next_step", "error", "budget_stop"]:
    """Pick the next node from the validation outcome (pure: reads state only)."""
    if state.get("next_action") == "budget_stop":
        return "budget_stop"
    if state.get("next_action") == "error":
        return "error"
    validation = state.get("validation") or {}
    if validation.get("ok"):
        return "persist_proposal"
    if state.get("validation_attempts", 0) > MAX_VALIDATION_RETRIES:
        return "error"
    return "plan_next_step"
