"""Deterministic proposal validation: `validate_payload`, the `validate_proposal` node and the
`route_after_validate` router.

`validate_payload` is ordinary code with no model in the loop (spec §10: facts such as allowed
recipients, durations and required fields are checked by code). It returns error codes; an empty
list means valid. It is shared by two callers:
- `validate_proposal` (this module), for model-authored drafts (`author="model"`);
- the edit-and-approve path (Task 9), for owner-edited payloads (`author="owner"`). The owner may
  add recipients, pick another time and drop evidence, so those three checks are skipped; every
  safety check (field shape, address parsing, header injection, body size, time zone,
  end after start, duration bounds) still applies.

Node: `validate_proposal(state, runtime)` returns a partial update with `validation` and, on
failure, an owner-invisible `validation:<n>` finding (merged into `read_findings` by its reducer)
so the planner can see what went wrong. Runtime context supplies only the owner's identities.

Conditional edge: `route_after_validate(state)` is a pure function of state, as every router must
be, so the same state always takes the same path. It sends a valid draft to
`persist_proposal`, the first invalid one back to `plan_next_step`, and a second one to `error`.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from langgraph.runtime import Runtime

from app.graphs.deps import GraphDeps
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
        if allowed is not None and addr not in allowed:
            errors.append(f"{not_allowed_code}:{_show(addr)}")
    return errors


def _known_evidence(
    thread: ThreadSnapshot, findings: Sequence[dict[str, Any]], owner_inputs: Sequence[Any]
) -> set[str]:
    known = {f"message:{m.provider_message_id}" for m in thread.messages}
    known.add(f"thread:{thread.provider_thread_id}")
    known |= {f"owner_input:{n}" for n in range(len(owner_inputs))}
    for finding in availability_findings(findings):
        key = str(finding["key"])
        known |= {key, f"availability:{key}"}
    return known


def _evidence_errors(
    refs: list[str],
    thread: ThreadSnapshot,
    findings: Sequence[dict[str, Any]],
    owner_inputs: Sequence[Any],
) -> list[str]:
    if not refs:
        return ["missing_evidence"]
    known = _known_evidence(thread, findings, owner_inputs)
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


def _in_candidate_slot(start: datetime, findings: Sequence[dict[str, Any]]) -> bool:
    for finding in availability_findings(findings):
        for slot in finding.get("candidate_slots", []):
            if datetime.fromisoformat(slot["start"]) <= start < datetime.fromisoformat(slot["end"]):
                return True
    return False


def _event_errors(
    p: dict[str, Any],
    findings: Sequence[dict[str, Any]],
    allowed: set[str] | None,
    check_slot: bool,
) -> list[str]:
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
    if check_slot and start is not None and not _in_candidate_slot(start, findings):
        errors.append("time_not_from_availability")

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
) -> list[str]:
    """Return error codes for `payload` (empty means valid). See the module docstring.

    `owner_inputs` is only needed to resolve `owner_input:<n>` evidence in model drafts.
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
        errors = _event_errors(payload, findings, allowed, check_slot=by_model)
    if by_model:
        errors += _evidence_errors(payload["evidence_refs"], thread, findings, owner_inputs)
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
