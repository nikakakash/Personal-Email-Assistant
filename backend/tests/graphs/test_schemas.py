"""Tests for the graph output schemas and the state reducer."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.graphs.schemas import FreeBusyArgs, MeetingDraft, NextStep, ReplyDraft, TriageDecision
from app.graphs.state import merge_by_key


def test_nextstep_freebusy_requires_args() -> None:
    with pytest.raises(ValidationError):
        NextStep(action="get_freebusy", reason="need times")
    ok = NextStep(
        action="get_freebusy",
        reason="need times",
        freebusy=FreeBusyArgs(start_date=date(2026, 11, 23), end_date=date(2026, 11, 27)),
    )
    assert ok.freebusy is not None


def test_nextstep_ask_owner_requires_question() -> None:
    with pytest.raises(ValidationError):
        NextStep(action="ask_owner", reason="unknown zone")


def test_freebusy_range_max_14_days() -> None:
    FreeBusyArgs(start_date=date(2026, 11, 23), end_date=date(2026, 12, 7))
    with pytest.raises(ValidationError):
        FreeBusyArgs(start_date=date(2026, 11, 23), end_date=date(2026, 12, 8))
    with pytest.raises(ValidationError):
        FreeBusyArgs(start_date=date(2026, 11, 23), end_date=date(2026, 11, 22))


def test_triage_reason_length_limit() -> None:
    base = {
        "classification": "respond",
        "urgency": "low",
        "evidence_refs": ["message:m1"],
        "needs_clarification": False,
    }
    TriageDecision(reason="x" * 300, **base)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TriageDecision(reason="x" * 301, **base)  # type: ignore[arg-type]


def test_evidence_refs_format_enforced() -> None:
    with pytest.raises(ValidationError):
        ReplyDraft(to=["a@b.c"], cc=[], body_text="hi", evidence_refs=["bogus:1"])
    with pytest.raises(ValidationError):
        ReplyDraft(to=["a@b.c"], cc=[], body_text="hi", evidence_refs=["message:has space"])
    ReplyDraft(to=["a@b.c"], cc=[], body_text="hi", evidence_refs=["message:m-1", "thread:t.2"])


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ReplyDraft(to=[], cc=[], body_text="x", evidence_refs=[], subject="no")  # type: ignore[call-arg]


def _meeting(**over: object) -> MeetingDraft:
    data: dict[str, object] = {
        "title": "Sync",
        "start_local": "2026-11-24T10:00:00",
        "end_local": "2026-11-24T10:30:00",
        "time_zone": "Europe/Berlin",
        "attendees": ["a@b.c"],
        "location": None,
        "send_updates": "none",
        "evidence_refs": ["availability:2026-11-24"],
    }
    data.update(over)
    return MeetingDraft.model_validate(data)


def test_meeting_rejects_aware_datetimes_and_bad_zone() -> None:
    assert _meeting().start_local.tzinfo is None
    with pytest.raises(ValidationError):
        _meeting(start_local="2026-11-24T10:00:00+01:00")
    with pytest.raises(ValidationError):
        _meeting(end_local="2026-11-24T10:30:00Z")
    with pytest.raises(ValidationError):
        _meeting(time_zone="Mars/Olympus")


def test_merge_by_key_dedupes() -> None:
    left = [{"key": "a", "v": 1}, {"key": "b", "v": 1}]
    right = [{"key": "a", "v": 2}, {"key": "c", "v": 1}]
    assert merge_by_key(left, right) == [
        {"key": "a", "v": 2},
        {"key": "b", "v": 1},
        {"key": "c", "v": 1},
    ]


def test_merge_by_key_none_and_missing_key() -> None:
    assert merge_by_key(None, [{"key": "a"}]) == [{"key": "a"}]
    assert merge_by_key([{"key": "a"}], None) == [{"key": "a"}]
    with pytest.raises(ValueError):
        merge_by_key([], [{"v": 1}])
