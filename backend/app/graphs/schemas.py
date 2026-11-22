"""Structured outputs the model must return at each LLM step of the workflow.

These pydantic models double as JSON schemas sent to Claude (structured outputs) and as code-side
validators. If the reply fails a validator here, the LLM adapter makes one repair attempt.
The `Field(description=...)` texts are part of the prompt: they guide the model field by field.
"""

import re
from datetime import date, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

_EVIDENCE_RE = re.compile(r"(message|availability|thread|owner_input):[A-Za-z0-9._:@-]+")


def _check_evidence(refs: list[str]) -> list[str]:
    for ref in refs:
        if not _EVIDENCE_RE.fullmatch(ref):
            raise ValueError(
                f"bad evidence ref {ref!r}: use message:<id>, availability:<key>, "
                "thread:<id> or owner_input:<id>"
            )
    return refs


EvidenceRefs = Annotated[
    list[str],
    AfterValidator(_check_evidence),
    Field(
        description=(
            "Sources for every fact used, as message:<id>, availability:<key>, thread:<id> "
            "or owner_input:<id>."
        )
    ),
]


class TriageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["ignore", "notify", "respond"] = Field(
        description="ignore: no attention needed; notify: owner should see it; "
        "respond: a reply or meeting is needed."
    )
    urgency: Literal["low", "normal", "high"] = Field(
        description="How time-sensitive it is. Independent of whether a reply is needed."
    )
    reason: str = Field(
        max_length=300, description="One or two short sentences. No reasoning chain."
    )
    evidence_refs: EvidenceRefs
    needs_clarification: bool = Field(
        description="True when the case is uncertain or may conflict with an owner rule; "
        "it is then sent to the owner for review."
    )


class FreeBusyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date = Field(description="First day to check (owner's calendar days).")
    end_date: date = Field(description="Last day to check, at most 14 days after start_date.")

    @model_validator(mode="after")
    def _range(self) -> "FreeBusyArgs":
        if not 0 <= (self.end_date - self.start_date).days <= 14:
            raise ValueError("end_date must be 0 to 14 days after start_date")
        return self


class NextStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "read_thread", "get_freebusy", "ask_owner", "draft_reply", "draft_meeting", "finish"
    ] = Field(description="The single next step to take.")
    reason: str = Field(
        max_length=300, description="One or two short sentences. No reasoning chain."
    )
    freebusy: FreeBusyArgs | None = Field(
        default=None, description="Required when action is get_freebusy; otherwise null."
    )
    question: str | None = Field(
        default=None, description="Required when action is ask_owner: the question for the owner."
    )

    @model_validator(mode="after")
    def _required_args(self) -> "NextStep":
        if self.action == "get_freebusy" and self.freebusy is None:
            raise ValueError("get_freebusy requires freebusy")
        if self.action == "ask_owner" and not self.question:
            raise ValueError("ask_owner requires question")
        return self


class ReplyDraft(BaseModel):
    """Subject and threading headers are derived by code, never by the model."""

    model_config = ConfigDict(extra="forbid")

    to: list[str] = Field(description="Recipient email addresses.")
    cc: list[str] = Field(description="Cc email addresses; empty if none.")
    body_text: str = Field(description="Plain-text reply body.")
    evidence_refs: EvidenceRefs


class MeetingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Event title.")
    start_local: datetime = Field(
        description="Start as local wall-clock time WITHOUT offset, e.g. 2026-11-24T10:00:00."
    )
    end_local: datetime = Field(description="End as local wall-clock time WITHOUT offset.")
    time_zone: str = Field(
        description="IANA time zone of start_local/end_local, e.g. Europe/Berlin."
    )
    attendees: list[str] = Field(description="Attendee email addresses.")
    location: str | None = Field(default=None, description="Location or link; null if none.")
    send_updates: Literal["all", "none"] = Field(
        description="all: email invitations to attendees; none: do not."
    )
    evidence_refs: EvidenceRefs

    @field_validator("start_local", "end_local")
    @classmethod
    def _naive(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("give local time without an offset; put the zone in time_zone")
        return v

    @field_validator("time_zone")
    @classmethod
    def _zone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"unknown IANA time zone {v!r}") from exc
        return v
