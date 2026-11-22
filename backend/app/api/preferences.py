"""Versioned owner preferences per namespace (`triage`, `writing`, `scheduling`).

Each change inserts revision n+1 as `active` and marks the previous active row `superseded`,
so the history stays and a proposal can cite the revision it was drafted under. The caller
sends the revision it last saw (`expected_revision`, 0 for none); a different current
revision is a 409, so two tabs cannot silently overwrite each other. Values are validated
per namespace and stored in the validated, normalized form.
"""

import uuid
from datetime import datetime, time
from typing import Annotated, Any, Literal, Self
from zoneinfo import available_timezones

from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.actions.decisions import DecisionConflict
from app.api.deps import OWNER_ONLY, OwnerId, SessionDep, SessionmakerDep
from app.db.models import PreferenceVersion

router = APIRouter(prefix="/api/v1/preferences", tags=["preferences"], dependencies=OWNER_ONLY)

Namespace = Literal["triage", "writing", "scheduling"]
_HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]


class TriagePrefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vip: list[Annotated[str, StringConstraints(max_length=320)]] = Field(default=[], max_length=200)
    ignore_senders: list[Annotated[str, StringConstraints(max_length=320)]] = Field(
        default=[], max_length=200
    )
    notes: Annotated[str, StringConstraints(max_length=2000)] | None = None


class WritingPrefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tone: ShortText | None = None
    language: Annotated[str, StringConstraints(strip_whitespace=True, max_length=50)] | None = None
    signature: Annotated[str, StringConstraints(max_length=1000)] | None = None


class SchedulingPrefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_zone: str = "UTC"
    work_start: Annotated[str, StringConstraints(pattern=_HHMM)] = "09:00"
    work_end: Annotated[str, StringConstraints(pattern=_HHMM)] = "17:00"
    work_days: list[Annotated[int, Field(ge=0, le=6)]] = Field(
        default=[0, 1, 2, 3, 4], min_length=1, max_length=7
    )
    buffer_minutes: Annotated[int, Field(ge=0, le=240)] = 10
    default_duration_minutes: Annotated[int, Field(ge=5, le=480)] = 30

    @field_validator("time_zone")
    @classmethod
    def _iana_zone(cls, value: str) -> str:
        if value not in available_timezones():
            raise ValueError("unknown IANA time zone")
        return value

    @field_validator("work_days")
    @classmethod
    def _unique_days(cls, value: list[int]) -> list[int]:
        return sorted(set(value))

    @model_validator(mode="after")
    def _start_before_end(self) -> Self:
        if time.fromisoformat(self.work_start) >= time.fromisoformat(self.work_end):
            raise ValueError("work_start must be before work_end")
        return self


_MODELS: dict[str, type[BaseModel]] = {
    "triage": TriagePrefs,
    "writing": WritingPrefs,
    "scheduling": SchedulingPrefs,
}


class PreferenceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: dict[str, Any]
    expected_revision: Annotated[int, Field(ge=0)]


class PreferenceOut(BaseModel):
    namespace: str
    revision: int
    value: dict[str, Any]
    created_at: datetime


def _validated(namespace: str, value: dict[str, Any]) -> dict[str, Any]:
    """The value in its validated, normalized form; a 422 like any request validation error."""
    try:
        return _MODELS[namespace].model_validate(value).model_dump(mode="json")
    except ValidationError as exc:
        raise RequestValidationError(
            [{**e, "loc": ("body", "value", *e["loc"])} for e in exc.errors()]
        ) from None


def _out(row: PreferenceVersion) -> PreferenceOut:
    return PreferenceOut(
        namespace=row.namespace, revision=row.revision, value=row.value, created_at=row.created_at
    )


@router.get("")
async def preferences(owner_id: OwnerId, session: SessionDep) -> dict[str, PreferenceOut]:
    """The active revision per namespace (namespaces never set are absent)."""
    rows = (
        await session.execute(
            select(PreferenceVersion)
            .where(PreferenceVersion.owner_id == owner_id, PreferenceVersion.status == "active")
            .order_by(PreferenceVersion.namespace, PreferenceVersion.revision.desc())
        )
    ).scalars()
    found: dict[str, PreferenceOut] = {}
    for row in rows:
        found.setdefault(row.namespace, _out(row))
    return found


@router.patch("/{namespace}")
async def update_preference(
    namespace: str, body: PreferenceIn, owner_id: OwnerId, maker: SessionmakerDep
) -> PreferenceOut:
    if namespace not in _MODELS:
        raise HTTPException(status_code=404, detail="not_found")
    value = _validated(namespace, body.value)
    try:
        async with maker() as session, session.begin():
            current = (
                (
                    await session.execute(
                        select(PreferenceVersion)
                        .where(
                            PreferenceVersion.owner_id == owner_id,
                            PreferenceVersion.namespace == namespace,
                            PreferenceVersion.status == "active",
                        )
                        .order_by(PreferenceVersion.revision.desc())
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )
            latest = await session.scalar(
                select(PreferenceVersion.revision)
                .where(
                    PreferenceVersion.owner_id == owner_id,
                    PreferenceVersion.namespace == namespace,
                )
                .order_by(PreferenceVersion.revision.desc())
                .limit(1)
            )
            active_revision = current[0].revision if current else 0
            if active_revision != body.expected_revision:
                raise DecisionConflict("revision_mismatch")
            for row in current:
                row.status = "superseded"
            new = PreferenceVersion(
                id=uuid.uuid4(),
                owner_id=owner_id,
                namespace=namespace,
                revision=(latest or 0) + 1,
                value=value,
                status="active",
                previous_revision=active_revision or None,
            )
            session.add(new)
            await session.flush()
            await session.refresh(new)
            return _out(new)
    except IntegrityError:
        # Two first-time writes raced to the same (owner, namespace, revision).
        raise DecisionConflict("revision_mismatch") from None
