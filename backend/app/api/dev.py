"""Development-only routes: list and inject fixtures, flip mock faults, read the mock outbox.

Mounted by `create_app` only when `app_env != "production"`; a production app answers 404
for every `/api/v1/dev/*` path. Still owner-only and CSRF-checked like everything else.

Faults live on `app.state.faults`, shared by every mock provider this process builds. A
worker running in a separate process has its own `MockFaults`; flipping a fault here does not
reach it.
"""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints
from sqlalchemy import select

from app.api.deps import OWNER_ONLY, SessionDep, SessionmakerDep, SettingsDep, get_faults
from app.db.models import MockOutbox
from app.ingestion.ingest import IngestResult, ingest_message
from app.providers.fixtures import FixtureSummary, list_fixtures, load_fixture
from app.providers.mock_mail import FaultMode, MockFaults

router = APIRouter(prefix="/api/v1/dev", tags=["dev"], dependencies=OWNER_ONLY)

FixtureId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")]


class InjectIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: FixtureId


class InjectedMessage(IngestResult):
    provider_message_id: str


class InjectOut(BaseModel):
    fixture_id: str
    results: list[InjectedMessage]


class FaultsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    send: FaultMode | None = None
    calendar_create: FaultMode | None = None


class FaultsOut(BaseModel):
    send: FaultMode
    calendar_create: FaultMode


class OutboxOut(BaseModel):
    id: str
    operation_id: str
    payload: dict[str, Any]
    created_at: datetime


@router.get("/fixtures")
async def fixtures(settings: SettingsDep) -> list[FixtureSummary]:
    return list_fixtures(settings.fixtures_dir)


@router.post("/inject")
async def inject(body: InjectIn, settings: SettingsDep, maker: SessionmakerDep) -> InjectOut:
    """Ingest every message of the fixture's thread, in order, in one transaction."""
    try:
        fixture = load_fixture(body.fixture_id, settings.fixtures_dir)
    except KeyError:
        raise HTTPException(status_code=404, detail="not_found") from None
    identities = [settings.owner_email, *settings.owner_aliases]
    results = []
    async with maker() as session, session.begin():
        for message in fixture.thread:
            result = await ingest_message(session, message, identities, index=settings.rag_enabled)
            results.append(
                InjectedMessage(
                    **result.model_dump(), provider_message_id=message.provider_message_id
                )
            )
    return InjectOut(fixture_id=fixture.id, results=results)


@router.patch("/faults")
async def patch_faults(
    body: FaultsIn, faults: Annotated[MockFaults, Depends(get_faults)]
) -> FaultsOut:
    if body.send is not None:
        faults.send = body.send
    if body.calendar_create is not None:
        faults.calendar_create = body.calendar_create
    return FaultsOut(send=faults.send, calendar_create=faults.calendar_create)


@router.get("/outbox")
async def outbox(session: SessionDep) -> list[OutboxOut]:
    rows = (
        await session.execute(select(MockOutbox).order_by(MockOutbox.created_at, MockOutbox.id))
    ).scalars()
    return [
        OutboxOut(
            id=str(r.id), operation_id=r.operation_id, payload=r.payload, created_at=r.created_at
        )
        for r in rows
    ]
