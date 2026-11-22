"""`GET /events`: server-sent events from the append-only `workflow_events` table.

The cursor is the last event id the client has seen: `Last-Event-ID` (sent by the browser's
`EventSource` when it reconnects) or else `?after=`. The stream first replays every row with
`id > cursor`, then polls once a second, each time in a short new session, so no
transaction stays open while the client is idle. Each SSE message has `id:` = the row id and
`data:` = `{id, workflow_id, type, created_at, payload}`. sse-starlette sends a `: ping`
comment every 15 s (keepalive) and stops the generator when the client disconnects.

The table is the record; the stream is only a notification channel. After reconnecting, the
UI should re-read `GET /workflows/{id}` for the authoritative state (spec §12).

`once=true` (dev/test only) ends the stream after the replay, so a test client that reads
the whole response body terminates.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sse_starlette import EventSourceResponse, ServerSentEvent

from app.api.deps import OWNER_ONLY, SessionmakerDep, SettingsDep
from app.api.views import EventOut
from app.db.models import WorkflowEvent

router = APIRouter(prefix="/api/v1", tags=["events"], dependencies=OWNER_ONLY)

POLL_SECONDS = 1.0
PING_SECONDS = 15
BATCH = 500


async def _batch(
    maker: async_sessionmaker[AsyncSession], cursor: int, workflow_id: uuid.UUID | None
) -> list[WorkflowEvent]:
    stmt = select(WorkflowEvent).where(WorkflowEvent.id > cursor)
    if workflow_id is not None:
        stmt = stmt.where(WorkflowEvent.workflow_id == workflow_id)
    async with maker() as session:
        return list((await session.execute(stmt.order_by(WorkflowEvent.id).limit(BATCH))).scalars())


async def _stream(
    request: Request,
    maker: async_sessionmaker[AsyncSession],
    cursor: int,
    workflow_id: uuid.UUID | None,
    once: bool,
) -> AsyncIterator[ServerSentEvent]:
    while True:
        rows = await _batch(maker, cursor, workflow_id)
        for row in rows:
            cursor = row.id
            yield ServerSentEvent(data=EventOut.of(row).model_dump_json(), id=str(row.id))
        if len(rows) == BATCH:
            continue  # more to replay
        if once or await request.is_disconnected():
            return
        await asyncio.sleep(POLL_SECONDS)


def _cursor(after: int | None, last_event_id: str | None) -> int:
    if last_event_id:
        try:
            return max(int(last_event_id), 0)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid_last_event_id") from None
    return after or 0


@router.get("/events")
async def events(
    request: Request,
    maker: SessionmakerDep,
    settings: SettingsDep,
    after: Annotated[int | None, Query(ge=0)] = None,
    workflow_id: uuid.UUID | None = None,
    once: bool = False,
    last_event_id: Annotated[str | None, Header(max_length=20)] = None,
) -> EventSourceResponse:
    if once and settings.app_env == "production":
        raise HTTPException(status_code=422, detail="once_not_allowed")
    cursor = _cursor(after, last_event_id)
    return EventSourceResponse(
        _stream(request, maker, cursor, workflow_id, once), ping=PING_SECONDS
    )
