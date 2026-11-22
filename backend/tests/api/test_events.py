"""`GET /events`: server-sent events replayed from `workflow_events` after a cursor."""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import WorkflowEvent, WorkflowRun

WorkflowFactory = Callable[[], Awaitable[WorkflowRun]]


def _parse_sse(text: str) -> list[dict[str, Any]]:
    """Each `id:`/`data:` block of the stream as {"id": ..., "data": parsed JSON}."""
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.split("\n") if line and not line.startswith(":")
        )
        if "data" in fields:
            events.append({"id": fields.get("id"), "data": json.loads(fields["data"])})
    return events


async def _events(session: AsyncSession, run: WorkflowRun, n: int) -> list[WorkflowEvent]:
    rows = [
        WorkflowEvent(workflow_id=run.id, event_type="node_started", payload={"node": f"n{i}"})
        for i in range(n)
    ]
    session.add_all(rows)
    await session.commit()
    return rows


async def test_sse_replays_after_cursor(
    authed: AsyncClient, session: AsyncSession, workflow_factory: WorkflowFactory
) -> None:
    run = await workflow_factory()
    first, second, third = await _events(session, run, 3)

    r = await authed.get("/api/v1/events", params={"after": first.id, "once": "true"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert [e["id"] for e in events] == [str(second.id), str(third.id)]
    data = events[0]["data"]
    assert data == {
        "id": second.id,
        "workflow_id": str(run.id),
        "type": "node_started",
        "created_at": data["created_at"],
        "payload": {"node": "n1"},
    }


async def test_sse_honours_last_event_id_and_workflow_filter(
    authed: AsyncClient, session: AsyncSession, workflow_factory: WorkflowFactory
) -> None:
    run, other = await workflow_factory(), await workflow_factory()
    first, second = await _events(session, run, 2)
    await _events(session, other, 1)

    r = await authed.get(
        "/api/v1/events",
        params={"workflow_id": str(run.id), "once": "true"},
        headers={"Last-Event-ID": str(first.id)},
    )
    assert [e["id"] for e in _parse_sse(r.text)] == [str(second.id)]


async def test_sse_requires_owner(anon: AsyncClient) -> None:
    assert (await anon.get("/api/v1/events", params={"once": "true"})).status_code == 401
