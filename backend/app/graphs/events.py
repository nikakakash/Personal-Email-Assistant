"""Workflow events: `emit` writes one row to the append-only `workflow_events` log.

The log feeds the UI's live progress view (server-sent events). Payloads are small and
structural (node names, ids, finding keys, statuses); they never contain email text.

`traced(name, node)` wraps a node so it first records `node_started`, in its own short
transaction. LangGraph only sees a plain async node with the usual `(state, runtime)`
signature: it injects the runtime because the parameter is named `runtime`. Interrupt nodes
are deliberately not wrapped: they contain nothing but the `interrupt()` call (see `ask.py`).
"""

import uuid
from collections.abc import Awaitable
from typing import Any, Protocol

from langgraph.runtime import Runtime
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import WorkflowEvent
from app.graphs.deps import GraphDeps
from app.graphs.state import EmailState


class Node(Protocol):
    """An async node taking `(state, runtime)`, as LangGraph calls it."""

    def __call__(
        self, state: EmailState, runtime: Runtime[GraphDeps]
    ) -> Awaitable[dict[str, Any]]: ...


async def emit(
    session: AsyncSession,
    workflow_id: str | uuid.UUID,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Add a workflow event to `session`. Does not commit; the caller owns the transaction."""
    session.add(
        WorkflowEvent(
            workflow_id=uuid.UUID(str(workflow_id)), event_type=event_type, payload=payload
        )
    )
    await session.flush()


async def emit_now(
    deps: GraphDeps, workflow_id: str, events: list[tuple[str, dict[str, Any]]]
) -> None:
    """Write `events` in one short transaction of their own."""
    if not events:
        return
    async with deps.sessionmaker() as session, session.begin():
        for event_type, payload in events:
            await emit(session, workflow_id, event_type, payload)


def traced(name: str, node: Node) -> Node:
    """Wrap `node` so it emits `node_started` before it runs."""

    async def run(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
        await emit_now(runtime.context, state["workflow_id"], [("node_started", {"node": name})])
        return await node(state, runtime)

    run.__name__ = name
    return run
