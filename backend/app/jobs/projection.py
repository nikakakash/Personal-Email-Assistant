"""Interrupt projection: copy the graph's pending interrupts onto review items.

Why this exists. When a node calls `interrupt(value)`, LangGraph stops the run and stores the
pause *in the checkpoint*, not in our tables. Each pending interrupt has an `id`. Resuming it
means `graph.ainvoke(Command(resume=...), config)` on the same `thread_id`. The owner's
decision API (`decide_proposal`) must name the interrupt it answers, so that a click on an old
screen cannot resume a different, later pause (spec §9). The API never reads checkpoints, so
after every worker run this module copies each pending interrupt's id onto the review item the
interrupt is about: `ReviewItem.interrupt_id = interrupt.id`.

Finding the interrupts. Read the checkpoint with `await graph.aget_state(config,
subgraphs=True)`. It returns a `StateSnapshot`:

- `snap.next`: the nodes that would run next. Empty means the run is finished (or never
  started); non-empty with no pending interrupt means it stopped mid-way (a crash or an
  exception) and `ainvoke(None, config)` continues it.
- `snap.tasks`: one `PregelTask` per pending node. `task.interrupts` holds its pending
  `Interrupt`s (`.id`, `.value`). For a subgraph node (our `response` node), `subgraphs=True`
  also loads the subgraph's own checkpoint into `task.state`, another `StateSnapshot` with its
  own tasks, so the walk below recurses into it.

While the run is paused inside the subgraph, the parent's `snap.values` do not include the
subgraph's updates (they flow back only when the subgraph finishes), so the review item id is
read from the interrupt's value, which every interrupt node sets:
`{"type": ..., "review_item_id": ...}`.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Interrupt, StateSnapshot
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ReviewItem, WorkflowRun
from app.graphs.nodes.finalize import FINAL_STATUSES


def pending_interrupts(snap: StateSnapshot) -> list[Interrupt]:
    """Every interrupt still waiting for an answer, from the parent and all subgraphs.

    The same interrupt can appear at several levels (a subgraph's interrupt is also listed on
    the parent task that runs the subgraph), so the result is de-duplicated by id.
    """
    found: dict[str, Interrupt] = {}

    def walk(s: StateSnapshot) -> None:
        for interrupt in s.interrupts:
            found.setdefault(interrupt.id, interrupt)
        for task in s.tasks:
            for interrupt in task.interrupts:
                found.setdefault(interrupt.id, interrupt)
            if isinstance(task.state, StateSnapshot):
                walk(task.state)

    walk(snap)
    return list(found.values())


def _review_item_id(interrupt: Interrupt) -> uuid.UUID | None:
    value: Any = interrupt.value
    raw = value.get("review_item_id") if isinstance(value, dict) else None
    try:
        return uuid.UUID(str(raw)) if raw is not None else None
    except ValueError:
        return None


async def project_interrupts(
    session: AsyncSession, workflow_id: uuid.UUID, snap: StateSnapshot
) -> list[Interrupt]:
    """Copy pending interrupt ids onto their review items; mark the run `waiting_review`.

    Only an item of this workflow that is still `pending` is updated (a resolved or stale item
    keeps the id it was decided under). The run status changes only if it is not final.
    Flushes but does not commit. Returns the pending interrupts.
    """
    interrupts = pending_interrupts(snap)
    for interrupt in interrupts:
        item_id = _review_item_id(interrupt)
        if item_id is None:
            continue
        await session.execute(
            update(ReviewItem)
            .where(
                ReviewItem.id == item_id,
                ReviewItem.workflow_id == workflow_id,
                ReviewItem.status == "pending",
            )
            .values(interrupt_id=interrupt.id)
        )
    if interrupts:
        await session.execute(
            update(WorkflowRun)
            .where(WorkflowRun.id == workflow_id, WorkflowRun.status.not_in(FINAL_STATUSES))
            .values(status="waiting_review", updated_at=datetime.now(UTC))
        )
    return interrupts
