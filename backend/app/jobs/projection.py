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

Trust the innermost level. A subgraph's interrupt is also listed on the parent's `response`
task (and in the parent's `snap.interrupts`), and the parent keeps listing it until the whole
subgraph node finishes: the parent only checkpoints between its own steps. So after a resume
that answered the interrupt and then crashed inside the subgraph (say, during `preflight`),
the parent still shows the interrupt as pending while the subgraph, which has moved on,
shows none. Only the subgraph's view is current, so for a task that has a subgraph state the
walk uses that state and ignores the parent's copy. An interrupt raised by a parent node
itself (`notification_interrupt`) is on a task without a subgraph state, so it is read there.

The invariant this relies on: interrupts are raised only by leaf nodes (nodes that are plain
functions), and every subgraph is added to its parent directly as a compiled graph
(`g.add_node("response", build_response_graph().compile())`). LangGraph then recognises the
node as a subgraph and `subgraphs=True` fills `task.state`. If a subgraph were wrapped in a
function node (`async def response(state): return await sub.ainvoke(state)`), `task.state`
would stay empty, that task would look like a leaf, and the walk would trust the parent's
possibly stale copy of the subgraph's interrupt. Do not wrap subgraphs.

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
    """Every interrupt still waiting for an answer, read at the innermost level.

    Interrupts are collected from tasks without a subgraph; a task running a subgraph is
    represented by the subgraph's own snapshot (see the module docstring). De-duplicated by id.
    """
    found: dict[str, Interrupt] = {}

    def walk(s: StateSnapshot) -> None:
        for task in s.tasks:
            if isinstance(task.state, StateSnapshot):
                walk(task.state)
            else:
                for interrupt in task.interrupts:
                    found.setdefault(interrupt.id, interrupt)

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
