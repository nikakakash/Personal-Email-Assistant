"""Asking the owner a question: `persist_question`, `question_interrupt`, `record_answer`.

Why three nodes? `interrupt()` pauses the graph: LangGraph saves a checkpoint and the run
returns with `result["__interrupt__"]`. Later the app resumes the same `thread_id` with
`graph.ainvoke(Command(resume=value), config, ...)`, and `interrupt()` returns `value`. But on
resume the interrupted node runs again **from its top**: anything before the `interrupt()` call
executes a second time. So:

1. `persist_question` writes the review item (idempotent: a deterministic id plus
   `ON CONFLICT DO NOTHING`, so a re-run writes nothing new). It finishes before the pause, and
   its output is checkpointed, so it does not run again on resume.
2. `question_interrupt` contains only the `interrupt()` call and returns the owner's reply as
   `{"review": value}`. Re-running it is harmless because it has no side effects.
3. `record_answer` turns the reply into an `owner_inputs` entry. `owner_inputs` has an
   `operator.add` reducer, so the returned one-item list is appended, not overwritten.

Resume values (sent by the API): `{"decision": "answer", "answer": "<text>"}`, or
`{"decision": "superseded"}` / `{"decision": "cancelled"}`.

`persist_review_item` is shared with `notify.py`.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert

from app.db.models import ReviewItem, WorkflowRun
from app.graphs.deps import GraphDeps
from app.graphs.events import emit
from app.graphs.nodes.finalize import FINAL_STATUSES
from app.graphs.state import EmailState


async def persist_review_item(
    deps: GraphDeps,
    workflow_id: str,
    item_id: uuid.UUID,
    kind: Literal["notification", "question"],
    payload: dict[str, Any],
) -> None:
    """Insert a pending review item once; set the run to `waiting_review` unless it is final.

    `approval_required` is emitted only by the call that actually inserted the row.
    """
    async with deps.sessionmaker() as session, session.begin():
        inserted = (
            await session.execute(
                insert(ReviewItem)
                .values(
                    id=item_id,
                    workflow_id=uuid.UUID(workflow_id),
                    kind=kind,
                    status="pending",
                    payload=payload,
                )
                .on_conflict_do_nothing(index_elements=[ReviewItem.id])
                .returning(ReviewItem.id)
            )
        ).scalar_one_or_none()
        await session.execute(
            update(WorkflowRun)
            .where(
                WorkflowRun.id == uuid.UUID(workflow_id),
                WorkflowRun.status.not_in(FINAL_STATUSES),
            )
            .values(status="waiting_review", updated_at=datetime.now(UTC))
        )
        if inserted is not None:
            await emit(
                session,
                workflow_id,
                "approval_required",
                {"review_item_id": str(item_id), "kind": kind},
            )


def question_item_id(workflow_id: str, step_count: int) -> uuid.UUID:
    return uuid.uuid5(uuid.UUID(workflow_id), f"question:{step_count}")


async def persist_question(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Store the planner's question as a pending review item."""
    item_id = question_item_id(state["workflow_id"], state.get("step_count", 0))
    await persist_review_item(
        runtime.context,
        state["workflow_id"],
        item_id,
        "question",
        {"question": state.get("pending_question") or ""},
    )
    return {"review_item_id": str(item_id)}


def question_interrupt(state: EmailState) -> dict[str, Any]:
    """Pause until the owner answers. Nothing else belongs here: it re-runs on resume."""
    value = interrupt({"type": "question", "review_item_id": state["review_item_id"]})
    return {"review": value}


def route_after_question(
    state: EmailState,
) -> Literal["record_answer", "finish", "unknown_decision"]:
    decision = (state.get("review") or {}).get("decision")
    if decision == "answer":
        return "record_answer"
    if decision in ("superseded", "cancelled"):
        return "finish"
    return "unknown_decision"


async def record_answer(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Append the owner's answer to `owner_inputs` (trusted input the planner can cite)."""
    answer = str((state.get("review") or {}).get("answer") or "")
    return {"owner_inputs": [{"kind": "answer", "text": answer}], "pending_question": None}
