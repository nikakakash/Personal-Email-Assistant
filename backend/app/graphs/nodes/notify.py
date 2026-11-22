"""Notifications: `persist_notification`, `notification_interrupt`, `route_after_notification`.

Same split as `ask.py`: the side effect (an idempotent review-item insert with the
deterministic id `uuid5(workflow_id, "notification")`) lives in its own node; the interrupt
node holds only `interrupt()`, because LangGraph re-runs an interrupted node from its top when
the graph is resumed with `Command(resume=...)`.

Resume values: `{"decision": "dismiss"}`, `{"decision": "respond"}` (the owner wants a reply
after all, so the run enters the response subgraph), `{"decision": "superseded"}` or
`{"decision": "cancelled"}`.
"""

import uuid
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.graphs.deps import GraphDeps
from app.graphs.nodes.ask import persist_review_item
from app.graphs.state import EmailState


def notification_item_id(workflow_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.UUID(workflow_id), "notification")


async def persist_notification(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Store the notification as a pending review item."""
    item_id = notification_item_id(state["workflow_id"])
    opened = await persist_review_item(
        runtime.context,
        state["workflow_id"],
        item_id,
        "notification",
        {
            "classification": state.get("classification"),
            "urgency": state.get("urgency"),
            "reason": state.get("decision_summary"),
            "evidence_refs": state.get("evidence_refs") or [],
        },
    )
    if not opened:
        return {"run_final": True}  # cancelled meanwhile: no item, no interrupt
    return {"review_item_id": str(item_id)}


def route_after_persist_notification(
    state: EmailState,
) -> Literal["notification_interrupt", "finalize"]:
    return "finalize" if state.get("run_final") else "notification_interrupt"


def notification_interrupt(state: EmailState) -> dict[str, Any]:
    """Pause until the owner dismisses the notification or asks for a response."""
    value = interrupt({"type": "notification", "review_item_id": state["review_item_id"]})
    return {"review": value}


def route_after_notification(
    state: EmailState,
) -> Literal["response", "finalize", "unknown_decision"]:
    decision = (state.get("review") or {}).get("decision")
    if decision == "respond":
        return "response"
    if decision in ("dismiss", "superseded", "cancelled"):
        return "finalize"
    return "unknown_decision"
