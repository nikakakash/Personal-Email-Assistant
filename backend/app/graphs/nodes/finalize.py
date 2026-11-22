"""Ending a run: `unknown_decision`, `finish` (response subgraph exit) and `finalize` (parent).

`finish` is the single exit node of the response subgraph. When a subgraph reaches END, its
final state flows back into the parent (both use `EmailState`) and the parent continues with
its next edge, `response → finalize`. `finish` only normalizes `next_action` so that
`finalize` can tell an error ending from a normal one.

`unknown_decision` handles a resume value no router recognizes. Routers must stay pure (they
cannot write state), so they route here and this node records `error_code="unknown_decision"`.

`finalize` writes the user-visible outcome to `workflow_runs`, first match wins:

1. `failed` with `error_code="outcome_unknown"`: some action's outcome is unknown (even if
   another one succeeded), so the owner must check the provider;
2. `paused`: state status is paused;
3. `failed`: an error ending (`error_code` set and `next_action == "error"`);
4. `done_with_actions`: an action succeeded;
5. `done_no_action`.

It also stores `step_count` and `error_code`, and writes an
audit event plus `run_completed` / `run_failed`. The update is a compare-and-set that skips a
run already in a final status, so a re-run (or a resume replayed from an old checkpoint) does
not write a second audit event.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.runtime import Runtime
from sqlalchemy import update

from app.db.models import AuditEvent, WorkflowRun
from app.graphs.deps import GraphDeps
from app.graphs.events import emit
from app.graphs.state import EmailState

FINAL_STATUSES = ("done_no_action", "done_with_actions", "failed", "cancelled")


async def unknown_decision(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Record that the resume value was not a decision this interrupt accepts."""
    return {"error_code": "unknown_decision", "next_action": "error"}


async def finish(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Leave the response subgraph; an `error_code` (unless paused) marks an error ending."""
    if state.get("error_code") and state.get("status") != "paused":
        return {"next_action": "error"}
    if state.get("status") == "paused":
        return {}
    return {"next_action": "finish"}


def _outcome_unknown(state: EmailState) -> bool:
    return any(r.get("status") == "outcome_unknown" for r in state.get("action_results") or [])


def run_status(state: EmailState) -> str:
    if _outcome_unknown(state):
        return "failed"  # needs the owner's review, even if another action succeeded
    if state.get("status") == "paused":
        return "paused"
    if state.get("error_code") and state.get("next_action") == "error":
        return "failed"
    if any(r.get("status") == "succeeded" for r in state.get("action_results") or []):
        return "done_with_actions"
    return "done_no_action"


async def finalize(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Persist the run's final status, audit it and emit the completion event."""
    status = run_status(state)
    workflow_id = uuid.UUID(state["workflow_id"])
    error_code = "outcome_unknown" if _outcome_unknown(state) else state.get("error_code")
    step_count = state.get("step_count", 0)
    async with runtime.context.sessionmaker() as session, session.begin():
        updated = (
            await session.execute(
                update(WorkflowRun)
                .where(WorkflowRun.id == workflow_id, WorkflowRun.status.not_in(FINAL_STATUSES))
                .values(
                    status=status,
                    step_count=step_count,
                    error_code=error_code,
                    updated_at=datetime.now(UTC),
                )
                .returning(WorkflowRun.id)
            )
        ).scalar_one_or_none()
        if updated is not None:
            event_type = "run_failed" if status == "failed" else "run_completed"
            detail = {
                "status": status,
                "step_count": step_count,
                "error_code": error_code,
                "classification": state.get("classification"),
                "actions": [
                    {"proposal_id": r.get("proposal_id"), "status": r.get("status")}
                    for r in state.get("action_results") or []
                ],
            }
            session.add(AuditEvent(workflow_id=workflow_id, event_type=event_type, detail=detail))
            await emit(session, workflow_id, event_type, detail)
    return {"status": status}
