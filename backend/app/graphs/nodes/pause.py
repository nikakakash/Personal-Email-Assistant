"""Recoverable pauses: `persist_pause`, `pause_interrupt` and their routers.

Some stops are not the end of the work, only a wait for a condition the owner controls: the
daily model budget is used up (`budget`), the planner used its step allowance (`step_budget`),
outbound sending is off (`outbound_disabled`), or a provider read for preflight failed
(`preflight_unavailable`). Ending the graph there would leave nothing to continue (spec §10:
"pauses the workflow with an explanation and a manual continuation option").

So such a stop routes to `persist_pause` → `pause_interrupt`, the same two-node pattern as
the other human-in-the-loop waits:

- `persist_pause` records the pause on the run (status `paused`, the reason as `error_code`,
  an audit event and a `run_paused` event). It is a compare-and-set that skips a final run
  (cancelled meanwhile); then it sets `run_final` and the router ends the graph instead.
- `pause_interrupt` calls only `interrupt({"type": "pause", "reason": ...})`. The owner's
  Retry (`POST /workflows/{id}/retry`) resumes it with `{"decision": "retry"}`; a cancel
  resumes it with `{"decision": "cancelled"}`. Resuming re-runs this node from its top, so it
  does no I/O: it only turns the resume value into a state update (clearing the pause, and for
  `step_budget` granting a fresh allowance via `step_base`).

Where a retry continues (`route_after_pause`, pure):

- A pausing node may name the node to continue at in `resume_at`. `mark_stale` does
  (`plan_next_step`): it has already retired the proposal, so there is nothing to
  re-check.
- Otherwise `outbound_disabled`, `preflight_unavailable` → `preflight`, with the same
  approval. Every
  check runs again (approval still accepting, TTL, thread unchanged, slot free), so a retry
  never sends what is no longer valid, and the model is not called for that payload.
- `budget`, `step_budget` (response subgraph) → `plan_next_step`.
- `budget` at triage (parent graph, `route_after_parent_pause`) → `triage`.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from sqlalchemy import update

from app.db.models import AuditEvent, WorkflowRun
from app.graphs.deps import GraphDeps
from app.graphs.events import emit
from app.graphs.nodes.finalize import FINAL_STATUSES
from app.graphs.state import EmailState

# Reasons that pause (and can be retried) rather than fail.
PAUSE_REASONS = ("budget", "step_budget", "outbound_disabled", "preflight_unavailable")
_BACK_TO_PREFLIGHT = ("outbound_disabled", "preflight_unavailable")


def is_pause(state: EmailState) -> bool:
    """Whether the state describes a recoverable stop (for routers)."""
    return state.get("status") == "paused" and state.get("error_code") in PAUSE_REASONS


async def persist_pause(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Mark the run paused with its reason; idempotent, and never un-finalizes a run."""
    workflow_id = uuid.UUID(state["workflow_id"])
    reason = state.get("error_code")
    async with runtime.context.sessionmaker() as session, session.begin():
        updated = (
            await session.execute(
                update(WorkflowRun)
                .where(WorkflowRun.id == workflow_id, WorkflowRun.status.not_in(FINAL_STATUSES))
                .values(
                    status="paused",
                    error_code=reason,
                    step_count=state.get("step_count", 0),
                    updated_at=datetime.now(UTC),
                )
                .returning(WorkflowRun.id)
            )
        ).scalar_one_or_none()
        if updated is None:
            return {"run_final": True}
        detail = {"status": "paused", "error_code": reason}
        session.add(AuditEvent(workflow_id=workflow_id, event_type="run_paused", detail=detail))
        await emit(session, workflow_id, "run_paused", detail)
    return {}


def route_after_persist_pause(state: EmailState) -> Literal["pause_interrupt", "end"]:
    return "end" if state.get("run_final") else "pause_interrupt"


def pause_interrupt(state: EmailState) -> dict[str, Any]:
    """Wait for Retry or cancel. Only `interrupt()` and pure state: it re-runs on resume."""
    reason = state.get("error_code")
    value = interrupt({"type": "pause", "reason": reason})
    decision = value.get("decision") if isinstance(value, dict) else None
    if decision != "retry":
        return {"next_action": "cancelled"}
    update: dict[str, Any] = {
        "status": "running",
        "error_code": None,
        "next_action": "retry",
        # Where the retry continues, decided now (pure) and read by `route_after_pause`.
        "retry_target": state.get("resume_at")
        or ("preflight" if reason in _BACK_TO_PREFLIGHT else "plan_next_step"),
        "resume_at": None,
    }
    if reason == "step_budget":
        update["step_base"] = state.get("step_count", 0)
    if reason in _BACK_TO_PREFLIGHT:
        update["preflight"] = None
    return update


def route_after_pause(state: EmailState) -> Literal["preflight", "plan_next_step", "finish"]:
    if state.get("next_action") != "retry":
        return "finish"
    return "preflight" if state.get("retry_target") == "preflight" else "plan_next_step"


def route_after_parent_pause(state: EmailState) -> Literal["triage", "finalize"]:
    return "triage" if state.get("next_action") == "retry" else "finalize"
