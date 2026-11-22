"""`triage` node and its router `route_after_triage`.

Node: one structured LLM call (purpose `triage`, prompt `triage_v1`) returns a
`TriageDecision`; the node stores classification, urgency, a short reason and evidence refs.

Uncertain cases go to the owner (spec §6: "if classification is uncertain … route to review"):
a refusal, output that stays malformed after the adapter's repair attempt, or
`needs_clarification=True` all become a `notify` with the reason prefixed "Needs review: ".
`BudgetExceeded` pauses the run. `LLMTransientError` is not caught: it propagates out of the
graph so the worker can retry later (the checkpoint before this node is kept).

`record_triage` is a separate node after it: side effects live in their own nodes. It copies
the classification and reason onto `workflow_runs` (an idempotent UPDATE, safe to re-run) so
the review queue and workflow lists can show them without reading checkpoints.

Conditional edge: `route_after_triage(state)` is a pure function of state that names the next
node; `add_conditional_edges` maps its return values to nodes.
"""

from typing import Any, Literal
from uuid import UUID

from langgraph.runtime import Runtime
from sqlalchemy import update

from app.db.models import WorkflowRun
from app.graphs.deps import GraphDeps
from app.graphs.nodes.shared import preferences_context, user_content
from app.graphs.schemas import TriageDecision
from app.graphs.state import EmailState
from app.llm.client import BudgetExceeded, LLMCall, LLMMalformedOutput, LLMRefusal
from app.prompts import load_prompt

NEEDS_REVIEW = "Needs review: "


def _needs_review(reason: str) -> dict[str, Any]:
    return {
        "classification": "notify",
        "urgency": "normal",
        "decision_summary": NEEDS_REVIEW + reason,
        "evidence_refs": [],
    }


async def triage(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Classify the thread as ignore, notify or respond."""
    prompt = load_prompt("triage")
    call = LLMCall(
        purpose="triage",
        system=prompt.system,
        cached_context=preferences_context(state),
        user_content=user_content(state),
        output_model=TriageDecision,
        workflow_id=UUID(state["workflow_id"]),
        script_key=state["provider_message_id"],
    )
    try:
        decision = (await runtime.context.llm.structured(call)).output
    except BudgetExceeded:
        return {"status": "paused", "error_code": "budget"}
    except LLMRefusal:
        return _needs_review("the model declined to triage this email.")
    except LLMMalformedOutput:
        return _needs_review("the model's triage output was invalid.")

    if decision.needs_clarification:
        outcome = _needs_review(decision.reason)
        outcome["urgency"] = decision.urgency
        outcome["evidence_refs"] = list(decision.evidence_refs)
        return outcome
    return {
        "classification": decision.classification,
        "urgency": decision.urgency,
        "decision_summary": decision.reason,
        "evidence_refs": list(decision.evidence_refs),
    }


async def record_triage(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Store triage's classification and reason on the run (skipped when paused)."""
    if state.get("status") == "paused" or not state.get("classification"):
        return {}
    async with runtime.context.sessionmaker() as session, session.begin():
        await session.execute(
            update(WorkflowRun)
            .where(WorkflowRun.id == UUID(state["workflow_id"]))
            .values(
                classification=state.get("classification"),
                triage_reason=state.get("decision_summary"),
            )
        )
    return {}


def route_after_triage(
    state: EmailState,
) -> Literal["finalize", "persist_notification", "response"]:
    if state.get("status") == "paused":
        return "finalize"
    classification = state.get("classification")
    if classification == "respond":
        return "response"
    if classification == "notify":
        return "persist_notification"
    return "finalize"
