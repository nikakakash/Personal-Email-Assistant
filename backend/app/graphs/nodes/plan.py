"""`plan_next_step` node and its conditional router `route_after_plan`.

Node: LangGraph calls `plan_next_step(state, runtime)` and merges the returned partial update into
the state. The node asks the model for one typed `NextStep` and records it; it does not act on it.

Runtime context: `runtime.context` is the `GraphDeps` passed to `graph.ainvoke(..., context=...)`
(LLM client, providers, settings, clock). It is not part of the checkpointed state, so live
clients never get serialized and are supplied fresh after a resume.

Conditional edge: after this node, LangGraph calls `route_after_plan(state)` to pick the next
node by name. Routers are pure functions of state (no I/O, no clock, no randomness): the route is
then fully explained by the checkpointed state, the same state always takes the same path (also
when a step is retried), and routers are trivial to unit-test. LangGraph does not checkpoint
anything a router does, so all work, including the LLM call, happens in the node; the router only
reads its result.
"""

from typing import Any, Literal
from uuid import UUID

from langgraph.runtime import Runtime

from app.graphs.deps import GraphDeps
from app.graphs.nodes.shared import llm_failure_update, preferences_context, user_content
from app.graphs.schemas import NextStep
from app.graphs.state import EmailState
from app.llm.client import BudgetExceeded, LLMCall, LLMMalformedOutput, LLMRefusal
from app.prompts import load_prompt

PlanRoute = Literal[
    "read_tools",
    "persist_question",
    "draft_reply",
    "draft_meeting",
    "finish",
    "budget_stop",
    "error",
]

_ROUTES: dict[str, PlanRoute] = {
    "read_thread": "read_tools",
    "get_freebusy": "read_tools",
    "ask_owner": "persist_question",
    "draft_reply": "draft_reply",
    "draft_meeting": "draft_meeting",
    "finish": "finish",
    "budget_stop": "budget_stop",
    "error": "error",
}


async def plan_next_step(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Choose the next action. Checks the step budget before spending on a model call."""
    deps = runtime.context
    steps = state.get("step_count", 0)
    if steps >= deps.settings.max_agent_steps:
        return {"next_action": "budget_stop", "status": "paused", "error_code": "step_budget"}

    prompt = load_prompt("plan")
    call = LLMCall(
        purpose="plan",
        system=prompt.system,
        cached_context=preferences_context(state),
        user_content=user_content(state),
        output_model=NextStep,
        workflow_id=UUID(state["workflow_id"]),
        script_key=state["provider_message_id"],
    )
    try:
        step = (await deps.llm.structured(call)).output
    except (LLMRefusal, LLMMalformedOutput, BudgetExceeded) as exc:
        return llm_failure_update(exc)

    return {
        "next_action": step.action,
        "step_count": steps + 1,
        "decision_summary": step.reason,
        "pending_freebusy": step.freebusy.model_dump(mode="json") if step.freebusy else None,
        "pending_question": step.question if step.action == "ask_owner" else None,
    }


def route_after_plan(state: EmailState) -> PlanRoute:
    """Map `next_action` to the next node; anything unexpected is an error."""
    return _ROUTES.get(state.get("next_action") or "", "error")
