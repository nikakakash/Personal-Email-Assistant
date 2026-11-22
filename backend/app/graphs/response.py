"""The response subgraph: plan, read, ask, draft, validate, review, preflight, execute.

LangGraph in brief, as used here:

- **StateGraph.** `StateGraph(EmailState, context_schema=GraphDeps)` declares a graph whose
  state is the `EmailState` TypedDict and whose runtime context (LLM client, providers,
  sessionmaker, settings, clock) is a `GraphDeps`. The context is passed per invocation with
  `ainvoke(..., context=deps)` and is never checkpointed; nodes read it as `runtime.context`.
- **Nodes return partial updates.** Each node returns only the keys it changed. LangGraph
  merges them into the state: a plain key is overwritten; a key with a reducer is combined
  (`owner_inputs` and `action_results` append with `operator.add`, `read_findings` merges by
  `"key"`). See `state.py`.
- **Edges and routers.** `add_edge(a, b)` always goes from a to b. `add_conditional_edges(a,
  router, path_map)` calls `router(state)` after `a` and follows the node it names. Routers
  are pure functions of state, so a path is fully explained by the checkpointed state.
- **Interrupts.** `approval_interrupt` and `question_interrupt` call `interrupt(...)`, which
  pauses the whole run (parent included): the caller gets `result["__interrupt__"]` and the
  state is saved by the checkpointer. `ainvoke(Command(resume=value), config)` on the same
  `thread_id` continues, and `interrupt()` returns `value`. The interrupted node re-runs from
  its top, so interrupt nodes contain nothing but that call; the writes they depend on happen
  in the `persist_*` nodes before them, and the effects of the answer in the nodes after them.

Flow:

    START → plan_next_step ─(route_after_plan)─┬→ read_tools → plan_next_step
                                               ├→ persist_question → question_interrupt
                                               │      ─(answer)→ record_answer → plan_next_step
                                               ├→ draft_reply | draft_meeting → validate_proposal
                                               │      ─(route_after_validate)→ persist_proposal
                                               │      | plan_next_step | finish
                                               └→ finish (finish, budget_stop, error)
    persist_proposal → approval_interrupt ─(route_after_review)─┬→ preflight
                                                                ├→ record_feedback → plan
                                                                └→ finish
    preflight ─┬(ok)→ execute_action ─→ plan_next_step  (succeeded, failed_definite)
               │                     ─→ finish  (outcome_unknown, refusal, blocked re-check)
               │                     ─→ mark_stale  (stale re-check)
               ├(stale)→ mark_stale → plan_next_step
               └(blocked)→ finish
    finish → END

Loops are bounded by `step_count`, which `plan_next_step` increments and checks against
`settings.max_agent_steps` before every model call. It is never reset.

Every node except the interrupt nodes is wrapped with `traced`, which emits `node_started`.
`read_tools` also emits one `tool_result` per finding it produced (tool name and finding key,
no content).

`build_response_graph()` returns the *uncompiled* builder; the parent compiles it and adds it
as a single node (see `parent.py`). The subgraph's updates reach the parent's state only when
the subgraph finishes: while paused at an interrupt in here, the parent's `ainvoke` result
and `aget_state(config)` show the state from before the subgraph started; use
`aget_state(config, subgraphs=True)` to see the subgraph's own state.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.graphs.deps import GraphDeps
from app.graphs.events import emit_now, traced
from app.graphs.nodes.ask import (
    persist_question,
    question_interrupt,
    record_answer,
    route_after_question,
)
from app.graphs.nodes.draft import draft_meeting, draft_reply
from app.graphs.nodes.execute import (
    execute_action,
    mark_stale,
    preflight,
    route_after_execute,
    route_after_preflight,
)
from app.graphs.nodes.finalize import finish, unknown_decision
from app.graphs.nodes.plan import plan_next_step, route_after_plan
from app.graphs.nodes.proposal import (
    approval_interrupt,
    persist_proposal,
    record_feedback,
    route_after_review,
)
from app.graphs.nodes.reads import read_tools
from app.graphs.nodes.validate import route_after_validate, validate_proposal
from app.graphs.state import EmailState


async def read_tools_with_events(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Run `read_tools`, then emit `tool_result` for each finding (key only, no content)."""
    tool = state.get("next_action")
    update = await read_tools(state, runtime)
    await emit_now(
        runtime.context,
        state["workflow_id"],
        [("tool_result", {"tool": tool, "key": f["key"]}) for f in update.get("read_findings", [])],
    )
    return update


def build_response_graph() -> StateGraph[EmailState, GraphDeps, EmailState, EmailState]:
    g = StateGraph(EmailState, context_schema=GraphDeps)

    for name, node in [
        ("plan_next_step", plan_next_step),
        ("read_tools", read_tools_with_events),
        ("persist_question", persist_question),
        ("record_answer", record_answer),
        ("draft_reply", draft_reply),
        ("draft_meeting", draft_meeting),
        ("validate_proposal", validate_proposal),
        ("persist_proposal", persist_proposal),
        ("record_feedback", record_feedback),
        ("preflight", preflight),
        ("mark_stale", mark_stale),
        ("execute_action", execute_action),
        ("unknown_decision", unknown_decision),
        ("finish", finish),
    ]:
        g.add_node(name, traced(name, node))
    # Interrupt nodes: only the interrupt() call, no events or other I/O.
    g.add_node("question_interrupt", question_interrupt)
    g.add_node("approval_interrupt", approval_interrupt)

    g.add_edge(START, "plan_next_step")
    g.add_conditional_edges(
        "plan_next_step",
        route_after_plan,
        {
            "read_tools": "read_tools",
            "persist_question": "persist_question",
            "draft_reply": "draft_reply",
            "draft_meeting": "draft_meeting",
            "finish": "finish",
            "budget_stop": "finish",
            "error": "finish",
        },
    )
    g.add_edge("read_tools", "plan_next_step")

    g.add_edge("persist_question", "question_interrupt")
    g.add_conditional_edges(
        "question_interrupt",
        route_after_question,
        ["record_answer", "finish", "unknown_decision"],
    )
    g.add_edge("record_answer", "plan_next_step")

    g.add_edge("draft_reply", "validate_proposal")
    g.add_edge("draft_meeting", "validate_proposal")
    g.add_conditional_edges(
        "validate_proposal",
        route_after_validate,
        {
            "persist_proposal": "persist_proposal",
            "plan_next_step": "plan_next_step",
            "error": "finish",
            "budget_stop": "finish",
        },
    )

    g.add_edge("persist_proposal", "approval_interrupt")
    g.add_conditional_edges(
        "approval_interrupt",
        route_after_review,
        ["preflight", "record_feedback", "finish", "unknown_decision"],
    )
    g.add_edge("record_feedback", "plan_next_step")

    g.add_conditional_edges(
        "preflight", route_after_preflight, ["execute_action", "mark_stale", "finish"]
    )
    g.add_edge("mark_stale", "plan_next_step")
    g.add_conditional_edges(
        "execute_action", route_after_execute, ["plan_next_step", "mark_stale", "finish"]
    )

    g.add_edge("unknown_decision", "finish")
    g.add_edge("finish", END)
    return g
