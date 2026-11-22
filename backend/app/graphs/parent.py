"""The parent workflow graph: context, preferences, triage, notification review, response.

    START → load_context → load_preferences → triage
    triage ─(route_after_triage)─┬→ finalize                (ignore, or budget pause)
                                 ├→ persist_notification    (notify, or needs review)
                                 │    → notification_interrupt
                                 │    ─(route_after_notification)─┬→ finalize  (dismiss)
                                 │                                └→ response  (respond)
                                 └→ response                (respond)
    response → finalize → END

Concepts, in the order you meet them:

- **StateGraph and state.** The graph is built over `EmailState` (a TypedDict, see
  `state.py`). The initial input is just `{workflow_id, account_id, provider_message_id,
  provider_thread_id}`; the `WorkflowRun` row already exists. Every node returns a partial
  update, which LangGraph merges into the state, using a key's reducer if it has one.
- **Runtime context.** `context_schema=GraphDeps` declares the per-invocation dependencies.
  They are passed as `graph.ainvoke(input, config, context=deps, ...)`, reach every node (and
  the subgraph) as `runtime.context`, and are never checkpointed, so a resume in a new process
  simply passes fresh ones.
- **Conditional edges.** Routers such as `route_after_triage` are pure functions of state that
  return the next node's name.
- **Subgraph as a node.** `response` is the compiled response subgraph added with
  `add_node("response", compiled)`. Because it uses the same `EmailState`, it reads the parent's
  state directly and its final state flows back into the parent. It inherits the parent's
  checkpointer, so an interrupt inside it pauses the whole run and is resumed through the
  parent.
- **Interrupts and `Command(resume=...)`.** `interrupt(value)` pauses the run;
  `ainvoke` returns a result whose `result["__interrupt__"]` lists `Interrupt` objects (`.id`,
  `.value`). The app resumes with `ainvoke(Command(resume=decision), config, context=deps,
  durability="sync")`, and the paused `interrupt()` call returns `decision`. The interrupted
  node re-runs from its top on resume, which is why interrupt nodes hold nothing but the call.
- **Checkpointer and `thread_id`.** `build_parent_graph(checkpointer)` compiles with a
  checkpointer (`InMemorySaver` in tests, `AsyncPostgresSaver` in the worker). State is saved
  per `config={"configurable": {"thread_id": ...}}`; the thread id is
  `f"{account_id}:{provider_message_id}:{GRAPH_VERSION}"`. Changing the graph's shape means a
  new `GRAPH_VERSION`, so old checkpoints are never resumed by incompatible code.
- **`durability="sync"`.** Each step's checkpoint is written before the next step starts, so a
  crash never loses a completed step (and its side effects are never repeated by replay). Every
  invocation passes it.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graphs.deps import GraphDeps
from app.graphs.events import traced
from app.graphs.nodes.context import load_context, load_preferences
from app.graphs.nodes.finalize import finalize, unknown_decision
from app.graphs.nodes.notify import (
    notification_interrupt,
    persist_notification,
    route_after_notification,
)
from app.graphs.nodes.triage import route_after_triage, triage
from app.graphs.response import build_response_graph
from app.graphs.state import EmailState

GRAPH_VERSION = "wf1"


def graph_thread_id(account_id: str, provider_message_id: str) -> str:
    return f"{account_id}:{provider_message_id}:{GRAPH_VERSION}"


def build_parent_graph(
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph[EmailState, GraphDeps, EmailState, EmailState]:
    g = StateGraph(EmailState, context_schema=GraphDeps)
    for name, node in [
        ("load_context", load_context),
        ("load_preferences", load_preferences),
        ("triage", triage),
        ("persist_notification", persist_notification),
        ("unknown_decision", unknown_decision),
        ("finalize", finalize),
    ]:
        g.add_node(name, traced(name, node))
    g.add_node("notification_interrupt", notification_interrupt)  # only interrupt(); no I/O
    g.add_node("response", build_response_graph().compile())

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "load_preferences")
    g.add_edge("load_preferences", "triage")
    g.add_conditional_edges(
        "triage", route_after_triage, ["finalize", "persist_notification", "response"]
    )
    g.add_edge("persist_notification", "notification_interrupt")
    g.add_conditional_edges(
        "notification_interrupt",
        route_after_notification,
        ["response", "finalize", "unknown_decision"],
    )
    g.add_edge("unknown_decision", "finalize")
    g.add_edge("response", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
