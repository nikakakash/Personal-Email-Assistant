# LangGraph concepts, as used here

Each section explains one concept in a few lines and points to where this codebase uses it.
Paths are relative to `backend/`. The module docstrings in `app/graphs/` go into more detail.

New to LangGraph? Start with the full teaching guide, [`learn-langgraph.md`](learn-langgraph.md):
diagrams, runnable toy examples, a step-by-step walkthrough of one email, and exercises.

## StateGraph and state

A `StateGraph` is a set of nodes (async functions) connected by edges. All nodes share one
state object. Each node receives the current state and returns a partial update, a dict of
the keys it changes. The graph is declared once, then compiled into a runnable.

- State schema: `EmailState`, a `TypedDict` in [`app/graphs/state.py`](../backend/app/graphs/state.py).
- Parent graph: `build_parent_graph` in [`app/graphs/parent.py`](../backend/app/graphs/parent.py)
  (`load_context` → `load_preferences` → `triage` → `record_triage` → …).
- Response graph: `build_response_graph` in [`app/graphs/response.py`](../backend/app/graphs/response.py).
- Example node: `triage` in [`app/graphs/nodes/triage.py`](../backend/app/graphs/nodes/triage.py)
  returns `{"classification": ..., "urgency": ..., ...}` and nothing else.

## Reducers

By default an update **replaces** a key. A key declared as `Annotated[T, reducer]` is
**merged** instead: LangGraph calls `reducer(old, new)`.

- `owner_inputs` and `action_results` use `operator.add`, which appends
  ([`state.py`](../backend/app/graphs/state.py)).
- `read_findings` uses `merge_by_key`. Re-reading free/busy for the same days replaces that
  finding instead of duplicating it, which keeps re-runs after a crash idempotent.
- A subtle case is documented at `g.add_node("response", ...)` in `parent.py`: a subgraph's
  output passes through the parent's reducers again.

## Conditional edges

`add_edge(a, b)` always goes from `a` to `b`. `add_conditional_edges(a, router, targets)` calls
a **router**, a pure function of state that returns the next node's name. Because routers are
pure, the path a run took can be explained entirely from its checkpointed state.

- `route_after_triage` (respond → `response`, notify → `persist_notification`, else `finalize`)
  in [`nodes/triage.py`](../backend/app/graphs/nodes/triage.py).
- `route_after_plan` maps the model's chosen `NextStep.action` to a node; anything unknown goes
  to `error` ([`nodes/plan.py`](../backend/app/graphs/nodes/plan.py)).
- `route_after_validate` ([`nodes/validate.py`](../backend/app/graphs/nodes/validate.py)) and
  `route_after_preflight` / `route_after_execute` ([`nodes/execute.py`](../backend/app/graphs/nodes/execute.py)).

## Subgraphs

A compiled graph can be added as a node of another graph. Here the parent decides *whether*
to respond. The `response` subgraph does the work: plan, read, ask, draft, validate, approve,
execute. It loops back to `plan_next_step` until the planner says `finish`.

- `g.add_node("response", build_response_graph().compile())` in [`parent.py`](../backend/app/graphs/parent.py).
- The subgraph shares `EmailState` with the parent, so it reads and writes the same keys.
- Interrupts inside the subgraph pause the whole run. The worker finds them with
  `aget_state(config, subgraphs=True)` ([`app/jobs/worker.py`](../backend/app/jobs/worker.py)).

## `interrupt()` and `Command(resume=...)`

`interrupt(value)` stops the run at that point. The state is checkpointed and `value` is
reported to the caller. Later, `graph.ainvoke(Command(resume=answer), config)` on the same
thread ID continues the run: the node **runs again from its first line**, and this time
`interrupt()` returns `answer`.

- `approval_interrupt` ([`nodes/proposal.py`](../backend/app/graphs/nodes/proposal.py)) waits
  for the owner's decision on a proposal.
- `question_interrupt` ([`nodes/ask.py`](../backend/app/graphs/nodes/ask.py)) waits for an answer.
- `notification_interrupt` ([`nodes/notify.py`](../backend/app/graphs/nodes/notify.py)) waits
  for dismiss or respond.
- The worker resumes with `Command(resume={interrupt_id: payload})`. Keying by interrupt ID
  means a resume can only answer the interrupt the owner actually saw
  ([`app/jobs/worker.py`](../backend/app/jobs/worker.py)).

### Why interrupt nodes contain no side effects

Resuming re-executes the interrupted node from the top, so anything before `interrupt()` would
run twice: two review items, two events, two emails. So each interrupt node contains only
the `interrupt()` call and pure reads of state. The writes it depends on live in the node
before it and are idempotent by app-table keys:

- `persist_proposal` (key `"{workflow_id}:{step_count}"`) → `approval_interrupt`
- `persist_question` (a deterministic `uuid5` review-item ID) → `question_interrupt`
- `persist_notification` → `notification_interrupt`

The approval itself is recorded by the API (`app/actions/decisions.py`) before the resume job
exists, never inside the graph.

## Runtime context

Some things a node needs should not be checkpointed: the LLM client, providers, the DB
sessionmaker and settings. They also must never end up in state, since secrets must not be
persisted. These go in the **runtime context**: `StateGraph(EmailState, context_schema=GraphDeps)`
declares its type, `ainvoke(..., context=deps)` passes it in, and a node reads it as
`runtime.context`.

- `GraphDeps` in [`app/graphs/deps.py`](../backend/app/graphs/deps.py).
- Built once per worker by `make_deps` in [`app/jobs/worker.py`](../backend/app/jobs/worker.py).
- Tests pass a scripted LLM and mock providers through the same parameter.

## Checkpointer

Compiling with a checkpointer makes every step durable, keyed by `thread_id`, so a run can
pause for days and resume in another process.

- Tests: `InMemorySaver`. Worker: `AsyncPostgresSaver`
  ([`app/jobs/worker.py`](../backend/app/jobs/worker.py)). Its tables are created by
  `python -m app migrate` (`prepare_storage`).
- Thread ID format: `{account_id}:{provider_message_id}:wf1`, one thread per triggering email.
- The UI never reads checkpoints. Nodes project what the UI needs into app tables (review
  items, proposals, events), and the API reads those ([`app/api/views.py`](../backend/app/api/views.py)).

## Durability modes

`durability` controls when checkpoints are written: `"exit"` (only at the end), `"async"`
(in the background) or `"sync"` (before the next step starts). This app always uses
`"sync"`. If the worker crashes, the next attempt continues from the last completed step
instead of redoing work, and a step's effects are never ahead of its checkpoint by more
than that one step. The idempotent side-effect nodes cover that last step.

- Every `ainvoke` in [`app/jobs/worker.py`](../backend/app/jobs/worker.py) passes `durability="sync"`.

## Putting it together: one approval

1. `persist_proposal` writes revision 1 and a pending review item (idempotent).
2. `approval_interrupt` calls `interrupt(...)`. The run pauses and the checkpoint is saved in Postgres.
3. The owner approves. The API stores the approval and enqueues a `resume` job.
4. The worker calls `ainvoke(Command(resume={id: decision}), config, context=deps, durability="sync")`.
5. `approval_interrupt` re-runs, `interrupt()` returns the decision, and `route_after_review`
   sends the run to `preflight`, then `execute_action`, which calls the executor and its
   ledger. The model is not called again for that payload.
