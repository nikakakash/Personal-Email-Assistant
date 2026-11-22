"""Graph state for the email workflow.

Graph state: LangGraph runs a graph as a series of nodes. Each node receives the current state
(a TypedDict here) and returns a *partial update*, a dict containing only the keys it changed.
LangGraph merges that update into the state and saves a checkpoint after each step, which is how
a workflow can pause for human approval and resume days later. Because state is checkpointed it
must hold JSON-serializable values only (str, int, dict, list, None). Pydantic models are
stored as `model_dump(mode="json")` dicts. Live objects (clients, DB sessions) never go in state;
see `deps.py`.

Reducers: by default a returned key *overwrites* the old value. A key annotated with
`Annotated[T, reducer]` is merged instead: LangGraph calls `reducer(old, new)`. This matters when
several nodes write the same key in one step (parallel branches), and for lists that should
grow. `operator.add` appends; `merge_by_key` below merges by an identifying key.

`total=False` means every key is optional, since state fills in as the graph progresses.
"""

import operator
from typing import Annotated, Any, TypedDict


def merge_by_key(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Reducer: merge findings by their ``"key"``, later entry wins, first-seen order kept.

    Unlike `operator.add`, re-running a node or merging parallel branches that produce the same
    finding does not create duplicates; the newest version replaces the old one.
    """
    merged: dict[str, dict[str, Any]] = {}
    for entry in [*(left or []), *(right or [])]:
        if "key" not in entry:
            raise ValueError("finding is missing required 'key'")
        merged[entry["key"]] = entry  # re-assigning keeps the original insertion position
    return list(merged.values())


class EmailState(TypedDict, total=False):
    # Identity of the run.
    workflow_id: str
    account_id: str
    provider_message_id: str
    provider_thread_id: str
    source_version: str

    # Context loaded from providers/stores. `thread` is ThreadSnapshot.model_dump(mode="json").
    thread: dict[str, Any]
    preferences: dict[str, Any]
    preference_version: int

    # Triage result.
    classification: str
    urgency: str
    decision_summary: str
    evidence_refs: list[str]

    # Accumulated by reducers (see module docstring).
    read_findings: Annotated[list[dict[str, Any]], merge_by_key]
    owner_inputs: Annotated[list[dict[str, Any]], operator.add]

    # Planning loop.
    next_action: str
    step_count: int

    # Drafting and validation.
    draft: dict[str, Any] | None
    draft_kind: str | None
    validation: dict[str, Any] | None
    validation_attempts: int
    preflight: dict[str, Any] | None
    pending_freebusy: dict[str, Any] | None
    pending_question: str | None

    # Proposal, approval and review records (ids refer to database rows).
    proposal_id: str | None
    proposal_revision: int | None
    approval_id: str | None
    review_item_id: str | None
    review: dict[str, Any] | None

    # Outcome.
    action_results: Annotated[list[dict[str, Any]], operator.add]
    status: str
    error_code: str | None
