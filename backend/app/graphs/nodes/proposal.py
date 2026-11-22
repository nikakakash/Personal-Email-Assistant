"""Proposal review: `persist_proposal`, `approval_interrupt`, `route_after_review`,
`record_feedback`.

`persist_proposal` saves the validated draft as an immutable proposal revision with a pending
review item (`create_proposal`). It is idempotent: the key `f"{workflow_id}:{step_count}"`
returns the existing proposal if the node runs again. `step_count` only ever grows (the planner
increments it), so each planning round gets a fresh key.

If the run is already final (the owner cancelled while the graph was drafting),
`create_proposal` raises `RunIsFinal`; the node then writes nothing and returns
`{"run_final": True}`, and `route_after_persist_proposal` sends the graph to `finish`.

`approval_interrupt` is only the `interrupt()` call. On resume LangGraph re-runs the node from
its top, so any write placed before `interrupt()` would run twice; the approval itself is
recorded by the API (`decide_proposal`) before the resume job exists. The resume values are the
decision payloads built there:

- `{"decision": "approve" | "edit_approve", "approval_id", "revision"}` → `preflight`
- `{"decision": "feedback", "feedback"}` → `record_feedback` → `plan_next_step`
- `{"decision": "reject" | "superseded" | "cancelled"}` → `finish`
- anything else → `unknown_decision` → `finish`

`route_after_review` is a conditional-edge router: a pure function of state returning a node
name.
"""

import uuid
from typing import Any, Literal, cast

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.proposals import ProposalKind, RunIsFinal, create_proposal
from app.db.models import Proposal, ReviewItem
from app.graphs.deps import GraphDeps
from app.graphs.state import EmailState

ReviewRoute = Literal["preflight", "record_feedback", "finish", "unknown_decision"]


async def persist_proposal(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Save the validated draft as proposal revision 1 and open its review item."""
    workflow_id = uuid.UUID(state["workflow_id"])
    draft = state.get("draft")
    if draft is None:
        raise ValueError("persist_proposal needs a validated draft")
    try:
        async with runtime.context.sessionmaker() as session, session.begin():
            proposal, revision = await create_proposal(
                session,
                workflow_id,
                cast(ProposalKind, state.get("draft_kind")),
                draft,
                state.get("source_version"),
                str(state.get("preference_version", 0)),
                idempotency_key=f"{state['workflow_id']}:{state.get('step_count', 0)}",
            )
            item_id = await _review_item_id(session, proposal.id)
    except RunIsFinal:
        return {"run_final": True}  # cancelled while drafting: nothing to review
    return {
        "proposal_id": str(proposal.id),
        "proposal_revision": revision.revision,
        "review_item_id": str(item_id),
        "approval_id": None,
        "preflight": None,
    }


def route_after_persist_proposal(state: EmailState) -> Literal["approval_interrupt", "finish"]:
    return "finish" if state.get("run_final") else "approval_interrupt"


async def _review_item_id(session: AsyncSession, proposal_id: uuid.UUID) -> uuid.UUID:
    return (
        await session.execute(
            select(ReviewItem.id)
            .where(ReviewItem.proposal_id == proposal_id, ReviewItem.kind == "proposal")
            .order_by(ReviewItem.created_at.desc())
            .limit(1)
        )
    ).scalar_one()


def approval_interrupt(state: EmailState) -> dict[str, Any]:
    """Pause until the owner decides. Nothing else belongs here: it re-runs on resume."""
    value = interrupt({"type": "approval", "review_item_id": state["review_item_id"]})
    return {"review": value}


def route_after_review(state: EmailState) -> ReviewRoute:
    decision = (state.get("review") or {}).get("decision")
    if decision in ("approve", "edit_approve"):
        return "preflight"
    if decision == "feedback":
        return "record_feedback"
    if decision in ("reject", "superseded", "cancelled"):
        return "finish"
    return "unknown_decision"


async def record_feedback(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Add the owner's feedback to `owner_inputs`; mark the reviewed proposal superseded.

    `decide_proposal` normally supersedes the proposal already; this compare-and-set only
    moves a proposal still `awaiting_approval`, so it is a no-op then and safe to re-run.
    """
    feedback = str((state.get("review") or {}).get("feedback") or "")
    proposal_id = state.get("proposal_id")
    if proposal_id:
        async with runtime.context.sessionmaker() as session, session.begin():
            await session.execute(
                update(Proposal)
                .where(
                    Proposal.id == uuid.UUID(proposal_id),
                    Proposal.status == "awaiting_approval",
                )
                .values(status="superseded")
            )
    return {"owner_inputs": [{"kind": "feedback", "text": feedback}]}
