"""After approval: `preflight`, `mark_stale`, `execute_action` and their routers.

`preflight` re-checks, right before execution, everything an approval depends on (spec §9). It
only reads (DB, mail, calendar) and returns `{"preflight": {"result", "reason"}}`, checking in
this order:

1. outbound disabled (setting off, or `app_flags.outbound_paused`) → `blocked:outbound_disabled`
2. approval missing, not accepting, or not for the active revision → `blocked:approval_invalid`
3. revision older than `PROPOSAL_TTL` → `stale:expired`
4. thread version differs from `source_version` → `stale:thread_changed`
5. meetings: a fresh free/busy read overlaps the slot (with buffer) → `stale:slot_taken`

A provider error or a timeout (`PREFLIGHT_TIMEOUT_SECONDS`) → `blocked:preflight_unavailable`;
any other exception is a bug and propagates. A blocked result also sets `error_code` (and
`status` where the run should read as paused or failed), so `finish` and `finalize` can report
it. An ok result stores the DB-verified `approval_id` and `proposal_revision` in state; later
nodes use those, never the ids in the owner's resume value. The checks live in
`verify_approval`, which `execute_action` calls again right before executing.

`route_after_preflight` (a conditional-edge router, pure function of state) sends `ok` to
`execute_action`, `stale` to `mark_stale` (which retires the proposal and loops back to
planning with a `stale:<n>` finding) and `blocked` to `finish`. The approved payload is never
regenerated: a stale proposal needs a new draft and a new approval.

`execute_action` first repeats the preflight checks (a retry may run long after preflight),
routing a failed re-check exactly as preflight would. It then hands the approval to the
deterministic `ActionExecutor`, which owns the idempotent action ledger; a re-run of this node
(e.g. after a crash) only reconciles. Outcomes:

- `succeeded` / `failed_definite` → `plan_next_step` (the planner sees the result);
- `outcome_unknown` → `finish` with `error_code="outcome_unknown"`: the provider may have
  accepted the write, so nothing is drafted again and the owner must look (spec §6, §9);
- an executor refusal → `finish` (paused if outbound is disabled or paused);
- `executing` (another caller's provider call is in progress) → `ActionInFlight` is raised.

Retrying after `ActionInFlight` (or a crash): the exception propagates out of `ainvoke`, but the
checkpoint from before `execute_action` was saved (`durability="sync"`). Calling
`graph.ainvoke(None, config, context=deps, durability="sync")` on the same `thread_id` resumes
from that checkpoint and re-runs only the failed node; input `None` means "continue", not
"start over". Completed nodes (preflight included) are not re-run.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Literal

from langgraph.runtime import Runtime
from sqlalchemy import update

from app.actions.decisions import ACCEPTING
from app.actions.executor import ActionExecutor, ExecutionRefused
from app.db.models import AppFlags, Approval, Proposal, ProposalRevision, ReviewItem
from app.graphs.deps import GraphDeps
from app.graphs.nodes.reads import THREAD_MESSAGE_LIMIT, scheduling_preferences
from app.graphs.state import EmailState
from app.providers.interfaces import ProviderTimeout
from app.scheduling.slots import Interval

PROPOSAL_TTL = timedelta(hours=72)
PREFLIGHT_TIMEOUT_SECONDS: float = 60

_BLOCKED_STATUS = {"outbound_disabled": "paused", "approval_invalid": "failed"}


_PROVIDER_ERRORS = (ProviderTimeout, KeyError, TimeoutError)  # asyncio.TimeoutError is TimeoutError


class _Unavailable(Exception):
    """A provider read needed by preflight failed or timed out."""


class ActionInFlight(Exception):  # noqa: N818 - name fixed by the contract
    """The operation is being executed by another caller right now; retry the resume later."""


def _ok() -> dict[str, Any]:
    return {"preflight": {"result": "ok", "reason": None}}


def _stale(reason: str) -> dict[str, Any]:
    return {"preflight": {"result": "stale", "reason": reason}}


def _blocked(reason: str) -> dict[str, Any]:
    update: dict[str, Any] = {
        "preflight": {"result": "blocked", "reason": reason},
        "error_code": reason,
    }
    if reason in _BLOCKED_STATUS:
        update["status"] = _BLOCKED_STATUS[reason]
    return update


def _uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


async def _approved_revision(
    state: EmailState, deps: GraphDeps, approval_id: uuid.UUID | None
) -> tuple[str, int, dict[str, Any], datetime] | str:
    """(kind, revision, payload, created_at) of the approved active revision, or a reason."""
    async with deps.sessionmaker() as session:
        flags = await session.get(AppFlags, 1)
        if flags is not None and flags.outbound_paused:
            return "outbound_disabled"
        approval = await session.get(Approval, approval_id) if approval_id else None
        if approval is None or approval.decision not in ACCEPTING:
            return "approval_invalid"
        revision = await session.get_one(ProposalRevision, approval.proposal_revision_id)
        proposal = await session.get_one(Proposal, revision.proposal_id)
        if (
            str(proposal.id) != state.get("proposal_id")
            or revision.revision != proposal.active_revision
            or proposal.status not in ("approved", "executed")
        ):
            return "approval_invalid"
        return proposal.kind, revision.revision, dict(revision.payload), revision.created_at


async def _slot_taken(state: EmailState, deps: GraphDeps, payload: dict[str, Any]) -> bool:
    buffer = timedelta(minutes=int(scheduling_preferences(state)["buffer_minutes"]))
    start = datetime.fromisoformat(payload["start"]) - buffer
    end = datetime.fromisoformat(payload["end"]) + buffer
    try:
        freebusy = await asyncio.wait_for(
            deps.calendar.get_freebusy(Interval(start, end), payload["time_zone"]),
            PREFLIGHT_TIMEOUT_SECONDS,
        )
    except _PROVIDER_ERRORS as exc:
        raise _Unavailable from exc
    return any(start < b.end and end > b.start for b in freebusy.busy)


async def _thread_version(state: EmailState, deps: GraphDeps) -> str:
    try:
        thread = await asyncio.wait_for(
            deps.mail.get_thread(state["provider_thread_id"], limit=THREAD_MESSAGE_LIMIT),
            PREFLIGHT_TIMEOUT_SECONDS,
        )
    except _PROVIDER_ERRORS as exc:
        raise _Unavailable from exc
    return thread.version


async def verify_approval(state: EmailState, deps: GraphDeps, approval_id: Any) -> dict[str, Any]:
    """The preflight checks, shared by `preflight` and `execute_action` (reads only).

    Returns the state update: `preflight` (result and reason), plus the DB-verified
    `approval_id` and `proposal_revision` when ok, or `error_code` / `status` when blocked.
    Only provider failures (timeouts, unknown thread) become `preflight_unavailable`; any other
    exception is a bug and propagates.
    """
    if not deps.settings.outbound_execution_enabled:
        return _blocked("outbound_disabled")
    verified_id = _uuid(approval_id)
    approved = await _approved_revision(state, deps, verified_id)
    if isinstance(approved, str):
        return _blocked(approved)
    kind, revision, payload, created_at = approved
    if created_at < deps.now() - PROPOSAL_TTL:
        return _stale("expired")
    try:
        if await _thread_version(state, deps) != state.get("source_version"):
            return _stale("thread_changed")
        if kind == "create_event" and await _slot_taken(state, deps, payload):
            return _stale("slot_taken")
    except _Unavailable:  # never execute on unverified context
        return _blocked("preflight_unavailable")
    return {**_ok(), "approval_id": str(verified_id), "proposal_revision": revision}


async def preflight(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Verify the approval named in the owner's decision, then the context (reads only)."""
    return await verify_approval(
        state, runtime.context, (state.get("review") or {}).get("approval_id")
    )


def route_after_preflight(state: EmailState) -> Literal["execute_action", "mark_stale", "finish"]:
    result = (state.get("preflight") or {}).get("result")
    if result == "ok":
        return "execute_action"
    if result == "stale":
        return "mark_stale"
    return "finish"


async def mark_stale(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Retire the stale proposal and tell the planner why (a `stale:<n>` finding).

    Both updates are compare-and-set, so a re-run changes nothing. If the thread changed, the
    fresh snapshot is loaded first (no transaction is open during the provider call).
    """
    deps = runtime.context
    reason = str((state.get("preflight") or {}).get("reason"))
    update_: dict[str, Any] = {}
    if reason == "thread_changed":
        snapshot = await asyncio.wait_for(
            deps.mail.get_thread(state["provider_thread_id"], limit=THREAD_MESSAGE_LIMIT),
            PREFLIGHT_TIMEOUT_SECONDS,
        )
        update_ = {"thread": snapshot.model_dump(mode="json"), "source_version": snapshot.version}

    async with deps.sessionmaker() as session, session.begin():
        proposal_id = _uuid(state.get("proposal_id"))
        if proposal_id is not None:
            await session.execute(
                update(Proposal)
                .where(Proposal.id == proposal_id, Proposal.status == "approved")
                .values(status="stale")
            )
        item_id = _uuid(state.get("review_item_id"))
        if item_id is not None:
            await session.execute(
                update(ReviewItem)
                .where(ReviewItem.id == item_id, ReviewItem.status != "stale")
                .values(status="stale")
            )

    n = 1 + sum(1 for f in state.get("read_findings") or [] if f.get("kind") == "stale")
    finding = {
        "key": f"stale:{n}",
        "kind": "stale",
        "reason": reason,
        "proposal_id": state.get("proposal_id"),
    }
    return {**update_, "read_findings": [finding], "preflight": None}


async def execute_action(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Re-verify, then execute the approved revision through the action ledger.

    The ids come from state as `preflight` verified them against the DB, never from the
    owner's resume value. The same checks run again first, because this node may be a retry
    (after a crash or `ActionInFlight`) long after preflight ran.
    """
    deps = runtime.context
    approval_id = state.get("approval_id")
    recheck = await verify_approval(state, deps, approval_id)
    if recheck["preflight"]["result"] != "ok":
        return recheck
    executor = ActionExecutor(deps.sessionmaker, deps.mail, deps.calendar, deps.settings)
    try:
        outcome = await executor.execute(uuid.UUID(recheck["approval_id"]))
    except ExecutionRefused as refused:
        update_: dict[str, Any] = {"error_code": refused.reason, "next_action": "error"}
        if refused.reason in ("outbound_disabled", "outbound_paused"):
            update_["status"] = "paused"
        return update_
    if outcome.status == "executing":
        raise ActionInFlight(str(approval_id))

    proposal_id, revision = state.get("proposal_id"), recheck["proposal_revision"]
    result = {
        "proposal_id": proposal_id,
        "revision": revision,
        "status": outcome.status,
        "provider_ref": outcome.provider_ref,
        "error": outcome.error,
    }
    # The planner reads findings, so the outcome is also recorded there (no email content).
    finding = {
        "key": f"action:{proposal_id}:{revision}",
        "kind": "action_result",
        "status": outcome.status,
        "error": outcome.error,
    }
    update_ = {**recheck, "action_results": [result], "read_findings": [finding]}
    if outcome.status == "outcome_unknown":
        # The provider may have accepted the write: never let the planner draft again.
        update_ |= {"error_code": "outcome_unknown", "next_action": "error"}
    return update_


def route_after_execute(state: EmailState) -> Literal["plan_next_step", "mark_stale", "finish"]:
    """Like preflight for a failed re-check; `finish` for refusals and unknown outcomes."""
    result = (state.get("preflight") or {}).get("result")
    if result == "stale":
        return "mark_stale"
    if result == "blocked" or state.get("next_action") == "error":
        return "finish"
    return "plan_next_step"
