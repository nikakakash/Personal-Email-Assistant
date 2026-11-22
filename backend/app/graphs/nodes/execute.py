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
it. Once the approval check passes (whatever the context checks then say), the DB-verified
`approval_id` and `proposal_revision` are stored in state; later nodes use those, never the
ids in the owner's resume value. The checks live in
`verify_approval`, which `execute_action` calls again right before executing.

`route_after_preflight` (a conditional-edge router, pure function of state) sends `ok` to
`execute_action`, `stale` to `mark_stale` (which retires the proposal and loops back to
planning with a `stale:<n>` finding) and `blocked` to `finish`. The approved payload is never
regenerated: a stale proposal needs a new draft and a new approval.

`execute_action` may be a retry (after a crash or `ActionInFlight`) long after preflight ran,
so it is *ledger first*:

1. The approval checks always run again (approval accepted, active revision; an `executed`
   proposal is accepted, since the retry may follow a successful send).
2. If an `actions` row already exists for the operation, in any status, the send has been
   claimed or made. The outbound and context checks are skipped: the kill switch means "no new
   writes", and returning a stored result or reconciling is a read; a thread that changed after
   the send must not turn a sent reply into a "stale" proposal. The executor then returns the
   stored result or reconciles; it never writes twice.
3. Only when no row exists does it repeat the outbound check and the context checks (TTL,
   thread version, slot), routing a failed one exactly as preflight would.

It then hands the approval to the deterministic `ActionExecutor`, which owns the idempotent
action ledger. Outcomes:

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
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from langgraph.runtime import Runtime
from sqlalchemy import exists, select, update

from app.actions.decisions import ACCEPTING
from app.actions.executor import ActionExecutor, ExecutionRefused, operation_id_for
from app.db.models import Action, AppFlags, Approval, Proposal, ProposalRevision, ReviewItem
from app.graphs.deps import GraphDeps
from app.graphs.nodes.reads import THREAD_MESSAGE_LIMIT, scheduling_preferences
from app.graphs.state import EmailState
from app.providers.interfaces import ProviderTimeout, ThreadNotFound, ThreadSnapshot
from app.scheduling.slots import Interval

PROPOSAL_TTL = timedelta(hours=72)
PREFLIGHT_TIMEOUT_SECONDS: float = 60

_BLOCKED_STATUS = {"outbound_disabled": "paused", "approval_invalid": "failed"}
# The only exceptions that mean "provider unavailable". asyncio.TimeoutError is TimeoutError.
_PROVIDER_ERRORS = (ProviderTimeout, ThreadNotFound, TimeoutError)


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


@dataclass(frozen=True)
class _Approved:
    """The DB-verified approval and the revision it authorizes."""

    approval_id: uuid.UUID
    proposal_id: uuid.UUID
    kind: str
    revision: int
    payload: dict[str, Any]
    created_at: datetime

    def ids(self) -> dict[str, Any]:
        """The verified ids as a state update, for every node downstream of the check."""
        return {"approval_id": str(self.approval_id), "proposal_revision": self.revision}

    def ok(self) -> dict[str, Any]:
        return {**_ok(), **self.ids()}


async def _outbound_blocked(deps: GraphDeps) -> bool:
    """The kill switch: the setting is off or `app_flags.outbound_paused` is set."""
    if not deps.settings.outbound_execution_enabled:
        return True
    async with deps.sessionmaker() as session:
        flags = await session.get(AppFlags, 1)
    return flags is not None and flags.outbound_paused


async def _check_approval(state: EmailState, deps: GraphDeps, approval_id: Any) -> _Approved | str:
    """The approval is accepted and valid for the active revision, or a blocking reason."""
    verified_id = _uuid(approval_id)
    async with deps.sessionmaker() as session:
        approval = await session.get(Approval, verified_id) if verified_id else None
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
        return _Approved(
            approval_id=approval.id,
            proposal_id=proposal.id,
            kind=proposal.kind,
            revision=revision.revision,
            payload=dict(revision.payload),
            created_at=revision.created_at,
        )


async def _fresh_thread(state: EmailState, deps: GraphDeps) -> ThreadSnapshot:
    thread_id = state["provider_thread_id"]  # read outside the try: a missing key is a bug
    try:
        return await asyncio.wait_for(
            deps.mail.get_thread(thread_id, limit=THREAD_MESSAGE_LIMIT),
            PREFLIGHT_TIMEOUT_SECONDS,
        )
    except _PROVIDER_ERRORS as exc:
        raise _Unavailable from exc


async def _slot_taken(state: EmailState, deps: GraphDeps, payload: dict[str, Any]) -> bool:
    buffer = timedelta(minutes=int(scheduling_preferences(state)["buffer_minutes"]))
    window = Interval(
        datetime.fromisoformat(payload["start"]) - buffer,
        datetime.fromisoformat(payload["end"]) + buffer,
    )
    time_zone = payload["time_zone"]
    try:
        freebusy = await asyncio.wait_for(
            deps.calendar.get_freebusy(window, time_zone), PREFLIGHT_TIMEOUT_SECONDS
        )
    except _PROVIDER_ERRORS as exc:
        raise _Unavailable from exc
    return any(window.start < b.end and window.end > b.start for b in freebusy.busy)


async def _check_context(
    state: EmailState, deps: GraphDeps, approved: _Approved
) -> dict[str, Any] | None:
    """TTL, thread version and (meetings) slot; a stale/blocked update, or None if current."""
    if approved.created_at < deps.now() - PROPOSAL_TTL:
        return _stale("expired")
    try:
        if (await _fresh_thread(state, deps)).version != state.get("source_version"):
            return _stale("thread_changed")
        if approved.kind == "create_event" and await _slot_taken(state, deps, approved.payload):
            return _stale("slot_taken")
    except _Unavailable:  # never execute on unverified context
        return _blocked("preflight_unavailable")
    return None


async def verify_approval(state: EmailState, deps: GraphDeps, approval_id: Any) -> dict[str, Any]:
    """All preflight checks (reads only); returns the state update.

    `preflight` (result and reason), plus the DB-verified `approval_id` and
    `proposal_revision` whenever the approval check passed (also when the context then turns
    out stale or unverifiable, so `mark_stale` and `execute_action` still have them), and
    `error_code` / `status` when blocked. Only provider
    failures (timeouts, unknown thread) become `preflight_unavailable`; any other exception is
    a bug and propagates.
    """
    if await _outbound_blocked(deps):
        return _blocked("outbound_disabled")
    approved = await _check_approval(state, deps, approval_id)
    if isinstance(approved, str):
        return _blocked(approved)
    problem = await _check_context(state, deps, approved)
    return {**approved.ids(), **problem} if problem is not None else approved.ok()


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


def _has_action(operation_id: str) -> Any:
    """SQL `EXISTS`: the operation has an action-ledger row (claimed, sent or finished)."""
    return exists(select(Action.id).where(Action.operation_id == operation_id))


async def _active_operation(deps: GraphDeps, proposal_id: uuid.UUID) -> str | None:
    """The ledger key of the proposal's active revision (the key `execute_action` checks)."""
    async with deps.sessionmaker() as session:
        proposal = await session.get(Proposal, proposal_id)
    return None if proposal is None else operation_id_for(proposal.id, proposal.active_revision)


async def _ledger_row_exists(deps: GraphDeps, operation_id: str) -> bool:
    async with deps.sessionmaker() as session:
        return bool(await session.scalar(select(_has_action(operation_id))))


async def mark_stale(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Retire the stale proposal and tell the planner why (a `stale:<n>` finding).

    A proposal whose active revision already has an action-ledger row (claimed or sent) is not
    stale: nothing is retired and no finding is added. The approval is verified again (the id
    from state, else the owner's decision) and, if valid, its verified ids go into state and
    the run goes back to `execute_action`, whose executor returns the stored result or
    reconciles; an invalid approval ends the run blocked, as in preflight. The ledger key is
    the same `operation_id_for(proposal, active revision)` that `execute_action` checks, so
    the two nodes cannot disagree and loop. Otherwise both updates are compare-and-set (also
    guarded by `NOT EXISTS` on that ledger key, against a race), so a re-run changes nothing. If
    the thread changed, the fresh snapshot is loaded first, with no transaction open; if that
    read fails the run ends `blocked:preflight_unavailable` (the proposal is still retired).
    """
    deps = runtime.context
    proposal_id = _uuid(state.get("proposal_id"))
    op = await _active_operation(deps, proposal_id) if proposal_id is not None else None
    if op is not None and await _ledger_row_exists(deps, op):
        approval_id = state.get("approval_id") or (state.get("review") or {}).get("approval_id")
        approved = await _check_approval(state, deps, approval_id)
        return _blocked(approved) if isinstance(approved, str) else approved.ok()
    reason = str((state.get("preflight") or {}).get("reason"))
    update_: dict[str, Any] = {}
    if reason == "thread_changed":
        try:
            snapshot = await _fresh_thread(state, deps)
            update_ = {
                "thread": snapshot.model_dump(mode="json"),
                "source_version": snapshot.version,
            }
        except _Unavailable:
            update_ = _blocked("preflight_unavailable")

    item_id = _uuid(state.get("review_item_id"))
    if proposal_id is not None and op is not None:
        async with deps.sessionmaker() as session, session.begin():
            await session.execute(
                update(Proposal)
                .where(
                    Proposal.id == proposal_id,
                    Proposal.status == "approved",
                    ~_has_action(op),
                )
                .values(status="stale")
            )
            if item_id is not None:
                await session.execute(
                    update(ReviewItem)
                    .where(
                        ReviewItem.id == item_id,
                        ReviewItem.status != "stale",
                        ~_has_action(op),
                    )
                    .values(status="stale")
                )

    n = 1 + sum(1 for f in state.get("read_findings") or [] if f.get("kind") == "stale")
    finding = {
        "key": f"stale:{n}",
        "kind": "stale",
        "reason": reason,
        "proposal_id": state.get("proposal_id"),
    }
    if "preflight" in update_:  # blocked: keep the blocked result for routing
        return {**update_, "read_findings": [finding]}
    return {**update_, "read_findings": [finding], "preflight": None}


def route_after_mark_stale(
    state: EmailState,
) -> Literal["plan_next_step", "execute_action", "finish"]:
    result = (state.get("preflight") or {}).get("result")
    if result == "ok":  # the proposal has a ledger row: let the executor reconcile
        return "execute_action"
    return "finish" if result == "blocked" else "plan_next_step"


async def execute_action(state: EmailState, runtime: Runtime[GraphDeps]) -> dict[str, Any]:
    """Execute the approved revision through the action ledger (ledger first on a retry).

    The ids come from state as `preflight` verified them against the DB, never from the
    owner's resume value, and are verified again here. See the module docstring for why the
    context checks run only when no ledger row exists yet.
    """
    deps = runtime.context
    approved = await _check_approval(state, deps, state.get("approval_id"))
    if isinstance(approved, str):
        return _blocked(approved)
    if not await _ledger_row_exists(
        deps, operation_id_for(approved.proposal_id, approved.revision)
    ):
        if await _outbound_blocked(deps):
            return _blocked("outbound_disabled")
        problem = await _check_context(state, deps, approved)
        if problem is not None:
            return {**approved.ids(), **problem}
    recheck = approved.ok()

    executor = ActionExecutor(deps.sessionmaker, deps.mail, deps.calendar, deps.settings)
    try:
        outcome = await executor.execute(approved.approval_id)
    except ExecutionRefused as refused:
        update_: dict[str, Any] = {"error_code": refused.reason, "next_action": "error"}
        if refused.reason in ("outbound_disabled", "outbound_paused"):
            update_["status"] = "paused"
        return update_
    if outcome.status == "executing":
        raise ActionInFlight(recheck["approval_id"])

    proposal_id, revision = state.get("proposal_id"), approved.revision
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
