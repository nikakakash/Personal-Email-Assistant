"""The deterministic executor: runs one approved operation through an idempotent action ledger.

Each approved revision maps to one stable operation id. The ledger row in `actions` is claimed
(`status="executing"`, with the request fingerprint) and committed *before* the provider call,
the call runs with no transaction open, and the result is recorded in a new transaction. So:

- a repeat `execute` of a finished operation returns the stored result;
- a repeat of an `executing` or `outcome_unknown` operation only reconciles (looks the write up
  by operation id or event id) and never writes again; an ambiguous send stays
  `outcome_unknown` for the owner to inspect (spec §9, §15);
- the outbound kill switch (`OUTBOUND_EXECUTION_ENABLED=false` or `app_flags.outbound_paused`)
  refuses only a new claim: returning a stored result and reconciling are reads, always allowed;
- an `executing` row younger than `IN_FLIGHT_GRACE` is assumed to be another caller's call in
  progress and is left alone (status `executing` is returned, nothing is recorded).

Errors stored in `actions.error`, audit events and workflow events are fixed codes
(`ErrorCode`), never provider text.

The request sent is built from the approved revision's stored payload; nothing is regenerated.
"""

import base64
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.hashing import payload_hash
from app.config import Settings
from app.db.models import (
    Action,
    AppFlags,
    Approval,
    AuditEvent,
    Proposal,
    ProposalRevision,
    WorkflowEvent,
)
from app.mail.addresses import normalize_address
from app.providers.interfaces import (
    CalendarProvider,
    EventRequest,
    MailProvider,
    OutgoingReply,
    ProviderResult,
    ProviderTimeout,
)

ActionStatus = Literal["executing", "succeeded", "failed_definite", "outcome_unknown"]
ErrorCode = Literal[
    "provider_failed",
    "provider_timeout",
    "event_id_conflict",
    "not_found_on_reconcile",
    "unexpected_error",
]
_TERMINAL = ("succeeded", "failed_definite")
# An `executing` row this fresh is probably another caller's provider call still in progress.
IN_FLIGHT_GRACE = timedelta(minutes=2)
_REPLY_FIELDS = ("provider_thread_id", "in_reply_to", "references", "to", "cc", "subject")
_EVENT_FIELDS = (
    "calendar_id",
    "title",
    "start",
    "end",
    "time_zone",
    "attendees",
    "location",
    "send_updates",
)


class ExecutionRefused(Exception):  # noqa: N818 - name fixed by the contract
    """A precondition failed; nothing was claimed or sent."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ActionOutcome:
    status: ActionStatus
    provider_ref: str | None
    error: ErrorCode | None


def operation_id_for(proposal_id: uuid.UUID, revision: int) -> str:
    return f"op-{proposal_id}-{revision}"


def calendar_event_id(operation_id: str) -> str:
    """Stable Google-valid event id: base32hex (0-9a-v) of sha256's first 20 bytes, 32 chars."""
    digest = hashlib.sha256(operation_id.encode("utf-8")).digest()[:20]
    return base64.b32hexencode(digest).decode("ascii").lower()


def build_reply(payload: dict[str, Any]) -> OutgoingReply:
    return OutgoingReply.model_validate({k: payload[k] for k in (*_REPLY_FIELDS, "body_text")})


def build_event(payload: dict[str, Any]) -> EventRequest:
    return EventRequest.model_validate({k: payload[k] for k in _EVENT_FIELDS})


def _addresses(values: Any) -> list[str]:
    return sorted(normalize_address(v) for v in values)


def event_matches(intended: EventRequest, found: dict[str, Any]) -> bool:
    """Same title, start, end and attendees (the fields that make it the approved event)."""
    try:
        return (
            found.get("title") == intended.title
            and datetime.fromisoformat(str(found["start"])) == intended.start
            and datetime.fromisoformat(str(found["end"])) == intended.end
            and _addresses(found.get("attendees") or []) == _addresses(intended.attendees)
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class _Claim:
    action_id: uuid.UUID
    workflow_id: uuid.UUID
    proposal_id: uuid.UUID
    kind: str
    operation_id: str
    status: str
    provider_ref: str | None
    error: ErrorCode | None
    new: bool
    in_flight: bool
    request: OutgoingReply | EventRequest


class ActionExecutor:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        mail: MailProvider,
        calendar: CalendarProvider,
        settings: Settings,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._mail = mail
        self._calendar = calendar
        self._settings = settings

    async def execute(self, approval_id: uuid.UUID) -> ActionOutcome:
        """Execute (or reconcile) the operation an accepted approval authorizes.

        Raises `ExecutionRefused` when a precondition fails, before any ledger row exists.
        """
        claim = await self._claim(approval_id)
        if claim.status in _TERMINAL or claim.in_flight:
            return ActionOutcome(cast(ActionStatus, claim.status), claim.provider_ref, claim.error)
        if claim.new:
            outcome = await self._call_provider(claim)
            attempted = True
        else:
            outcome = await self._reconcile(claim)
            attempted = False
        return await self._record(claim, outcome, attempted)

    async def _claim(self, approval_id: uuid.UUID) -> _Claim:
        async with self._sessionmaker() as session, session.begin():
            approval = await session.get(Approval, approval_id)
            if approval is None:
                raise ExecutionRefused("approval_not_found")
            if approval.decision not in ("approve", "edit_approve"):
                raise ExecutionRefused("not_approved")
            revision = await session.get_one(ProposalRevision, approval.proposal_revision_id)
            proposal = await session.get_one(Proposal, revision.proposal_id)
            if revision.revision != proposal.active_revision:
                raise ExecutionRefused("revision_not_active")
            if proposal.status not in ("approved", "executed"):
                raise ExecutionRefused("proposal_not_approved")
            fingerprint = payload_hash(revision.payload)
            if fingerprint != revision.payload_hash:
                raise ExecutionRefused("payload_hash_mismatch")
            try:
                request: OutgoingReply | EventRequest = (
                    build_reply(revision.payload)
                    if proposal.kind == "send_reply"
                    else build_event(revision.payload)
                )
            except (KeyError, ValidationError) as exc:
                raise ExecutionRefused("invalid_payload") from exc

            op = operation_id_for(proposal.id, revision.revision)
            # The kill switch blocks only NEW writes. An existing row is returned or
            # reconciled (reads) whatever the flags say, so a recorded send is never lost.
            existing = await session.scalar(select(Action.id).where(Action.operation_id == op))
            if existing is None:
                if not self._settings.outbound_execution_enabled:
                    raise ExecutionRefused("outbound_disabled")
                flags = await session.get(AppFlags, 1)
                if flags is not None and flags.outbound_paused:
                    raise ExecutionRefused("outbound_paused")
            new_id = (
                await session.execute(
                    insert(Action)
                    .values(
                        operation_id=op,
                        approval_id=approval.id,
                        kind=proposal.kind,
                        request_fingerprint=fingerprint,
                        status="executing",
                        attempts=1,  # the claim is committed only to make this one call
                    )
                    .on_conflict_do_nothing(index_elements=[Action.operation_id])
                    .returning(Action.id)
                )
            ).scalar_one_or_none()
            action, now = (
                await session.execute(select(Action, func.now()).where(Action.operation_id == op))
            ).one()
            new = new_id is not None
            return _Claim(
                action_id=action.id,
                workflow_id=proposal.workflow_id,
                proposal_id=proposal.id,
                kind=proposal.kind,
                operation_id=op,
                status=action.status,
                provider_ref=action.provider_ref,
                error=cast(ErrorCode | None, action.error),
                new=new,
                in_flight=(
                    not new
                    and action.status == "executing"
                    and now - action.updated_at < IN_FLIGHT_GRACE
                ),
                request=request,
            )

    async def _call_provider(self, claim: _Claim) -> ActionOutcome:
        """The one real write attempt. No transaction is open here."""
        if isinstance(claim.request, OutgoingReply):
            try:
                result = await self._mail.send_reply(claim.request, claim.operation_id)
            except ProviderTimeout:
                return ActionOutcome("outcome_unknown", None, "provider_timeout")
            except Exception:  # the provider may have accepted it; never assume not
                return ActionOutcome("outcome_unknown", None, "unexpected_error")
            return _from_result(result)

        event_id = calendar_event_id(claim.operation_id)
        try:
            result = await self._calendar.create_event(claim.request, event_id)
        except ProviderTimeout:
            return await self._lookup_event(
                claim.request, event_id, unknown_error="provider_timeout"
            )
        except Exception:
            return ActionOutcome("outcome_unknown", None, "unexpected_error")
        return _from_result(result)

    async def _reconcile(self, claim: _Claim) -> ActionOutcome:
        """Look the earlier write up; never repeat it."""
        if isinstance(claim.request, OutgoingReply):
            found = await self._mail.find_sent(claim.operation_id)
            if found is not None and found.status == "confirmed":
                return ActionOutcome("succeeded", found.provider_ref, None)
            return ActionOutcome("outcome_unknown", None, claim.error or "not_found_on_reconcile")
        return await self._lookup_event(
            claim.request,
            calendar_event_id(claim.operation_id),
            unknown_error=claim.error or "not_found_on_reconcile",
        )

    async def _lookup_event(
        self, intended: EventRequest, event_id: str, *, unknown_error: ErrorCode
    ) -> ActionOutcome:
        found = await self._calendar.get_event(event_id)
        if found is None:
            return ActionOutcome("outcome_unknown", None, unknown_error)
        if event_matches(intended, found):
            return ActionOutcome("succeeded", event_id, None)
        return ActionOutcome("failed_definite", event_id, "event_id_conflict")

    async def _record(
        self, claim: _Claim, outcome: ActionOutcome, attempted: bool
    ) -> ActionOutcome:
        """Store the result (compare-and-set), audit it and emit `action_result`.

        `attempted` is False for a reconcile lookup; attempts were already counted at claim time.
        """
        async with self._sessionmaker() as session, session.begin():
            values: dict[str, Any] = {
                "status": outcome.status,
                "provider_ref": outcome.provider_ref,
                "error": outcome.error,
                "result": {
                    "status": outcome.status,
                    "provider_ref": outcome.provider_ref,
                    "error": outcome.error,
                },
                "updated_at": func.now(),
            }
            # Compare-and-set: a finished row is never overwritten.
            updated = (
                await session.execute(
                    update(Action)
                    .where(Action.id == claim.action_id, Action.status.not_in(_TERMINAL))
                    .values(**values)
                    .returning(Action.id)
                )
            ).scalar_one_or_none()
            if updated is None:
                row = await session.get_one(Action, claim.action_id)
                return ActionOutcome(
                    cast(ActionStatus, row.status),
                    row.provider_ref,
                    cast(ErrorCode | None, row.error),
                )
            if outcome.status == "succeeded":
                await session.execute(
                    update(Proposal)
                    .where(Proposal.id == claim.proposal_id)
                    .values(status="executed")
                )
            detail = {
                "action_id": str(claim.action_id),
                "operation_id": claim.operation_id,
                "proposal_id": str(claim.proposal_id),
                "kind": claim.kind,
                "status": outcome.status,
                "error": outcome.error,
                "reconciled": not attempted,
            }
            session.add(
                AuditEvent(
                    workflow_id=claim.workflow_id,
                    action_id=claim.action_id,
                    event_type="action_result",
                    detail=detail,
                )
            )
            session.add(
                WorkflowEvent(
                    workflow_id=claim.workflow_id, event_type="action_result", payload=detail
                )
            )
        return outcome


def _from_result(result: ProviderResult) -> ActionOutcome:
    if result.status == "confirmed":
        return ActionOutcome("succeeded", result.provider_ref, None)
    # Provider error text is never stored or shown; only the fixed code is.
    if result.status == "failed_definite":
        return ActionOutcome("failed_definite", result.provider_ref, "provider_failed")
    return ActionOutcome("outcome_unknown", result.provider_ref, "unexpected_error")
