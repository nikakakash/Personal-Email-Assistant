"""ActionExecutor: ledger claim, at-most-once sends, timeout reconciliation, preconditions."""

import asyncio
import re
import uuid
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.decisions import DecisionIn, decide_proposal
from app.actions.executor import (
    IN_FLIGHT_GRACE,
    ActionExecutor,
    ExecutionRefused,
    calendar_event_id,
    operation_id_for,
)
from app.config import Settings
from app.db.models import (
    Action,
    AppFlags,
    AuditEvent,
    MockCalendarEvent,
    MockOutbox,
    Proposal,
    ProposalRevision,
    WorkflowEvent,
)
from app.providers.mock_calendar import MockCalendarProvider
from app.providers.mock_mail import MockFaults, MockMailProvider
from tests.actions.conftest import Seeded, SeedFactory


def _settings(enabled: bool = True) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="test",
        app_base_url="http://localhost:8000",
        database_url="postgresql+psycopg://u:p@localhost:5432/test",
        checkpoint_database_url="postgresql://u:p@localhost:5432/test",
        owner_email="owner@example.com",
        owner_passphrase="a-real-passphrase",
        session_secret="a-real-session-secret",
        outbound_execution_enabled=enabled,
    )


async def _count(session: AsyncSession, model: type) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def _decide(
    session: AsyncSession,
    owner_id: uuid.UUID,
    s: Seeded,
    mail: MockMailProvider,
    decision: str = "approve",
) -> uuid.UUID:
    body = DecisionIn(
        decision=decision,  # type: ignore[arg-type]
        proposal_revision=s.revision,
        payload_hash=s.payload_hash,
        interrupt_id=s.interrupt_id,
        request_id=f"req-{uuid.uuid4().hex}",
    )
    out = await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    return out.approval_id


def _count_calls(monkeypatch: pytest.MonkeyPatch, provider: Any, name: str) -> list[int]:
    calls: list[int] = []
    original = getattr(provider, name)

    async def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return await original(*args, **kwargs)

    monkeypatch.setattr(provider, name, spy)
    return calls


@pytest.fixture
def executor(
    sessionmaker: async_sessionmaker[AsyncSession],
    mail: MockMailProvider,
    calendar: MockCalendarProvider,
) -> ActionExecutor:
    return ActionExecutor(sessionmaker, mail, calendar, _settings())


async def _action(session: AsyncSession) -> Action:
    return (
        await session.execute(select(Action).execution_options(populate_existing=True))
    ).scalar_one()


def test_event_id_charset() -> None:
    op = operation_id_for(uuid.uuid4(), 1)
    event_id = calendar_event_id(op)
    assert re.fullmatch(r"^[0-9a-v]{32}$", event_id)
    assert calendar_event_id(op) == event_id
    assert calendar_event_id(op + "x") != event_id


def test_operation_id_format() -> None:
    pid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert operation_id_for(pid, 3) == f"op-{pid}-3"


async def test_execute_twice_sends_once(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    executor: ActionExecutor,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    first = await executor.execute(approval_id)
    second = await executor.execute(approval_id)
    op = operation_id_for(s.proposal_id, 1)
    assert first.status == "succeeded" and first.provider_ref == op
    assert second == first
    assert await _count(session, MockOutbox) == 1
    outbox = (await session.execute(select(MockOutbox))).scalar_one()
    assert outbox.operation_id == op
    assert outbox.payload["body_text"] == s.payload["body_text"]
    assert outbox.payload["in_reply_to"] == s.payload["in_reply_to"]

    action = await _action(session)
    assert action.status == "succeeded"
    assert action.attempts == 1
    assert action.request_fingerprint == s.payload_hash
    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    assert proposal.status == "executed"
    events = (
        (
            await session.execute(
                select(WorkflowEvent.event_type).where(WorkflowEvent.event_type == "action_result")
            )
        )
        .scalars()
        .all()
    )
    assert len(events) == 1
    audit = (
        await session.execute(select(AuditEvent).where(AuditEvent.event_type == "action_result"))
    ).scalar_one()
    assert audit.action_id == action.id
    assert "Tuesday" not in str(audit.detail)


async def test_failed_definite_is_stored_and_not_retried(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    faults: MockFaults,
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    faults.send = "fail"
    first = await executor.execute(approval_id)
    assert first.status == "failed_definite"
    # The mock's error text ("mock send failure") is replaced by a fixed code everywhere.
    assert first.error == "provider_failed"
    action = await _action(session)
    assert action.error == "provider_failed"
    assert action.result is not None and "mock" not in str(action.result)
    audit = (await session.execute(select(AuditEvent.detail))).scalars().all()
    events = (await session.execute(select(WorkflowEvent.payload))).scalars().all()
    assert "mock" not in str(audit) and "mock" not in str(events)
    faults.send = "ok"
    calls = _count_calls(monkeypatch, mail, "send_reply")
    assert await executor.execute(approval_id) == first
    assert calls == []
    assert await _count(session, MockOutbox) == 0


async def test_timeout_after_accept_reconciles_to_succeeded_without_resend(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    faults: MockFaults,
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    faults.send = "timeout_after_accept"
    first = await executor.execute(approval_id)
    assert first.status == "outcome_unknown"
    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    assert proposal.status == "approved"

    faults.send = "ok"
    calls = _count_calls(monkeypatch, mail, "send_reply")
    second = await executor.execute(approval_id)
    assert second.status == "succeeded"
    assert calls == []
    assert await _count(session, MockOutbox) == 1
    action = await _action(session)
    assert action.status == "succeeded" and action.attempts == 1
    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    assert proposal.status == "executed"


async def test_timeout_before_accept_stays_unknown_no_resend(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    faults: MockFaults,
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    faults.send = "timeout_before_accept"
    assert (await executor.execute(approval_id)).status == "outcome_unknown"

    faults.send = "ok"
    calls = _count_calls(monkeypatch, mail, "send_reply")
    assert (await executor.execute(approval_id)).status == "outcome_unknown"
    assert calls == []
    assert await _count(session, MockOutbox) == 0
    action = await _action(session)
    assert action.status == "outcome_unknown" and action.attempts == 1


async def test_calendar_timeout_compares_event_by_id(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    faults: MockFaults,
    executor: ActionExecutor,
) -> None:
    s = await seed("create_event")
    approval_id = await _decide(session, owner_id, s, mail)
    faults.calendar_create = "timeout_after_accept"
    out = await executor.execute(approval_id)
    event_id = calendar_event_id(operation_id_for(s.proposal_id, 1))
    assert out.status == "succeeded"
    assert out.provider_ref == event_id
    row = await session.get_one(MockCalendarEvent, event_id)
    assert row.payload["title"] == "Lunch with Alice"
    assert await _count(session, MockCalendarEvent) == 1


async def test_calendar_timeout_with_conflicting_event_is_failed_definite(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    faults: MockFaults,
    executor: ActionExecutor,
) -> None:
    s = await seed("create_event")
    approval_id = await _decide(session, owner_id, s, mail)
    event_id = calendar_event_id(operation_id_for(s.proposal_id, 1))
    other = {**s.payload, "title": "Something else"}
    session.add(MockCalendarEvent(event_id=event_id, payload=other))
    await session.commit()
    faults.calendar_create = "timeout_after_accept"
    out = await executor.execute(approval_id)
    assert out.status == "failed_definite"
    assert out.error == "event_id_conflict"


async def test_calendar_timeout_without_event_is_unknown(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    faults: MockFaults,
    calendar: MockCalendarProvider,
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed("create_event")
    approval_id = await _decide(session, owner_id, s, mail)
    faults.calendar_create = "timeout_before_accept"
    assert (await executor.execute(approval_id)).status == "outcome_unknown"
    faults.calendar_create = "ok"
    calls = _count_calls(monkeypatch, calendar, "create_event")
    assert (await executor.execute(approval_id)).status == "outcome_unknown"
    assert calls == []
    assert await _count(session, MockCalendarEvent) == 0


async def test_executor_refuses_when_outbound_disabled(
    session: AsyncSession,
    sessionmaker: async_sessionmaker[AsyncSession],
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    calendar: MockCalendarProvider,
    executor: ActionExecutor,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    disabled = ActionExecutor(sessionmaker, mail, calendar, _settings(enabled=False))
    with pytest.raises(ExecutionRefused) as exc_info:
        await disabled.execute(approval_id)
    assert exc_info.value.reason == "outbound_disabled"

    session.add(AppFlags(id=1, outbound_paused=True))
    await session.commit()
    with pytest.raises(ExecutionRefused) as exc_info:
        await executor.execute(approval_id)
    assert exc_info.value.reason == "outbound_paused"
    assert await _count(session, Action) == 0
    assert await _count(session, MockOutbox) == 0


async def test_executor_refuses_unapproved_revision(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    executor: ActionExecutor,
) -> None:
    rejected = await seed()
    reject_id = await _decide(session, owner_id, rejected, mail, decision="reject")
    with pytest.raises(ExecutionRefused) as exc_info:
        await executor.execute(reject_id)
    assert exc_info.value.reason == "not_approved"

    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    session.add(
        ProposalRevision(
            proposal_id=s.proposal_id,
            revision=2,
            payload=s.payload,
            payload_hash=s.payload_hash,
            author="owner",
        )
    )
    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    proposal.active_revision = 2
    await session.commit()
    with pytest.raises(ExecutionRefused) as exc_info:
        await executor.execute(approval_id)
    assert exc_info.value.reason == "revision_not_active"

    with pytest.raises(ExecutionRefused) as exc_info:
        await executor.execute(uuid.uuid4())
    assert exc_info.value.reason == "approval_not_found"
    assert await _count(session, Action) == 0


def _crash_record_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the first `_record` raise, as if the process died after the provider call."""
    original = ActionExecutor._record
    crashed: list[bool] = []

    async def flaky(self: ActionExecutor, *args: Any, **kwargs: Any) -> Any:
        if not crashed:
            crashed.append(True)
            raise RuntimeError("process died before recording the result")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ActionExecutor, "_record", flaky)


async def _age_action(session: AsyncSession, by: timedelta) -> None:
    await session.execute(update(Action).values(updated_at=func.now() - by))
    await session.commit()


async def test_crash_after_provider_accept_before_record_reconciles(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    calls = _count_calls(monkeypatch, mail, "send_reply")
    _crash_record_once(monkeypatch)
    with pytest.raises(RuntimeError):
        await executor.execute(approval_id)
    assert calls == [1]
    assert await _count(session, MockOutbox) == 1
    assert (await _action(session)).status == "executing"

    await _age_action(session, IN_FLIGHT_GRACE + timedelta(seconds=30))
    out = await executor.execute(approval_id)
    assert out.status == "succeeded"
    assert calls == [1]  # reconciled through find_sent, never sent again
    assert await _count(session, MockOutbox) == 1
    action = await _action(session)
    assert action.status == "succeeded" and action.attempts == 1
    proposal = await session.get_one(Proposal, s.proposal_id, populate_existing=True)
    assert proposal.status == "executed"


async def test_in_flight_row_within_grace_is_left_alone(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    _crash_record_once(monkeypatch)
    with pytest.raises(RuntimeError):
        await executor.execute(approval_id)

    sends = _count_calls(monkeypatch, mail, "send_reply")
    lookups = _count_calls(monkeypatch, mail, "find_sent")
    out = await executor.execute(approval_id)
    assert out.status == "executing"
    assert sends == [] and lookups == []
    assert await _count(session, WorkflowEvent) == 1  # only approval_required, no action_result
    assert await _count(session, AuditEvent) == 1  # only decision_recorded
    assert (await _action(session)).status == "executing"

    await _age_action(session, IN_FLIGHT_GRACE + timedelta(seconds=30))
    assert (await executor.execute(approval_id)).status == "succeeded"
    assert sends == [] and lookups == [1]


async def test_concurrent_execute_sends_once(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    executor: ActionExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = await seed()
    approval_id = await _decide(session, owner_id, s, mail)
    calls = _count_calls(monkeypatch, mail, "send_reply")
    results = await asyncio.gather(executor.execute(approval_id), executor.execute(approval_id))
    assert calls == [1]
    assert await _count(session, MockOutbox) == 1
    assert "succeeded" in {r.status for r in results}
    assert {r.status for r in results} <= {"succeeded", "executing"}
    action = await _action(session)
    assert action.status == "succeeded" and action.attempts == 1


async def test_execute_edit_approved_revision(
    session: AsyncSession,
    owner_id: uuid.UUID,
    seed: SeedFactory,
    mail: MockMailProvider,
    executor: ActionExecutor,
) -> None:
    s = await seed()
    body = DecisionIn(
        decision="edit_approve",
        proposal_revision=s.revision,
        payload_hash=s.payload_hash,
        interrupt_id=s.interrupt_id,
        request_id="req-edit",
        edited_payload={"body_text": "Edited: Wednesday instead — ünïcode ok?"},
    )
    decided = await decide_proposal(session, owner_id, s.proposal_id, body, mail=mail)
    out = await executor.execute(decided.approval_id)
    assert out.status == "succeeded"  # no payload_hash_mismatch after the JSONB round trip
    outbox = (await session.execute(select(MockOutbox))).scalar_one()
    assert outbox.operation_id == operation_id_for(s.proposal_id, 2)
    assert outbox.operation_id.endswith("-2")
    assert outbox.payload["body_text"] == "Edited: Wednesday instead — ünïcode ok?"
