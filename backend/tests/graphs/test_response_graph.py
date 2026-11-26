"""Response subgraph, driven through the compiled parent graph: reads, drafts, interrupts,
decisions, preflight and execution."""

import asyncio
import dataclasses
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langgraph.runtime import Runtime
from langgraph.types import Command
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.executor import IN_FLIGHT_GRACE, build_reply, operation_id_for
from app.db.models import (
    Action,
    AppFlags,
    Approval,
    EmailMessage,
    MockCalendarEvent,
    MockOutbox,
    Owner,
    Proposal,
    ProposalRevision,
    ReviewItem,
    WorkflowEvent,
)
from app.graphs.deps import GraphDeps
from app.graphs.nodes import execute
from app.graphs.nodes.ask import question_item_id
from app.graphs.nodes.execute import (
    PROPOSAL_TTL,
    ActionInFlight,
)
from app.graphs.nodes.proposal import persist_proposal
from app.llm.client import BudgetExceeded, LLMCall, LLMResult, T
from app.llm.scripted import ScriptedLLMClient
from app.providers.interfaces import OutgoingReply, ProviderResult, ThreadSnapshot
from app.providers.mock_mail import MockFaults, MockMailProvider
from tests.graphs.conftest import (
    AVAILABILITY_KEY,
    NOW,
    TUESDAY,
    DepsFactory,
    Graph,
    SeededRun,
    count,
    decide,
    freebusy_plan,
    meeting_out,
    plan_out,
    reply_out,
    resume,
    run_graph,
    run_row,
    subgraph_values,
    triage_out,
)

Session = async_sessionmaker[AsyncSession]


def _reply_script(mid: str, *, plans_after: int = 1) -> dict[str, Any]:
    """Triage respond → freebusy → draft_reply, then `plans_after` finish steps."""
    return {
        mid: {
            "triage": [triage_out("respond", mid)],
            "plan": [freebusy_plan(), plan_out("draft_reply")] + [plan_out("finish")] * plans_after,
            "draft_reply": [reply_out(mid)],
        }
    }


def _meeting_script(mid: str) -> dict[str, Any]:
    return {
        mid: {
            "triage": [triage_out("respond", mid)],
            "plan": [freebusy_plan(), plan_out("draft_meeting"), plan_out("finish")],
            "draft_meeting": [meeting_out()],
        }
    }


async def _proposals(sessionmaker: Session) -> list[Proposal]:
    async with sessionmaker() as s:
        return list((await s.execute(select(Proposal).order_by(Proposal.created_at))).scalars())


async def _event_payloads(sessionmaker: Session, event_type: str) -> list[dict[str, Any]]:
    async with sessionmaker() as s:
        rows = await s.execute(
            select(WorkflowEvent.payload)
            .where(WorkflowEvent.event_type == event_type)
            .order_by(WorkflowEvent.id)
        )
        return [dict(p) for p in rows.scalars()]


async def _add_followup(sessionmaker: Session, seeded: SeededRun) -> None:
    """A follow-up message arrives in the thread (its version changes)."""
    async with sessionmaker() as s:
        s.add(
            EmailMessage(
                thread_id=seeded.thread_row_id,
                account_id="mock-account",
                provider_message_id=f"{seeded.provider_message_id}-2",
                sender="alice@example.com",
                to_addrs=["owner@example.com"],
                subject="Re: Meeting next week?",
                body_text="Actually, Wednesday is better.",
                received_at=NOW,
                content_hash="h-followup",
            )
        )
        await s.commit()


async def test_respond_scheduling_full_path(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    deps = make_deps(_reply_script(seeded.provider_message_id))

    paused = await run_graph(graph, seeded.input, seeded, deps)

    (interrupt,) = paused["__interrupt__"]
    assert interrupt.value["type"] == "approval"
    inner = await subgraph_values(graph, seeded)
    finding = next(f for f in inner["read_findings"] if f["key"] == AVAILABILITY_KEY)
    assert finding["candidate_slots"][0]["start"] == f"{TUESDAY}T09:00:00+00:00"
    assert await _event_payloads(sessionmaker, "tool_result") == [
        {"tool": "get_freebusy", "key": AVAILABILITY_KEY}
    ]
    (proposal,) = await _proposals(sessionmaker)
    assert (proposal.kind, proposal.status) == ("send_reply", "awaiting_approval")
    assert interrupt.value["review_item_id"] == inner["review_item_id"]
    assert (await run_row(sessionmaker, seeded)).status == "waiting_review"

    done = await resume(graph, await decide(sessionmaker, owner, paused, mail), seeded, deps)

    assert "__interrupt__" not in done
    (result,) = done["action_results"]
    assert result["status"] == "succeeded"
    assert result["proposal_id"] == str(proposal.id)
    assert await count(sessionmaker, MockOutbox) == 1
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.step_count, run.error_code) == ("done_with_actions", 3, None)
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "executed"
    async with sessionmaker() as s:  # ids in state are the DB-verified ones
        approval = await s.get_one(Approval, uuid.UUID(done["approval_id"]))
        revision = await s.get_one(ProposalRevision, approval.proposal_revision_id)
    assert (revision.proposal_id, revision.revision) == (proposal.id, done["proposal_revision"])


async def test_question_interrupt_answer_resumes_same_thread(
    graph: Graph, make_deps: DepsFactory, seeded: SeededRun, sessionmaker: Session
) -> None:
    mid = seeded.provider_message_id
    deps = make_deps(
        {
            mid: {
                "triage": [triage_out("respond", mid)],
                "plan": [
                    plan_out("ask_owner", question="Which Tuesday do they mean?"),
                    plan_out("finish"),
                ],
            }
        }
    )

    paused = await run_graph(graph, seeded.input, seeded, deps)

    item_id = question_item_id(seeded.workflow_id, 1)
    assert paused["__interrupt__"][0].value == {
        "type": "question",
        "review_item_id": str(item_id),
    }
    async with sessionmaker() as s:
        item = await s.get_one(ReviewItem, item_id)
        assert (item.kind, item.status) == ("question", "pending")
        assert item.payload == {"question": "Which Tuesday do they mean?"}

    done = await resume(graph, {"decision": "answer", "answer": "Next Tuesday."}, seeded, deps)

    assert "__interrupt__" not in done
    assert done["owner_inputs"] == [{"kind": "answer", "text": "Next Tuesday."}]
    state = await graph.aget_state(seeded.config)  # type: ignore[arg-type]
    assert state.values["owner_inputs"] == done["owner_inputs"]
    assert state.values["step_count"] == 2
    assert (await run_row(sessionmaker, seeded)).status == "done_no_action"


async def test_feedback_loops_to_new_revision(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    mid = seeded.provider_message_id
    deps = make_deps(
        {
            mid: {
                "triage": [triage_out("respond", mid)],
                "plan": [freebusy_plan(), plan_out("draft_reply"), plan_out("draft_reply")],
                "draft_reply": [reply_out(mid), reply_out(mid, body="Tuesday 09:00 works.")],
            }
        }
    )
    first = await run_graph(graph, seeded.input, seeded, deps)
    first_item = (await subgraph_values(graph, seeded))["review_item_id"]

    feedback = await decide(
        sessionmaker, owner, first, mail, decision="feedback", feedback="Shorter, please."
    )
    second = await resume(graph, feedback, seeded, deps)

    (interrupt,) = second["__interrupt__"]
    inner = await subgraph_values(graph, seeded)
    assert interrupt.value["review_item_id"] == inner["review_item_id"] != first_item
    assert inner["owner_inputs"] == [{"kind": "feedback", "text": "Shorter, please."}]
    old, new = await _proposals(sessionmaker)
    assert (old.status, new.status) == ("superseded", "awaiting_approval")
    assert inner["proposal_id"] == str(new.id)
    assert new.idempotency_key == f"{seeded.workflow_id}:3"
    async with sessionmaker() as s:
        revision = (
            await s.execute(select(ProposalRevision).where(ProposalRevision.proposal_id == new.id))
        ).scalar_one()
    assert revision.payload["body_text"] == "Tuesday 09:00 works."


async def test_reject_sends_nothing(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    deps = make_deps(_reply_script(seeded.provider_message_id))
    paused = await run_graph(graph, seeded.input, seeded, deps)

    done = await resume(
        graph, await decide(sessionmaker, owner, paused, mail, decision="reject"), seeded, deps
    )

    assert "__interrupt__" not in done
    assert done.get("action_results", []) == []
    assert await count(sessionmaker, MockOutbox) == 0
    assert await count(sessionmaker, Action) == 0
    assert (await run_row(sessionmaker, seeded)).status == "done_no_action"
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "rejected"


async def test_preflight_stale_when_thread_changed(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    deps = make_deps(_reply_script(seeded.provider_message_id))
    paused = await run_graph(graph, seeded.input, seeded, deps)
    inner = await subgraph_values(graph, seeded)
    approve = await decide(sessionmaker, owner, paused, mail)
    await _add_followup(sessionmaker, seeded)

    done = await resume(graph, approve, seeded, deps)

    assert await count(sessionmaker, MockOutbox) == 0
    assert await count(sessionmaker, Action) == 0
    stale = next(f for f in done["read_findings"] if f["kind"] == "stale")
    assert stale["key"] == "stale:1"
    assert stale["reason"] == "thread_changed"
    assert done["source_version"] != inner["source_version"]
    assert len(done["thread"]["messages"]) == 2
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "stale"
    async with sessionmaker() as s:
        item = await s.get_one(ReviewItem, uuid.UUID(inner["review_item_id"]))
    assert item.status == "stale"
    assert (await run_row(sessionmaker, seeded)).status == "done_no_action"


async def test_preflight_slot_taken_for_meeting(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    deps = make_deps(_meeting_script(seeded.provider_message_id))
    paused = await run_graph(graph, seeded.input, seeded, deps)
    assert (await subgraph_values(graph, seeded))["draft"]["start"] == f"{TUESDAY}T09:00:00+00:00"
    approve = await decide(sessionmaker, owner, paused, mail)
    async with sessionmaker() as s:  # someone books the slot after the approval
        s.add(
            MockCalendarEvent(
                event_id="other-booking",
                payload={
                    "start": f"{TUESDAY}T09:15:00+00:00",
                    "end": f"{TUESDAY}T10:00:00+00:00",
                },
            )
        )
        await s.commit()

    done = await resume(graph, approve, seeded, deps)

    stale = next(f for f in done["read_findings"] if f["kind"] == "stale")
    assert stale["reason"] == "slot_taken"
    assert await count(sessionmaker, Action) == 0
    assert await count(sessionmaker, MockCalendarEvent) == 1
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "stale"


async def test_meeting_executes_when_slot_still_free(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    deps = make_deps(_meeting_script(seeded.provider_message_id))
    paused = await run_graph(graph, seeded.input, seeded, deps)

    done = await resume(graph, await decide(sessionmaker, owner, paused, mail), seeded, deps)

    assert [r["status"] for r in done["action_results"]] == ["succeeded"]
    assert await count(sessionmaker, MockCalendarEvent) == 1
    assert (await run_row(sessionmaker, seeded)).status == "done_with_actions"


async def test_preflight_expired_proposal(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    deps = make_deps(_reply_script(seeded.provider_message_id))
    paused = await run_graph(graph, seeded.input, seeded, deps)
    inner = await subgraph_values(graph, seeded)
    approve = await decide(sessionmaker, owner, paused, mail)
    async with sessionmaker() as s:
        await s.execute(
            update(ProposalRevision)
            .where(ProposalRevision.proposal_id == uuid.UUID(inner["proposal_id"]))
            .values(created_at=NOW - PROPOSAL_TTL - timedelta(minutes=1))
        )
        await s.commit()

    done = await resume(graph, approve, seeded, deps)

    stale = next(f for f in done["read_findings"] if f["kind"] == "stale")
    assert stale["reason"] == "expired"
    assert await count(sessionmaker, MockOutbox) == 0
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "stale"


async def test_outbound_disabled_blocks_execution(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    """Blocked after approval: the run waits at a pause interrupt instead of ending."""
    script = _reply_script(seeded.provider_message_id)
    deps = make_deps(script, outbound=False)
    paused = await run_graph(graph, seeded.input, seeded, deps)

    blocked = await resume(graph, await decide(sessionmaker, owner, paused, mail), seeded, deps)

    (pause,) = blocked["__interrupt__"]
    assert pause.value == {"type": "pause", "reason": "outbound_disabled"}
    inner = await subgraph_values(graph, seeded)
    assert inner["preflight"] == {"result": "blocked", "reason": "outbound_disabled"}
    assert await count(sessionmaker, MockOutbox) == 0
    assert await count(sessionmaker, Action) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("paused", "outbound_disabled")


async def test_retry_after_outbound_enabled_sends_once(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    """Retry goes back to preflight with the same approval: every check runs again."""
    script = _reply_script(seeded.provider_message_id)
    llm = ScriptedLLMClient(script)
    paused = await run_graph(graph, seeded.input, seeded, make_deps(llm=llm, outbound=False))
    approve = await decide(sessionmaker, owner, paused, mail)
    await resume(graph, approve, seeded, make_deps(llm=llm, outbound=False))

    # Still disabled: pauses again, nothing sent, the model is not called.
    again = await resume(graph, {"decision": "retry"}, seeded, make_deps(llm=llm, outbound=False))
    assert again["__interrupt__"][0].value == {"type": "pause", "reason": "outbound_disabled"}
    assert await count(sessionmaker, MockOutbox) == 0

    done = await resume(graph, {"decision": "retry"}, seeded, make_deps(llm=llm, outbound=True))

    assert "__interrupt__" not in done
    assert [r["status"] for r in done["action_results"]] == ["succeeded"]
    assert await count(sessionmaker, MockOutbox) == 1
    assert len(await _proposals(sessionmaker)) == 1  # no re-draft: the approved payload is sent
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("done_with_actions", None)


async def test_retry_after_step_budget_grants_new_steps(
    graph: Graph, make_deps: DepsFactory, seeded: SeededRun, sessionmaker: Session
) -> None:
    mid = seeded.provider_message_id
    script = {
        mid: {
            "triage": [triage_out("respond", mid)],
            "plan": [freebusy_plan(), freebusy_plan(), plan_out("finish")],
        }
    }
    deps = make_deps(script)
    deps = dataclasses.replace(
        deps, settings=deps.settings.model_copy(update={"max_agent_steps": 2})
    )

    paused = await run_graph(graph, seeded.input, seeded, deps)

    assert paused["__interrupt__"][0].value == {"type": "pause", "reason": "step_budget"}
    assert (await run_row(sessionmaker, seeded)).status == "paused"

    done = await resume(graph, {"decision": "retry"}, seeded, deps)

    assert "__interrupt__" not in done
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("done_no_action", None)


async def test_approval_for_inactive_revision_is_blocked(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    deps = make_deps(_reply_script(seeded.provider_message_id))
    paused = await run_graph(graph, seeded.input, seeded, deps)
    approve = await decide(sessionmaker, owner, paused, mail)
    async with sessionmaker() as s:
        await s.execute(
            update(Approval)
            .where(Approval.id == uuid.UUID(approve["approval_id"]))
            .values(decision="reject")
        )
        await s.commit()

    done = await resume(graph, approve, seeded, deps)

    assert done["preflight"] == {"result": "blocked", "reason": "approval_invalid"}
    assert await count(sessionmaker, MockOutbox) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("failed", "approval_invalid")


class BudgetOnDraft:
    """Scripted outputs, except that drafting hits the daily budget."""

    def __init__(self, inner: ScriptedLLMClient) -> None:
        self.inner = inner

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        if call.purpose == "draft_reply":
            raise BudgetExceeded("daily budget reached")
        return await self.inner.structured(call)


async def test_validate_budget_stop_route(
    graph: Graph, make_deps: DepsFactory, seeded: SeededRun, sessionmaker: Session
) -> None:
    llm = BudgetOnDraft(ScriptedLLMClient(_reply_script(seeded.provider_message_id)))

    paused = await run_graph(graph, seeded.input, seeded, make_deps(llm=llm))

    assert paused["__interrupt__"][0].value == {"type": "pause", "reason": "budget"}
    assert (await subgraph_values(graph, seeded))["next_action"] == "budget_stop"
    assert await count(sessionmaker, Proposal) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("paused", "budget")


async def test_cancelled_while_paused_ends_cleanly(
    graph: Graph, make_deps: DepsFactory, seeded: SeededRun, sessionmaker: Session
) -> None:
    llm = BudgetOnDraft(ScriptedLLMClient(_reply_script(seeded.provider_message_id)))
    deps = make_deps(llm=llm)
    await run_graph(graph, seeded.input, seeded, deps)

    done = await resume(graph, {"decision": "cancelled"}, seeded, deps)

    assert "__interrupt__" not in done
    assert await count(sessionmaker, Proposal) == 0


class CountingLLM:
    """Scripted outputs; keeps the purpose of every call."""

    def __init__(self, inner: ScriptedLLMClient) -> None:
        self.inner = inner
        self.purposes: list[str] = []

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        self.purposes.append(call.purpose)
        return await self.inner.structured(call)


class SlowMail:
    """The mock mail provider, except that `get_thread` hangs once enabled.

    Set `hang = True` to hang every read, or `hang_after = n` to let n more reads through.
    """

    def __init__(self, inner: MockMailProvider) -> None:
        self.inner = inner
        self.hang = False
        self.hang_after: int | None = None

    async def get_thread(self, provider_thread_id: str, limit: int = 20) -> ThreadSnapshot:
        if self.hang_after is not None:
            if self.hang_after == 0:
                self.hang = True
            self.hang_after -= 1
        if self.hang:
            await asyncio.sleep(5)
        return await self.inner.get_thread(provider_thread_id, limit)

    async def send_reply(self, req: OutgoingReply, operation_id: str) -> ProviderResult:
        return await self.inner.send_reply(req, operation_id)

    async def find_sent(self, operation_id: str) -> ProviderResult | None:
        return await self.inner.find_sent(operation_id)


async def _raise_in_flight(
    graph: Graph,
    deps: GraphDeps,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> uuid.UUID:
    """Approve, let another caller hold a fresh `executing` claim, resume: ActionInFlight."""
    paused = await run_graph(graph, seeded.input, seeded, deps)
    inner = await subgraph_values(graph, seeded)
    approve = await decide(sessionmaker, owner, paused, mail)
    action_id = uuid.uuid4()
    async with sessionmaker() as s:  # another caller has just claimed the operation
        s.add(
            Action(
                id=action_id,
                operation_id=operation_id_for(uuid.UUID(inner["proposal_id"]), 1),
                approval_id=uuid.UUID(approve["approval_id"]),
                kind="send_reply",
                request_fingerprint="in-flight",
                status="executing",
                attempts=1,
                updated_at=datetime.now(UTC),
            )
        )
        await s.commit()

    with pytest.raises(ActionInFlight):
        await resume(graph, approve, seeded, deps)

    assert await count(sessionmaker, MockOutbox) == 0
    state = await graph.aget_state(seeded.config)  # type: ignore[arg-type]
    assert state.next == ("response",)  # the resume can be retried from here
    return action_id


async def test_execute_in_flight_raises(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    """The worker's retry, `ainvoke(None, config)`, re-runs only `execute_action`."""
    deps = make_deps(_reply_script(seeded.provider_message_id))
    action_id = await _raise_in_flight(graph, deps, seeded, sessionmaker, owner, mail)
    async with sessionmaker() as s:  # the other caller's send went through
        await s.execute(
            update(Action)
            .where(Action.id == action_id)
            .values(status="succeeded", provider_ref="sent-by-other-caller")
        )
        await s.commit()

    done = await run_graph(graph, None, seeded, deps)

    assert "__interrupt__" not in done
    assert [(r["status"], r["provider_ref"]) for r in done["action_results"]] == [
        ("succeeded", "sent-by-other-caller")
    ]
    assert await count(sessionmaker, Action) == 1
    assert (await run_row(sessionmaker, seeded)).status == "done_with_actions"


async def test_retry_in_flight_with_changed_thread_never_goes_stale(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    """Ledger first: with a claimed action, a changed thread does not make the retry stale."""
    llm = CountingLLM(ScriptedLLMClient(_reply_script(seeded.provider_message_id)))
    deps = make_deps(llm=llm)
    action_id = await _raise_in_flight(graph, deps, seeded, sessionmaker, owner, mail)
    await _add_followup(sessionmaker, seeded)

    with pytest.raises(ActionInFlight):  # still the other caller's send
        await run_graph(graph, None, seeded, deps)

    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "approved"  # not flipped to stale mid-send
    async with sessionmaker() as s:  # the other caller finishes and records its result
        await s.execute(
            update(Action)
            .where(Action.id == action_id)
            .values(status="succeeded", provider_ref="sent-by-other-caller")
        )
        await s.execute(update(Proposal).values(status="executed"))
        await s.commit()

    done = await run_graph(graph, None, seeded, deps)

    assert [r["status"] for r in done["action_results"]] == ["succeeded"]
    assert not [f for f in done["read_findings"] if f["kind"] == "stale"]
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "executed"
    assert llm.purposes == ["triage", "plan", "plan", "draft_reply", "plan"]
    assert (await run_row(sessionmaker, seeded)).status == "done_with_actions"


class WorkerCrash(BaseException):
    """Stands in for the process dying: not an Exception, so nothing in the app catches it."""


class CrashAfterSend:
    """The mock mail provider, except that the process "dies" right after a send is accepted."""

    def __init__(self, inner: MockMailProvider) -> None:
        self.inner = inner
        self.crash = True
        self.hide_sent = False  # the provider's search misses the sent message

    async def get_thread(self, provider_thread_id: str, limit: int = 20) -> ThreadSnapshot:
        return await self.inner.get_thread(provider_thread_id, limit)

    async def send_reply(self, req: OutgoingReply, operation_id: str) -> ProviderResult:
        result = await self.inner.send_reply(req, operation_id)
        if self.crash:
            raise WorkerCrash
        return result

    async def find_sent(self, operation_id: str) -> ProviderResult | None:
        if self.hide_sent:
            return None
        return await self.inner.find_sent(operation_id)


async def test_retry_after_crash_post_send_with_changed_thread(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
) -> None:
    """The send went out, the worker died before recording it, then the thread changed.

    The retry must reconcile the sent reply, not call it stale and draft another one.
    """
    crashing = CrashAfterSend(mail)
    llm = CountingLLM(ScriptedLLMClient(_reply_script(seeded.provider_message_id)))
    deps = make_deps(llm=llm, mail_provider=crashing)
    paused = await run_graph(graph, seeded.input, seeded, deps)
    approve = await decide(sessionmaker, owner, paused, mail)

    with pytest.raises(WorkerCrash):
        await resume(graph, approve, seeded, deps)

    assert await count(sessionmaker, MockOutbox) == 1
    crashing.crash = False
    await _add_followup(sessionmaker, seeded)
    async with sessionmaker() as s:  # the dead worker's claim is past the in-flight grace
        await s.execute(update(Action).values(updated_at=datetime.now(UTC) - IN_FLIGHT_GRACE * 2))
        await s.commit()

    done = await run_graph(graph, None, seeded, deps)

    assert [r["status"] for r in done["action_results"]] == ["succeeded"]
    assert not [f for f in done["read_findings"] if f["kind"] == "stale"]
    assert await count(sessionmaker, MockOutbox) == 1
    assert await count(sessionmaker, Action) == 1
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "executed"
    assert llm.purposes == ["triage", "plan", "plan", "draft_reply", "plan"]
    assert (await run_row(sessionmaker, seeded)).status == "done_with_actions"


@pytest.mark.parametrize(
    ("row_status", "hide_sent", "kill_switch", "run_status", "error_code"),
    [
        ("succeeded", False, "flag", "done_with_actions", None),
        ("succeeded", False, "setting", "done_with_actions", None),
        ("outcome_unknown", False, "flag", "done_with_actions", None),
        ("outcome_unknown", True, "flag", "failed", "outcome_unknown"),
    ],
)
async def test_retry_with_outbound_paused_still_records_the_send(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
    row_status: str,
    hide_sent: bool,
    kill_switch: str,
    run_status: str,
    error_code: str | None,
) -> None:
    """The kill switch blocks new sends only; a send already in the ledger is still recorded."""
    crashing = CrashAfterSend(mail)
    llm = CountingLLM(ScriptedLLMClient(_reply_script(seeded.provider_message_id)))
    deps = make_deps(llm=llm, mail_provider=crashing)
    paused = await run_graph(graph, seeded.input, seeded, deps)
    approve = await decide(sessionmaker, owner, paused, mail)
    with pytest.raises(WorkerCrash):  # the reply is sent, then the worker dies
        await resume(graph, approve, seeded, deps)
    crashing.crash, crashing.hide_sent = False, hide_sent
    values: dict[str, Any] = {"status": row_status, "updated_at": datetime.now(UTC)}
    if row_status == "succeeded":  # the result was recorded just before the crash
        values["provider_ref"] = "recorded-ref"
    async with sessionmaker() as s:
        await s.execute(update(Action).values(**values))
        if kill_switch == "flag":
            s.add(AppFlags(id=1, outbound_paused=True))
        await s.commit()
    if kill_switch == "setting":
        deps = make_deps(llm=llm, mail_provider=crashing, outbound=False)

    done = await run_graph(graph, None, seeded, deps)

    assert [r["status"] for r in done["action_results"]] == [
        "outcome_unknown" if hide_sent else "succeeded"
    ]
    assert await count(sessionmaker, MockOutbox) == 1
    assert await count(sessionmaker, Action) == 1
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == (run_status, error_code)
    expected_llm = ["triage", "plan", "plan", "draft_reply"] + ([] if hide_sent else ["plan"])
    assert llm.purposes == expected_llm


@pytest.mark.parametrize("row_status", ["executing", "succeeded"])
async def test_stale_preflight_with_claimed_action_goes_to_the_executor(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
    row_status: str,
) -> None:
    """Another caller claims (or completes) the send, then the thread changes, then we resume.

    preflight → stale → `mark_stale` sees the ledger row for the active operation, keeps the
    verified approval and routes to `execute_action`; nothing is marked stale.
    """
    llm = CountingLLM(ScriptedLLMClient(_reply_script(seeded.provider_message_id)))
    deps = make_deps(llm=llm)
    paused = await run_graph(graph, seeded.input, seeded, deps)
    inner = await subgraph_values(graph, seeded)
    approve = await decide(sessionmaker, owner, paused, mail)
    proposal_id = uuid.UUID(inner["proposal_id"])
    op = operation_id_for(proposal_id, 1)
    values: dict[str, Any] = {"status": "executing"}
    if row_status == "succeeded":  # the other caller's send went through and was recorded
        async with sessionmaker() as s:
            revision = await s.scalar(
                select(ProposalRevision).where(ProposalRevision.proposal_id == proposal_id)
            )
        assert revision is not None
        sent = await mail.send_reply(build_reply(revision.payload), op)
        values = {"status": "succeeded", "provider_ref": sent.provider_ref}
    async with sessionmaker() as s:
        s.add(
            Action(
                operation_id=op,
                approval_id=uuid.UUID(approve["approval_id"]),
                kind="send_reply",
                request_fingerprint="other-caller",
                attempts=1,
                updated_at=datetime.now(UTC),
                **values,
            )
        )
        if row_status == "succeeded":
            await s.execute(update(Proposal).values(status="executed"))
        await s.commit()
    await _add_followup(sessionmaker, seeded)  # preflight will see a changed thread

    if row_status == "executing":
        with pytest.raises(ActionInFlight):
            await resume(graph, approve, seeded, deps)
        (proposal,) = await _proposals(sessionmaker)
        assert proposal.status == "approved"
        async with sessionmaker() as s:
            item = await s.get_one(ReviewItem, uuid.UUID(inner["review_item_id"]))
        assert item.status != "stale"
        findings = (await subgraph_values(graph, seeded))["read_findings"]
        assert not [f for f in findings if f["kind"] == "stale"]
        assert await count(sessionmaker, MockOutbox) == 0
        return

    done = await resume(graph, approve, seeded, deps)

    assert [r["status"] for r in done["action_results"]] == ["succeeded"]
    assert not [f for f in done["read_findings"] if f["kind"] == "stale"]
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "executed"
    assert llm.purposes == ["triage", "plan", "plan", "draft_reply", "plan"]
    assert await count(sessionmaker, MockOutbox) == 1
    assert await count(sessionmaker, Action) == 1
    assert (await run_row(sessionmaker, seeded)).status == "done_with_actions"


async def test_mark_stale_thread_reread_failure_blocks(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight sees the change, then the re-read in `mark_stale` times out."""
    slow = SlowMail(mail)
    deps = make_deps(_reply_script(seeded.provider_message_id), mail_provider=slow)
    paused = await run_graph(graph, seeded.input, seeded, deps)
    approve = await decide(sessionmaker, owner, paused, mail)
    await _add_followup(sessionmaker, seeded)
    monkeypatch.setattr(execute, "PREFLIGHT_TIMEOUT_SECONDS", 1.0)
    slow.hang_after = 1  # preflight's read works, the next one hangs

    blocked = await resume(graph, approve, seeded, deps)

    assert blocked["__interrupt__"][0].value == {
        "type": "pause",
        "reason": "preflight_unavailable",
    }
    inner = await subgraph_values(graph, seeded)
    assert inner["preflight"] == {"result": "blocked", "reason": "preflight_unavailable"}
    (proposal,) = await _proposals(sessionmaker)
    assert proposal.status == "stale"
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("paused", "preflight_unavailable")

    # The proposal is already retired, so Retry re-plans instead of re-checking it.
    slow.hang, slow.hang_after = False, None
    done = await resume(graph, {"decision": "retry"}, seeded, deps)

    assert "__interrupt__" not in done
    assert await count(sessionmaker, MockOutbox) == 0
    assert (await run_row(sessionmaker, seeded)).status == "done_no_action"


async def test_outcome_unknown_needs_review(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
    faults: MockFaults,
) -> None:
    mid = seeded.provider_message_id
    script = _reply_script(mid)
    # If the planner ran again it would draft a second reply; it must not.
    script[mid]["plan"][2:] = [plan_out("draft_reply")]
    script[mid]["draft_reply"].append(reply_out(mid, body="Second try."))
    llm = CountingLLM(ScriptedLLMClient(script))
    deps = make_deps(llm=llm)
    paused = await run_graph(graph, seeded.input, seeded, deps)
    faults.send = "timeout_before_accept"

    done = await resume(graph, await decide(sessionmaker, owner, paused, mail), seeded, deps)

    assert "__interrupt__" not in done
    assert [r["status"] for r in done["action_results"]] == ["outcome_unknown"]
    assert llm.purposes == ["triage", "plan", "plan", "draft_reply"]
    assert await count(sessionmaker, Action) == 1
    assert await count(sessionmaker, Proposal) == 1
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("failed", "outcome_unknown")


async def test_preflight_provider_timeout_blocks(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: Session,
    owner: Owner,
    mail: MockMailProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute, "PREFLIGHT_TIMEOUT_SECONDS", 1.0)
    slow = SlowMail(mail)
    deps = make_deps(_reply_script(seeded.provider_message_id), mail_provider=slow)
    paused = await run_graph(graph, seeded.input, seeded, deps)
    approve = await decide(sessionmaker, owner, paused, mail)
    slow.hang = True

    blocked = await resume(graph, approve, seeded, deps)

    assert blocked["__interrupt__"][0].value == {
        "type": "pause",
        "reason": "preflight_unavailable",
    }
    assert await count(sessionmaker, Action) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("paused", "preflight_unavailable")

    # The provider is back: Retry re-runs preflight with the same approval and sends once.
    slow.hang = False
    done = await resume(graph, {"decision": "retry"}, seeded, deps)

    assert [r["status"] for r in done["action_results"]] == ["succeeded"]
    assert await count(sessionmaker, MockOutbox) == 1


async def test_persist_proposal_replay_is_idempotent(
    make_deps: DepsFactory, seeded: SeededRun, sessionmaker: Session, mail: MockMailProvider
) -> None:
    """A crash after the write but before the checkpoint re-runs the node with the same state."""
    thread = await mail.get_thread(seeded.provider_thread_id)
    rfc_id = thread.messages[0].rfc_message_id
    assert rfc_id is not None
    state: dict[str, Any] = {
        **seeded.input,
        "source_version": thread.version,
        "preference_version": 0,
        "step_count": 2,
        "draft_kind": "send_reply",
        "draft": {
            "provider_thread_id": seeded.provider_thread_id,
            "in_reply_to": rfc_id,
            "references": [rfc_id],
            "to": ["alice@example.com"],
            "cc": [],
            "subject": "Re: Meeting next week?",
            "body_text": "Tuesday works.",
            "evidence_refs": [f"message:{seeded.provider_message_id}"],
        },
    }
    runtime = Runtime(context=make_deps())

    first = await persist_proposal(state, runtime)  # type: ignore[arg-type]
    second = await persist_proposal(state, runtime)  # type: ignore[arg-type]

    assert first == second
    assert await count(sessionmaker, Proposal) == 1
    assert await count(sessionmaker, ReviewItem) == 1
    assert len(await _event_payloads(sessionmaker, "approval_required")) == 1


async def test_persist_proposal_records_cited_availability_check_time(
    make_deps: DepsFactory, seeded: SeededRun, sessionmaker: Session
) -> None:
    """The review item carries when each cited availability finding was read (UI shows it)."""
    cited = {
        "key": "availability:2026-11-24:2026-11-24",
        "kind": "availability",
        "busy": [],
        "candidate_slots": [],
        "time_zone": "Europe/Paris",
        "retrieved_at": "2026-11-23T09:00:00+00:00",
    }
    uncited = {**cited, "key": "availability:2026-11-25:2026-11-25"}
    state: dict[str, Any] = {
        **seeded.input,
        "source_version": "sv",
        "preference_version": 0,
        "step_count": 3,
        "read_findings": [cited, uncited],
        "draft_kind": "create_event",
        "draft": {
            "calendar_id": "primary",
            "title": "Sync",
            "start": "2026-11-24T13:00:00+00:00",
            "end": "2026-11-24T13:30:00+00:00",
            "time_zone": "Europe/Paris",
            "attendees": ["alice@example.com"],
            "location": None,
            "send_updates": "all",
            "evidence_refs": [cited["key"], f"message:{seeded.provider_message_id}"],
        },
    }

    await persist_proposal(state, Runtime(context=make_deps()))  # type: ignore[arg-type]

    async with sessionmaker() as session:
        item = (await session.execute(select(ReviewItem))).scalar_one()
        proposal = (await session.execute(select(Proposal))).scalar_one()
        revision = (await session.execute(select(ProposalRevision))).scalar_one()
    assert item.payload == {
        "proposal_kind": "create_event",
        "revision": 1,
        "availability": [
            {
                "key": cited["key"],
                "time_zone": "Europe/Paris",
                "retrieved_at": "2026-11-23T09:00:00+00:00",
            }
        ],
    }
    assert proposal.kind == "create_event"
    assert "availability" not in revision.payload  # the hashed payload is unchanged


async def test_unknown_decision_finishes(
    graph: Graph, make_deps: DepsFactory, seeded: SeededRun, sessionmaker: Session
) -> None:
    deps = make_deps(_reply_script(seeded.provider_message_id))
    await run_graph(graph, seeded.input, seeded, deps)

    done = await graph.ainvoke(
        Command(resume={"decision": "maybe"}), seeded.config, context=deps, durability="sync"
    )

    assert "__interrupt__" not in done
    assert await count(sessionmaker, MockOutbox) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("failed", "unknown_decision")
    assert len(await _event_payloads(sessionmaker, "run_failed")) == 1
