"""Response subgraph, driven through the compiled parent graph: reads, drafts, interrupts,
decisions, preflight and execution."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langgraph.types import Command
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.executor import operation_id_for
from app.db.models import (
    Action,
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
from app.graphs.nodes.ask import question_item_id
from app.graphs.nodes.execute import PROPOSAL_TTL, ActionInFlight
from app.llm.client import BudgetExceeded, LLMCall, LLMResult, T
from app.llm.scripted import ScriptedLLMClient
from app.providers.mock_mail import MockMailProvider
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
    async with sessionmaker() as s:  # a follow-up message arrives after the approval
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
    deps = make_deps(_reply_script(seeded.provider_message_id), outbound=False)
    paused = await run_graph(graph, seeded.input, seeded, deps)

    done = await resume(graph, await decide(sessionmaker, owner, paused, mail), seeded, deps)

    assert done["preflight"] == {"result": "blocked", "reason": "outbound_disabled"}
    assert await count(sessionmaker, MockOutbox) == 0
    assert await count(sessionmaker, Action) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("paused", "outbound_disabled")


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

    done = await run_graph(graph, seeded.input, seeded, make_deps(llm=llm))

    assert "__interrupt__" not in done
    assert done["next_action"] == "budget_stop"
    assert await count(sessionmaker, Proposal) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("paused", "budget")


async def test_execute_in_flight_raises(
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
    async with sessionmaker() as s:  # another caller has just claimed the operation
        s.add(
            Action(
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
