"""Parent graph: triage routes, notification review, the response subgraph and finalize."""

import uuid

from langgraph.runtime import Runtime
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    AuditEvent,
    Owner,
    PreferenceVersion,
    ReviewItem,
    WorkflowEvent,
    WorkflowRun,
)
from app.graphs.nodes.notify import notification_item_id, persist_notification
from app.graphs.parent import GRAPH_VERSION, graph_thread_id
from app.llm.client import BudgetExceeded, LLMCall, LLMMalformedOutput, LLMRefusal, LLMResult, T
from tests.graphs.conftest import (
    DepsFactory,
    Graph,
    SeededRun,
    count,
    plan_out,
    resume,
    run_graph,
    run_row,
    triage_out,
)


async def _events(sessionmaker: async_sessionmaker[AsyncSession], seeded: SeededRun) -> list[str]:
    async with sessionmaker() as s:
        rows = await s.execute(
            select(WorkflowEvent.event_type, WorkflowEvent.payload)
            .where(WorkflowEvent.workflow_id == uuid.UUID(seeded.workflow_id))
            .order_by(WorkflowEvent.id)
        )
        return [f"{t}:{p['node']}" if t == "node_started" else t for t, p in rows.all()]


def test_graph_version_and_thread_id() -> None:
    assert GRAPH_VERSION == "wf1"
    assert graph_thread_id("acc", "m1") == "acc:m1:wf1"


async def test_ignore_route_finishes_no_action(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    mid = seeded.provider_message_id
    deps = make_deps({mid: {"triage": [triage_out("ignore", mid)]}})

    result = await run_graph(graph, seeded.input, seeded, deps)

    assert "__interrupt__" not in result
    assert result["status"] == "done_no_action"
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.step_count, run.error_code) == ("done_no_action", 0, None)
    # triage stores its outcome on the run for the review queue and workflow lists.
    assert (run.classification, run.triage_reason) == ("ignore", "Classified as ignore.")
    assert await count(sessionmaker, ReviewItem) == 0
    assert await count(sessionmaker, AuditEvent, AuditEvent.event_type == "run_completed") == 1
    assert await _events(sessionmaker, seeded) == [
        "node_started:load_context",
        "node_started:load_preferences",
        "node_started:triage",
        "node_started:record_triage",
        "node_started:finalize",
        "run_completed",
    ]


async def test_notify_then_dismiss(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    mid = seeded.provider_message_id
    deps = make_deps({mid: {"triage": [triage_out("notify", mid)]}})

    paused = await run_graph(graph, seeded.input, seeded, deps)

    (interrupt,) = paused["__interrupt__"]
    item_id = notification_item_id(seeded.workflow_id)
    assert interrupt.value == {"type": "notification", "review_item_id": str(item_id)}
    assert (await run_row(sessionmaker, seeded)).status == "waiting_review"
    async with sessionmaker() as s:
        item = await s.get_one(ReviewItem, item_id)
        assert (item.kind, item.status) == ("notification", "pending")
        assert item.payload["reason"] == "Classified as notify."

    done = await resume(graph, {"decision": "dismiss"}, seeded, deps)

    assert "__interrupt__" not in done
    assert (await run_row(sessionmaker, seeded)).status == "done_no_action"
    events = await _events(sessionmaker, seeded)
    assert events.count("approval_required") == 1
    assert "node_started:plan_next_step" not in events


async def test_notify_then_request_response_enters_subgraph(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    mid = seeded.provider_message_id
    deps = make_deps({mid: {"triage": [triage_out("notify", mid)], "plan": [plan_out("finish")]}})
    await run_graph(graph, seeded.input, seeded, deps)

    done = await resume(graph, {"decision": "respond"}, seeded, deps)

    assert "__interrupt__" not in done
    assert done["next_action"] == "finish"
    assert done["step_count"] == 1
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.step_count) == ("done_no_action", 1)
    events = await _events(sessionmaker, seeded)
    assert events[-4:] == [
        "node_started:plan_next_step",
        "node_started:finish",
        "node_started:finalize",
        "run_completed",
    ]


async def test_interrupt_node_rerun_is_side_effect_free(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """Resuming twice from the same checkpoint re-runs only the interrupt node: no new rows."""
    mid = seeded.provider_message_id
    deps = make_deps({mid: {"triage": [triage_out("notify", mid)]}})
    await run_graph(graph, seeded.input, seeded, deps)
    checkpoint = (await graph.aget_state(seeded.config)).config  # type: ignore[arg-type]
    assert (await graph.aget_state(seeded.config)).next == ("notification_interrupt",)  # type: ignore[arg-type]

    for _ in range(2):
        done = await graph.ainvoke(
            Command(resume={"decision": "dismiss"}),
            checkpoint,
            context=deps,
            durability="sync",
        )
        assert done["status"] == "done_no_action"
        assert await count(sessionmaker, ReviewItem) == 1

    events = await _events(sessionmaker, seeded)
    assert events.count("approval_required") == 1
    assert events.count("node_started:persist_notification") == 1
    assert events.count("run_completed") == 1  # finalize is a compare-and-set
    assert await count(sessionmaker, AuditEvent, AuditEvent.event_type == "run_completed") == 1


class RaisingLLM:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        raise self.exc


async def test_triage_refusal_goes_to_owner_review(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    paused = await run_graph(
        graph, seeded.input, seeded, make_deps(llm=RaisingLLM(LLMRefusal("cyber")))
    )

    assert paused["__interrupt__"][0].value["type"] == "notification"
    assert paused["classification"] == "notify"
    assert paused["decision_summary"].startswith("Needs review: ")


async def test_triage_needs_clarification_goes_to_owner_review(
    graph: Graph, make_deps: DepsFactory, seeded: SeededRun
) -> None:
    mid = seeded.provider_message_id
    triage = triage_out("respond", mid, needs_clarification=True, reason="Unclear request.")
    paused = await run_graph(graph, seeded.input, seeded, make_deps({mid: {"triage": [triage]}}))

    assert paused["__interrupt__"][0].value["type"] == "notification"
    assert paused["decision_summary"] == "Needs review: Unclear request."


async def test_load_preferences_uses_newest_active_revision(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
    owner: Owner,
) -> None:
    async with sessionmaker() as s:
        s.add_all(
            [
                PreferenceVersion(
                    owner_id=owner.id, namespace="writing", revision=1, value={"tone": "warm"}
                ),
                PreferenceVersion(
                    owner_id=owner.id, namespace="writing", revision=2, value={"tone": "brief"}
                ),
                PreferenceVersion(
                    owner_id=owner.id,
                    namespace="writing",
                    revision=3,
                    value={"tone": "rejected"},
                    status="inactive",
                ),
                PreferenceVersion(
                    owner_id=owner.id, namespace="triage", revision=1, value={"vip": []}
                ),
            ]
        )
        await s.commit()
    mid = seeded.provider_message_id
    deps = make_deps({mid: {"triage": [triage_out("ignore", mid)]}})

    done = await run_graph(graph, seeded.input, seeded, deps)

    assert done["preferences"] == {"writing": {"tone": "brief"}, "triage": {"vip": []}}
    assert done["preference_version"] == 2


async def test_triage_malformed_output_goes_to_owner_review(
    graph: Graph, make_deps: DepsFactory, seeded: SeededRun
) -> None:
    deps = make_deps(llm=RaisingLLM(LLMMalformedOutput("bad")))

    paused = await run_graph(graph, seeded.input, seeded, deps)

    assert paused["__interrupt__"][0].value["type"] == "notification"
    assert paused["classification"] == "notify"
    assert paused["decision_summary"].startswith("Needs review: ")


async def test_triage_budget_exceeded_pauses(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    deps = make_deps(llm=RaisingLLM(BudgetExceeded("daily budget reached")))

    paused = await run_graph(graph, seeded.input, seeded, deps)

    assert paused["__interrupt__"][0].value == {"type": "pause", "reason": "budget"}
    assert await count(sessionmaker, ReviewItem) == 0
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code) == ("paused", "budget")
    assert "run_paused" in await _events(sessionmaker, seeded)


async def test_retry_after_triage_budget_triages_again(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    mid = seeded.provider_message_id
    await run_graph(
        graph, seeded.input, seeded, make_deps(llm=RaisingLLM(BudgetExceeded("daily budget")))
    )

    done = await resume(
        graph,
        {"decision": "retry"},
        seeded,
        make_deps({mid: {"triage": [triage_out("ignore", mid)]}}),
    )

    assert "__interrupt__" not in done
    run = await run_row(sessionmaker, seeded)
    assert (run.status, run.error_code, run.classification) == ("done_no_action", None, "ignore")


async def test_persist_review_item_keeps_final_run_status(
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    async with sessionmaker() as s:
        run = await s.get_one(WorkflowRun, uuid.UUID(seeded.workflow_id))
        run.status = "cancelled"
        await s.commit()
    state = {**seeded.input, "classification": "notify", "decision_summary": "FYI."}

    await persist_notification(state, Runtime(context=make_deps()))  # type: ignore[arg-type]

    assert (await run_row(sessionmaker, seeded)).status == "cancelled"


async def test_automated_sender_is_never_answered(
    graph: Graph,
    make_deps: DepsFactory,
    seeded: SeededRun,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """Code, not the model, forbids replying to automated mail: respond becomes notify."""
    mid = seeded.provider_message_id
    deps = make_deps({mid: {"triage": [triage_out("respond", mid)]}})

    paused = await run_graph(graph, {**seeded.input, "automated": True}, seeded, deps)

    assert paused["__interrupt__"][0].value["type"] == "notification"
    assert paused["classification"] == "notify"
    assert paused["decision_summary"].startswith("Automated sender")
    run = await run_row(sessionmaker, seeded)
    assert run.classification == "notify"
    assert "node_started:plan_next_step" not in await _events(sessionmaker, seeded)
