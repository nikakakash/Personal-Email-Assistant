"""POST /workflows/{id}/retry: only classified recoverable runs, once per request id."""

from collections.abc import Awaitable, Callable

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job, WorkflowEvent, WorkflowRun

WorkflowFactory = Callable[[], Awaitable[WorkflowRun]]


async def _with_status(
    session: AsyncSession, factory: WorkflowFactory, status: str, error_code: str | None
) -> WorkflowRun:
    run = await factory()
    run.status, run.error_code = status, error_code
    await session.commit()
    return run


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        ("paused", "outbound_disabled"),
        ("paused", "budget"),
        ("paused", "step_budget"),
        ("paused", "preflight_unavailable"),
        ("failed", "llm_transient"),
        ("failed", "internal_error"),
    ],
)
async def test_retry_queues_one_job(
    authed: AsyncClient,
    session: AsyncSession,
    workflow_factory: WorkflowFactory,
    status: str,
    error_code: str,
) -> None:
    run = await _with_status(session, workflow_factory, status, error_code)
    url = f"/api/v1/workflows/{run.id}/retry"

    r = await authed.post(url, json={"request_id": "r-1"})

    assert r.status_code == 202
    out = r.json()
    assert out["status"] == "accepted" and out["workflow_id"] == str(run.id)
    job = await session.scalar(select(Job).where(Job.kind == "retry_workflow"))
    assert job is not None and str(job.id) == out["job_id"]
    assert job.payload == {
        "workflow_id": str(run.id),
        "previous_status": status,
        "previous_error_code": error_code,
    }
    assert job.serial_key == run.provider_thread_id
    await session.refresh(run)
    assert (run.status, run.error_code) == ("running", None)
    events = await session.scalars(
        select(WorkflowEvent.event_type).where(WorkflowEvent.workflow_id == run.id)
    )
    assert list(events) == ["run_retried"]

    # The same click again returns the same job; a second, different click conflicts.
    again = await authed.post(url, json={"request_id": "r-1"})
    assert again.status_code == 202 and again.json() == out
    other = await authed.post(url, json={"request_id": "r-2"})
    assert other.status_code == 409 and other.json()["reason"] == "not_retryable"


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        ("failed", "outcome_unknown"),  # the send may have happened: the owner must check
        ("failed", "approval_invalid"),
        ("failed", "refusal"),
        ("done_with_actions", None),
        ("done_no_action", None),
        ("cancelled", None),
        ("waiting_review", None),
        ("running", None),
    ],
)
async def test_retry_refuses_unrecoverable_runs(
    authed: AsyncClient,
    session: AsyncSession,
    workflow_factory: WorkflowFactory,
    status: str,
    error_code: str | None,
) -> None:
    run = await _with_status(session, workflow_factory, status, error_code)

    r = await authed.post(f"/api/v1/workflows/{run.id}/retry", json={"request_id": "r-1"})

    assert r.status_code == 409 and r.json()["reason"] == "not_retryable"
    assert await session.scalar(select(Job).where(Job.kind == "retry_workflow")) is None


async def test_workflow_views_say_whether_retry_is_offered(
    authed: AsyncClient, session: AsyncSession, workflow_factory: WorkflowFactory
) -> None:
    paused = await _with_status(session, workflow_factory, "paused", "outbound_disabled")
    unknown = await _with_status(session, workflow_factory, "failed", "outcome_unknown")

    detail = (await authed.get(f"/api/v1/workflows/{paused.id}")).json()
    assert detail["run"]["retryable"] is True
    assert (await authed.get(f"/api/v1/workflows/{unknown.id}")).json()["run"]["retryable"] is False
    listing = (await authed.get("/api/v1/workflows")).json()["items"]
    assert {i["id"]: i["retryable"] for i in listing} == {
        str(paused.id): True,
        str(unknown.id): False,
    }
