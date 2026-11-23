"""Retrying a run: `POST /workflows/{id}/retry` (spec §12: "retry only classified recoverable
work; no blind resend").

Which runs can be retried (`is_retryable`):

- `paused`, for any reason: the graph waits at `pause_interrupt` (budget, step budget,
  outbound disabled, a provider outage at preflight), see `app/graphs/nodes/pause.py`.
- `failed` with `llm_transient` or `internal_error`: an exception stopped the graph mid-step,
  so the checkpoint still has the step that failed; the worker continues it with
  `ainvoke(None)`. `execute_action` is ledger first, so a retried send is reconciled, never
  repeated.

Never retried: `outcome_unknown` (the provider may have accepted the write; the owner must
check), `approval_invalid`, a model refusal or malformed output, and any final success or
cancel.

The API locks the run row, checks it is retryable, sets it to `running` (so a second click
gets a 409 instead of a second job, and the graph's own writes, which skip final runs, work
again) and enqueues one `retry_workflow` job, all in one transaction. The job's payload keeps
the previous status, so the worker can put it back if there turns out to be nothing to
continue. The dedup key includes the request id, so a repeated request returns its job.
"""

import uuid
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.decisions import DecisionConflict
from app.actions.reviews import ReviewNotFound
from app.db.models import AuditEvent, Job, WorkflowRun
from app.graphs.events import emit
from app.jobs.queue import enqueue

RETRYABLE_FAILURES = ("llm_transient", "internal_error")


def is_retryable(status: str, error_code: str | None) -> bool:
    return status == "paused" or (status == "failed" and error_code in RETRYABLE_FAILURES)


class RetryOutcome(BaseModel):
    status: Literal["accepted"] = "accepted"
    workflow_id: uuid.UUID
    job_id: uuid.UUID


def _dedup_key(workflow_id: uuid.UUID, request_id: str) -> str:
    return f"retry:{workflow_id}:{request_id}"


async def retry_workflow(
    session: AsyncSession, workflow_id: uuid.UUID, *, request_id: str
) -> RetryOutcome:
    """Queue a retry of a recoverable run; commits (or rolls back and raises)."""
    try:
        outcome = await _retry(session, workflow_id, request_id)
        await session.commit()
        return outcome
    except BaseException:
        await session.rollback()
        raise


async def _retry(session: AsyncSession, workflow_id: uuid.UUID, request_id: str) -> RetryOutcome:
    # FOR NO KEY UPDATE, like cancel: serializes with other run updates without blocking
    # inserts of rows that reference the run.
    run = await session.get(
        WorkflowRun, workflow_id, with_for_update={"key_share": True}, populate_existing=True
    )
    if run is None:
        raise ReviewNotFound(str(workflow_id))
    stored = await session.scalar(
        select(Job).where(Job.dedup_key == _dedup_key(workflow_id, request_id))
    )
    if stored is not None:
        return RetryOutcome(workflow_id=workflow_id, job_id=stored.id)
    if not is_retryable(run.status, run.error_code):
        raise DecisionConflict("not_retryable")

    previous = {"previous_status": run.status, "previous_error_code": run.error_code}
    run.status, run.error_code = "running", None
    job = await enqueue(
        session,
        "retry_workflow",
        _dedup_key(workflow_id, request_id),
        {"workflow_id": str(workflow_id), **previous},
        serial_key=run.provider_thread_id,
    )
    detail = {"request_id": request_id, **previous}
    session.add(AuditEvent(workflow_id=workflow_id, event_type="run_retried", detail=detail))
    await emit(session, workflow_id, "run_retried", detail)
    await session.flush()
    return RetryOutcome(workflow_id=workflow_id, job_id=job.id)
