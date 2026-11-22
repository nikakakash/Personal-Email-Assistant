"""Workflow list (Failed / Completed tabs), workflow detail and cancel."""

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.actions.reviews import CancelOutcome, cancel_workflow
from app.api.deps import OWNER_ONLY, SessionDep, SessionmakerDep
from app.api.pagination import DEFAULT_LIMIT, Limit, Page
from app.api.views import WorkflowOut, WorkflowSummaryOut, list_workflows, workflow_detail

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"], dependencies=OWNER_ONLY)

WorkflowStatus = Literal[
    "running",
    "waiting_review",
    "paused",
    "done_no_action",
    "done_with_actions",
    "failed",
    "cancelled",
]


class CancelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]


@router.get("")
async def workflows(
    session: SessionDep,
    status: Annotated[list[WorkflowStatus] | None, Query()] = None,
    cursor: Annotated[str | None, Query(max_length=500)] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> Page[WorkflowSummaryOut]:
    """Runs, newest first. `status` may repeat: `?status=failed&status=cancelled`."""
    return await list_workflows(session, statuses=list(status or []), cursor=cursor, limit=limit)


@router.get("/{workflow_id}")
async def workflow(workflow_id: uuid.UUID, session: SessionDep) -> WorkflowOut:
    detail = await workflow_detail(session, workflow_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="not_found")
    return detail


@router.post("/{workflow_id}/cancel", status_code=202)
async def cancel(workflow_id: uuid.UUID, body: CancelIn, maker: SessionmakerDep) -> CancelOutcome:
    async with maker() as session:
        return await cancel_workflow(session, workflow_id, request_id=body.request_id)
