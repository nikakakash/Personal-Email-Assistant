"""The review queue and the owner's answers to questions and notifications."""

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.actions.reviews import ReviewOutcome, answer_question, decide_notification
from app.api.deps import OWNER_ONLY, SessionDep, SessionmakerDep
from app.api.pagination import DEFAULT_LIMIT, Limit, Page
from app.api.views import ReviewItemOut, list_reviews

router = APIRouter(prefix="/api/v1", tags=["reviews"], dependencies=OWNER_ONLY)

ReviewStatus = Literal["pending", "stale", "resolved", "cancelled"]
ReviewKind = Literal["notification", "question", "proposal"]
RequestId = Annotated[str, StringConstraints(min_length=1, max_length=200)]


class AnswerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    request_id: RequestId
    interrupt_id: RequestId


class NotificationDecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["dismiss", "respond"]
    request_id: RequestId
    interrupt_id: RequestId


@router.get("/reviews")
async def reviews(
    session: SessionDep,
    status: ReviewStatus = "pending",
    kind: ReviewKind | None = None,
    cursor: Annotated[str | None, Query(max_length=500)] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> Page[ReviewItemOut]:
    return await list_reviews(session, status=status, kind=kind, cursor=cursor, limit=limit)


@router.post("/questions/{review_item_id}/answer", status_code=202)
async def answer(
    review_item_id: uuid.UUID, body: AnswerIn, maker: SessionmakerDep
) -> ReviewOutcome:
    async with maker() as session:
        return await answer_question(
            session,
            review_item_id,
            answer=body.answer,
            request_id=body.request_id,
            interrupt_id=body.interrupt_id,
        )


@router.post("/notifications/{review_item_id}/decision", status_code=202)
async def notification_decision(
    review_item_id: uuid.UUID, body: NotificationDecisionIn, maker: SessionmakerDep
) -> ReviewOutcome:
    async with maker() as session:
        return await decide_notification(
            session,
            review_item_id,
            decision=body.decision,
            request_id=body.request_id,
            interrupt_id=body.interrupt_id,
        )
