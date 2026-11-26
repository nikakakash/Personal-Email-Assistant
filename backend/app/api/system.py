"""Global pause flags and model usage."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Date, cast, func, select
from sqlalchemy.dialects.postgresql import insert

from app.api.deps import OWNER_ONLY, SessionDep, SessionmakerDep, SettingsDep
from app.db.models import AppFlags, LlmUsage

router = APIRouter(prefix="/api/v1", tags=["system"], dependencies=OWNER_ONLY)


class FlagsOut(BaseModel):
    ingestion_paused: bool
    outbound_paused: bool


class FlagsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingestion_paused: bool | None = None
    outbound_paused: bool | None = None


class DailyUsage(BaseModel):
    date: date
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: Decimal


class TodayUsage(BaseModel):
    cost_usd: Decimal
    budget_usd: Decimal
    remaining_usd: Decimal


class UsageOut(BaseModel):
    daily: list[DailyUsage]  # newest day first
    today: TodayUsage


@router.get("/flags")
async def get_flags(session: SessionDep) -> FlagsOut:
    flags = await session.get(AppFlags, 1)
    if flags is None:
        return FlagsOut(ingestion_paused=False, outbound_paused=False)
    return FlagsOut(ingestion_paused=flags.ingestion_paused, outbound_paused=flags.outbound_paused)


@router.patch("/flags")
async def patch_flags(body: FlagsIn, maker: SessionmakerDep) -> FlagsOut:
    """Upsert the singleton row (id 1); fields left out keep their value."""
    changes = body.model_dump(exclude_none=True)
    async with maker() as session, session.begin():
        stmt = insert(AppFlags).values(id=1, **changes)
        stmt = (
            stmt.on_conflict_do_update(index_elements=[AppFlags.id], set_=changes)
            if changes
            else stmt.on_conflict_do_nothing(index_elements=[AppFlags.id])
        )
        await session.execute(stmt)
        flags = await session.get_one(AppFlags, 1, populate_existing=True)
        return FlagsOut(
            ingestion_paused=flags.ingestion_paused, outbound_paused=flags.outbound_paused
        )


@router.get("/usage")
async def usage(
    session: SessionDep,
    settings: SettingsDep,
    days: Annotated[int, Query(ge=1, le=366)] = 30,
) -> UsageOut:
    """Token totals and cost per UTC day and model, plus today's spend against the budget."""
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    day = cast(func.timezone("UTC", LlmUsage.created_at), Date).label("day")
    rows = await session.execute(
        select(
            day,
            LlmUsage.model,
            func.count(),
            func.sum(LlmUsage.input_tokens),
            func.sum(LlmUsage.output_tokens),
            func.sum(LlmUsage.cache_read_tokens),
            func.sum(LlmUsage.cache_write_tokens),
            func.sum(LlmUsage.cost_usd),
        )
        .where(LlmUsage.created_at >= today - timedelta(days=days - 1))
        .group_by(day, LlmUsage.model)
        .order_by(day.desc(), LlmUsage.model)
    )
    daily = [
        DailyUsage(
            date=d,
            model=model,
            calls=calls,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
            cost_usd=cost,
        )
        for d, model, calls, inp, out, cr, cw, cost in rows
    ]
    spent = Decimal(
        await session.scalar(
            select(func.coalesce(func.sum(LlmUsage.cost_usd), 0)).where(
                LlmUsage.created_at >= today
            )
        )
        or 0
    )
    budget = settings.daily_budget_usd
    return UsageOut(
        daily=daily,
        today=TodayUsage(
            cost_usd=spent, budget_usd=budget, remaining_usd=max(budget - spent, Decimal(0))
        ),
    )
