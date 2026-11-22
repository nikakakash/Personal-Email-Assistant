"""Daily model-spend guard backed by the `llm_usage` table.

`reserve` runs under a per-day Postgres advisory lock so concurrent callers cannot both pass
the check. A reservation that is never settled keeps counting against the day (conservative).
Sessions are short and committed; no transaction is held open across a model call.
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import LlmUsage
from app.llm.client import BudgetExceeded, Purpose, Usage
from app.llm.pricing import cost_of


def _day_start() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


class BudgetGuard:
    def __init__(
        self, sessionmaker: async_sessionmaker[AsyncSession], daily_limit: Decimal
    ) -> None:
        self._sessionmaker = sessionmaker
        self._limit = daily_limit

    async def _total(self, session: AsyncSession, since: datetime) -> Decimal:
        stmt = select(func.coalesce(func.sum(LlmUsage.cost_usd), 0)).where(
            LlmUsage.created_at >= since
        )
        return Decimal((await session.execute(stmt)).scalar_one())

    async def today_total(self) -> Decimal:
        async with self._sessionmaker() as session:
            return await self._total(session, _day_start())

    async def reserve(
        self, purpose: Purpose, model: str, est_usd: Decimal, workflow_id: UUID | None
    ) -> UUID:
        day = _day_start()
        async with self._sessionmaker() as session, session.begin():
            await session.execute(
                text("select pg_advisory_xact_lock(hashtext('llm_budget:' || :day))"),
                {"day": day.date().isoformat()},
            )
            if await self._total(session, day) + est_usd > self._limit:
                raise BudgetExceeded(f"daily budget of {self._limit} USD would be exceeded")
            row = LlmUsage(
                workflow_id=workflow_id,
                purpose=purpose,
                model=model,
                cost_usd=est_usd,
                status="reserved",
            )
            session.add(row)
            await session.flush()
            return row.id

    async def settle_estimate(self, reservation_id: UUID) -> None:
        """Close a reservation keeping its estimated cost (outcome unknown: fail closed)."""
        async with self._sessionmaker() as session, session.begin():
            row = await session.get(LlmUsage, reservation_id)
            if row is None:
                raise LookupError(f"unknown reservation {reservation_id}")
            row.status = "settled"

    async def settle(self, reservation_id: UUID, usage: Usage, model: str) -> None:
        async with self._sessionmaker() as session, session.begin():
            row = await session.get(LlmUsage, reservation_id)
            if row is None:
                raise LookupError(f"unknown reservation {reservation_id}")
            row.model = model
            row.input_tokens = usage.input_tokens
            row.output_tokens = usage.output_tokens
            row.cache_read_tokens = usage.cache_read_tokens
            row.cache_write_tokens = usage.cache_write_tokens
            row.cost_usd = cost_of(model, usage)
            row.status = "settled"
