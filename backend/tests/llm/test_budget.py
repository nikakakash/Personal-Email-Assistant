"""Tests for BudgetGuard and cost_of."""

import asyncio
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.budget import BudgetGuard
from app.llm.client import BudgetExceeded, Usage
from app.llm.pricing import cost_of


def test_cost_of_uses_all_four_rates_quantized() -> None:
    usage = Usage(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert cost_of("claude-opus-5-5", usage) == Decimal("29.200000")
    assert cost_of("claude-haiku-5-5", Usage(1, 0, 0, 0)) == Decimal("0.000000")
    assert cost_of("claude-sonnet-5-5", Usage(0, 1, 0, 0)) == Decimal("0.000010")


async def test_reserve_settle_and_total(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    guard = BudgetGuard(sessionmaker, Decimal("1.00"))
    rid = await guard.reserve("plan", "claude-opus-5-5", Decimal("0.40"), None)
    assert await guard.today_total() == Decimal("0.40")
    await guard.settle(rid, Usage(1000, 100, 0, 0), "claude-opus-5-5")
    assert await guard.today_total() == cost_of("claude-opus-5-5", Usage(1000, 100, 0, 0))


async def test_reserved_counts_against_limit(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    guard = BudgetGuard(sessionmaker, Decimal("1.00"))
    await guard.reserve("plan", "claude-opus-5-5", Decimal("0.60"), None)
    try:
        await guard.reserve("plan", "claude-opus-5-5", Decimal("0.50"), None)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("expected BudgetExceeded")
    await guard.reserve("plan", "claude-opus-5-5", Decimal("0.40"), None)  # exactly at limit


async def test_concurrent_reserves_cannot_overspend(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    guard = BudgetGuard(sessionmaker, Decimal("1.00"))
    results = await asyncio.gather(
        *(guard.reserve("plan", "claude-opus-5-5", Decimal("0.30"), None) for _ in range(8)),
        return_exceptions=True,
    )
    ok = [r for r in results if not isinstance(r, BaseException)]
    assert len(ok) == 3
    assert all(isinstance(r, BudgetExceeded) for r in results if isinstance(r, BaseException))
    assert await guard.today_total() == Decimal("0.90")
