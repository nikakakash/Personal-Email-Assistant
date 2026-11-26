"""Chooses the LLM client implementation from settings."""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.llm.anthropic_client import AnthropicLLMClient, build_anthropic
from app.llm.budget import BudgetGuard
from app.llm.client import LLMClient
from app.llm.pricing import PRICES
from app.llm.scripted import ScriptedLLMClient, fixture_scripts


def make_llm_client(
    settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]
) -> LLMClient:
    if settings.llm_provider == "scripted":
        today = datetime.now(UTC).date()
        return ScriptedLLMClient(fixture_scripts(settings.fixtures_dir, today), by_step=True)
    for name in ("model_triage", "model_plan", "model_draft", "model_contextualize"):
        if getattr(settings, name) not in PRICES:
            raise ValueError(f"{name.upper()} is not a priced model: {getattr(settings, name)!r}")
    budget = BudgetGuard(sessionmaker, settings.daily_budget_usd)
    return AnthropicLLMClient(build_anthropic(settings), settings, budget)
