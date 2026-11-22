"""Chooses the LLM client implementation from settings."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.llm.anthropic_client import AnthropicLLMClient, build_anthropic
from app.llm.budget import BudgetGuard
from app.llm.client import LLMClient
from app.llm.scripted import ScriptedLLMClient


def make_llm_client(
    settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]
) -> LLMClient:
    if settings.llm_provider == "scripted":
        return ScriptedLLMClient({})
    budget = BudgetGuard(sessionmaker, settings.daily_budget_usd)
    return AnthropicLLMClient(build_anthropic(settings), settings, budget)
