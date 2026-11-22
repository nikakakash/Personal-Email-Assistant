"""Tests for ScriptedLLMClient and the client factory."""

from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.llm.anthropic_client import AnthropicLLMClient
from app.llm.client import LLMCall
from app.llm.factory import make_llm_client
from app.llm.scripted import ScriptedLLMClient


class Out(BaseModel):
    n: int


def call(key: str | None = "k", purpose: str = "triage") -> LLMCall[Out]:
    return LLMCall(
        purpose=purpose,  # type: ignore[arg-type]
        system="s",
        cached_context="c",
        user_content="u",
        output_model=Out,
        workflow_id=None,
        script_key=key,
    )


async def test_scripted_pops_in_order_and_raises_when_exhausted() -> None:
    client = ScriptedLLMClient({"k": {"triage": [{"n": 1}, {"n": 2}], "plan": [{"n": 9}]}})
    first = await client.structured(call())
    second = await client.structured(call())
    assert (first.output.n, second.output.n) == (1, 2)
    assert first.cost_usd == Decimal("0")
    assert (await client.structured(call(purpose="plan"))).output.n == 9
    with pytest.raises(LookupError, match="k.*triage"):
        await client.structured(call())


async def test_scripted_unknown_key_raises() -> None:
    with pytest.raises(LookupError, match="other"):
        await ScriptedLLMClient({}).structured(call(key="other"))


async def test_scripted_validates_output() -> None:
    client = ScriptedLLMClient({"k": {"triage": [{"n": "x"}]}})
    with pytest.raises(ValidationError):
        await client.structured(call())


def test_factory_selects_provider(
    base_env: None,
    monkeypatch: pytest.MonkeyPatch,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "scripted")
    scripted = make_llm_client(Settings(_env_file=None), sessionmaker)  # type: ignore[call-arg]
    assert isinstance(scripted, ScriptedLLMClient)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    real = make_llm_client(Settings(_env_file=None), sessionmaker)  # type: ignore[call-arg]
    assert isinstance(real, AnthropicLLMClient)
