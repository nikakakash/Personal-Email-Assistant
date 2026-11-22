"""Tests for AnthropicLLMClient against a respx-mocked Messages API."""

import asyncio
import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import LlmUsage
from app.llm.anthropic_client import AnthropicLLMClient
from app.llm.budget import BudgetGuard
from app.llm.client import (
    BudgetExceeded,
    LLMCall,
    LLMError,
    LLMMalformedOutput,
    LLMRefusal,
    LLMTransientError,
    Usage,
)
from app.llm.pricing import cost_of

URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-5-5"


class Verdict(BaseModel):
    label: str
    score: int


def message(
    text: str, *, stop_reason: str = "end_turn", model: str = MODEL, **usage: Any
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": None}
        if stop_reason == "refusal"
        else None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 100),
            "output_tokens": usage.get("output_tokens", 20),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        },
    }


GOOD = json.dumps({"label": "ok", "score": 3})
BAD = json.dumps({"label": "ok"})


@pytest.fixture
def settings(base_env: None) -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def make_client(
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    make_sdk: Callable[[], AsyncAnthropic],
) -> Callable[..., AnthropicLLMClient]:
    def make(limit: str = "2.00") -> AnthropicLLMClient:
        sdk = make_sdk()
        return AnthropicLLMClient(sdk, settings, BudgetGuard(sessionmaker, Decimal(limit)))

    return make


def a_call() -> LLMCall[Verdict]:
    return LLMCall(
        purpose="triage",
        system="You triage email.",
        cached_context="Owner preferences.",
        user_content="Email body here",
        output_model=Verdict,
        workflow_id=None,
        script_key=None,
    )


async def test_request_shape(
    make_client: Callable[..., AnthropicLLMClient], api: respx.MockRouter
) -> None:
    route = api.post(URL).respond(200, json=message(GOOD))
    result = await make_client().structured(a_call())
    assert result.output == Verdict(label="ok", score=3)
    request = route.calls.last.request
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "cache_control" not in body["system"][0]
    assert body["system"][1]["cache_control"] == {"type": "ephemeral"}
    for forbidden in ("thinking", "temperature", "tool_choice", "top_p", "top_k"):
        assert forbidden not in body
    assert body["messages"] == [{"role": "user", "content": "Email body here"}]


async def test_refusal_raises(
    make_client: Callable[..., AnthropicLLMClient], api: respx.MockRouter
) -> None:
    api.post(URL).respond(200, json=message("", stop_reason="refusal"))
    with pytest.raises(LLMRefusal) as info:
        await make_client().structured(a_call())
    assert info.value.category == "cyber"


async def test_malformed_then_repair_succeeds(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
) -> None:
    route = api.post(URL).mock(
        side_effect=[
            httpx.Response(200, json=message(BAD)),
            httpx.Response(200, json=message(GOOD)),
        ]
    )
    result = await make_client().structured(a_call())
    assert result.output.score == 3
    assert route.call_count == 2
    repair = json.loads(route.calls[1].request.content)["messages"]
    assert [m["role"] for m in repair] == ["user", "assistant", "user"]
    assert repair[0]["content"] == "Email body here"
    assert repair[1]["content"] == BAD
    assert repair[2]["content"].startswith(
        "Your previous reply did not match the required JSON schema. Validation error:"
    )
    assert repair[2]["content"].endswith("Reply again with only valid output.")
    assert result.usage.input_tokens == 200  # both attempts are summed


async def test_malformed_twice_raises(
    make_client: Callable[..., AnthropicLLMClient], api: respx.MockRouter
) -> None:
    route = api.post(URL).respond(200, json=message(BAD))
    with pytest.raises(LLMMalformedOutput):
        await make_client().structured(a_call())
    assert route.call_count == 2


async def test_budget_blocks_when_exhausted(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
) -> None:
    route = api.post(URL).respond(200, json=message(GOOD))
    with pytest.raises(BudgetExceeded):
        await make_client(limit="0.01").structured(a_call())
    assert route.call_count == 0


async def test_usage_settled_with_actual_tokens(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
    session: AsyncSession,
) -> None:
    usage: dict[str, int] = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 300,
    }
    api.post(URL).respond(200, json=message(GOOD, **usage))
    result = await make_client().structured(a_call())
    expected = cost_of(MODEL, Usage(1000, 500, 2000, 300))
    assert result.cost_usd == expected
    rows = (await session.execute(select(LlmUsage))).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == "settled"
    assert rows[0].cost_usd == expected
    assert (rows[0].input_tokens, rows[0].output_tokens) == (1000, 500)
    assert (rows[0].cache_read_tokens, rows[0].cache_write_tokens) == (2000, 300)
    assert rows[0].purpose == "triage"


ESTIMATE = Decimal("0.160048")  # (50 chars // 4) * $4/MTok + 8000 * $20/MTok


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_transient_status_maps_and_settles_at_estimate(
    status: int,
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
    session: AsyncSession,
) -> None:
    api.post(URL).respond(status, json={"type": "error", "error": {"type": "x", "message": "m"}})
    with pytest.raises(LLMTransientError):
        await make_client().structured(a_call())
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert (row.status, row.cost_usd) == ("settled", ESTIMATE)


async def test_connection_error_is_transient_and_keeps_estimate(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
    session: AsyncSession,
) -> None:
    route = api.post(URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(LLMTransientError):
        await make_client().structured(a_call())
    assert route.called
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert (row.status, row.cost_usd) == ("settled", ESTIMATE)


async def test_timeout_keeps_estimate(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
    session: AsyncSession,
) -> None:
    route = api.post(URL).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(LLMTransientError):
        await make_client().structured(a_call())
    assert route.called
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert (row.status, row.cost_usd) == ("settled", ESTIMATE)


async def test_cancellation_keeps_estimate(
    make_client: Callable[..., AnthropicLLMClient],
    monkeypatch: pytest.MonkeyPatch,
    session: AsyncSession,
) -> None:
    client = make_client()
    monkeypatch.setattr(
        client._client.beta.messages, "create", AsyncMock(side_effect=asyncio.CancelledError())
    )
    with pytest.raises(asyncio.CancelledError):
        await client.structured(a_call())
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert (row.status, row.cost_usd) == ("settled", ESTIMATE)


async def test_client_error_is_not_transient_redacted_and_settles_zero(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
    session: AsyncSession,
) -> None:
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Email body here"},
    }
    api.post(URL).respond(400, json=body)
    with pytest.raises(LLMError) as info:
        await make_client().structured(a_call())
    assert not isinstance(info.value, LLMTransientError)
    assert "Email body here" not in str(info.value)
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert (row.status, row.cost_usd) == ("settled", Decimal("0"))


async def test_served_fallback_model_is_priced_and_recorded(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
    session: AsyncSession,
) -> None:
    api.post(URL).respond(200, json=message(GOOD, model="claude-opus-4-8"))
    result = await make_client().structured(a_call())
    expected = cost_of("claude-opus-4-8", Usage(100, 20, 0, 0))
    assert expected == Decimal("0.001000")  # 100 * 5 + 20 * 25 per MTok
    assert (result.model, result.cost_usd) == ("claude-opus-4-8", expected)
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert (row.model, row.cost_usd) == ("claude-opus-4-8", expected)


async def test_unknown_served_model_charged_at_highest_rate(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
    session: AsyncSession,
) -> None:
    api.post(URL).respond(
        200,
        json=message(
            GOOD,
            model="claude-mystery-9",
            input_tokens=1000,
            output_tokens=1000,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=1000,
        ),
    )
    result = await make_client().structured(a_call())
    # highest per-kind rates: 5 + 25 + 0.50 + 6.25 per MTok, 1000 tokens each
    assert result.cost_usd == Decimal("0.036750")
    row = (await session.execute(select(LlmUsage))).scalar_one()
    assert (row.model, row.cost_usd) == ("claude-mystery-9", Decimal("0.036750"))


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
async def test_truncation_raises_without_repair(
    stop_reason: str,
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
) -> None:
    route = api.post(URL).respond(200, json=message(BAD, stop_reason=stop_reason))
    with pytest.raises(LLMMalformedOutput, match="truncated"):
        await make_client().structured(a_call())
    assert route.call_count == 1


async def test_empty_text_raises_without_repair(
    make_client: Callable[..., AnthropicLLMClient],
    api: respx.MockRouter,
) -> None:
    route = api.post(URL).respond(200, json=message(""))
    with pytest.raises(LLMMalformedOutput, match="no text"):
        await make_client().structured(a_call())
    assert route.call_count == 1
