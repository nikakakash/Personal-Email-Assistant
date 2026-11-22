"""Anthropic structured-output client.

SDK route used: `client.beta.messages.create(...)` with `output_config.format` built from
`anthropic.transform_schema(output_model)`, then `model_validate_json` on the reply text.
The installed SDK (2.12.0) DOES accept `fallbacks` and `betas` on `beta.messages.parse`, but
`parse` validates inside the SDK call, so a schema failure raises without exposing the raw
reply or token usage. We need the raw text for the repair turn, the refusal check before any
parsing, and real usage to settle the budget, so we validate ourselves. The wire request is
identical to what `parse` would send.

Failure handling: refusal is checked first, then truncation (no repair), then empty text (no
repair); one repair call on invalid output; every attempt reserves and settles its own budget
row, priced at the model that actually served the reply. When the outcome is unknown (timeout,
connection error, 429, 5xx, cancellation) the reservation settles at its estimate; only a 4xx
rejection settles at zero. Prompt content is never logged or stored.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Any
from uuid import UUID

import anthropic
from anthropic import AsyncAnthropic
from pydantic import ValidationError

from app.config import Settings
from app.llm.budget import BudgetGuard
from app.llm.client import (
    LLMCall,
    LLMError,
    LLMMalformedOutput,
    LLMRefusal,
    LLMResult,
    LLMTransientError,
    Purpose,
    T,
    Usage,
)
from app.llm.pricing import cost_of, price_for

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MILLION = Decimal(1_000_000)
_QUANTUM = Decimal("0.000001")


def build_anthropic(settings: Settings) -> AsyncAnthropic:
    """Explicit key when configured, otherwise ambient credentials."""
    if settings.anthropic_api_key is not None:
        return AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(), max_retries=2, timeout=120
        )
    return AsyncAnthropic(max_retries=2, timeout=120)


class AnthropicLLMClient:
    def __init__(self, client: AsyncAnthropic, settings: Settings, budget: BudgetGuard) -> None:
        self._client = client
        self._settings = settings
        self._budget = budget

    def _model_and_effort(self, purpose: Purpose) -> tuple[str, str]:
        s = self._settings
        if purpose == "triage":
            return s.model_triage, s.effort_triage
        if purpose == "plan":
            return s.model_plan, s.effort_plan
        return s.model_draft, s.effort_draft

    @staticmethod
    def _estimate(call: LLMCall[T], model: str) -> Decimal:
        est_in = (len(call.system) + len(call.cached_context) + len(call.user_content)) // 4
        price = price_for(model)
        total = (est_in * price.input + call.max_tokens * price.output) / _MILLION
        return total.quantize(_QUANTUM)

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        model, effort = self._model_and_effort(call.purpose)
        messages: list[dict[str, Any]] = [{"role": "user", "content": call.user_content}]
        total = Usage()
        total_cost = Decimal("0")
        for _ in range(2):
            resp, usage = await self._attempt(call, model, effort, messages)
            served = str(resp.model)  # the model that actually answered (may be a fallback)
            total = Usage(
                total.input_tokens + usage.input_tokens,
                total.output_tokens + usage.output_tokens,
                total.cache_read_tokens + usage.cache_read_tokens,
                total.cache_write_tokens + usage.cache_write_tokens,
            )
            total_cost += cost_of(served, usage)
            if resp.stop_reason == "refusal":
                category = resp.stop_details.category if resp.stop_details else None
                raise LLMRefusal(category)
            if resp.stop_reason in ("max_tokens", "model_context_window_exceeded"):
                raise LLMMalformedOutput(f"reply truncated (stop_reason={resp.stop_reason})")
            raw_text = "".join(b.text for b in resp.content if b.type == "text")
            if not raw_text.strip():
                raise LLMMalformedOutput("model reply contained no text")
            try:
                output = call.output_model.model_validate_json(raw_text)
            except ValidationError as exc:
                logger.warning("llm output failed validation (purpose=%s)", call.purpose)
                # Repair turn: original user message, the bad reply, then a user turn.
                # The last turn is the user's, so this is not an assistant prefill.
                messages = [
                    messages[0],
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": "Your previous reply did not match the required JSON "
                        f"schema. Validation error: {exc}. Reply again with only valid output.",
                    },
                ]
                continue
            return LLMResult(output=output, usage=total, model=served, cost_usd=total_cost)
        raise LLMMalformedOutput(f"model output did not match {call.output_model.__name__}")

    async def _close_unknown(self, reservation: UUID) -> None:
        """Outcome unknown: keep the estimate. Shielded so cancellation cannot skip it."""
        try:
            await asyncio.shield(self._budget.settle_estimate(reservation))
        except Exception:
            logger.warning("could not settle reservation %s; it stays reserved", reservation)

    async def _close_rejected(self, reservation: UUID, model: str) -> None:
        """Request rejected before billing (4xx): release the reservation at zero."""
        try:
            await asyncio.shield(self._budget.settle(reservation, Usage(), model))
        except Exception:
            logger.warning("could not settle reservation %s; it stays reserved", reservation)

    async def _attempt(
        self, call: LLMCall[T], model: str, effort: str, messages: list[dict[str, Any]]
    ) -> tuple[Any, Usage]:
        # Plain dicts: the SDK's TypedDicts do not accept `effort` as a bare str.
        output_config: Any = {
            "effort": effort,
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(call.output_model),
            },
        }
        system: Any = [
            {"type": "text", "text": call.system},
            {"type": "text", "text": call.cached_context, "cache_control": {"type": "ephemeral"}},
        ]
        reservation = await self._budget.reserve(
            call.purpose, model, self._estimate(call, model), call.workflow_id
        )
        try:
            resp = await self._client.beta.messages.create(
                model=model,
                max_tokens=call.max_tokens,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config=output_config,
                system=system,
                messages=messages,  # type: ignore[arg-type]
            )
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                await self._close_unknown(reservation)
            else:
                await self._close_rejected(reservation, model)
            raise self._map_error(exc) from exc
        except anthropic.APIError as exc:  # timeout, connection, other: may have been billed
            await self._close_unknown(reservation)
            raise self._map_error(exc) from exc
        except BaseException:  # includes CancelledError
            await self._close_unknown(reservation)
            raise
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
        )
        await self._budget.settle(reservation, usage, str(resp.model))
        return resp, usage

    @staticmethod
    def _map_error(exc: anthropic.APIError) -> LLMError:
        """Typed mapping, most specific first. Messages carry no request/response content."""
        if isinstance(
            exc,
            anthropic.RateLimitError
            | anthropic.APITimeoutError
            | anthropic.APIConnectionError
            | anthropic.InternalServerError,
        ):
            return LLMTransientError(type(exc).__name__)
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code >= 500:
                return LLMTransientError(f"HTTP {exc.status_code}")
            return LLMError(f"Anthropic API error HTTP {exc.status_code}: {type(exc).__name__}")
        return LLMError(type(exc).__name__)
