"""LLM call contract: the request/result dataclasses, the client protocol and the error types."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Purpose = Literal["triage", "plan", "draft_reply", "draft_meeting", "embed", "contextualize"]


@dataclass
class LLMCall[T: BaseModel]:
    purpose: Purpose
    system: str
    cached_context: str
    user_content: str
    output_model: type[T]
    workflow_id: UUID | None = None
    script_key: str | None = None
    # The planner's step number; the scripted client can pick outputs by it (restart-safe).
    script_step: int | None = None
    max_tokens: int = 8000


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMResult[T: BaseModel]:
    output: T
    usage: Usage
    model: str
    cost_usd: Decimal


class LLMClient(Protocol):
    async def structured(self, call: LLMCall[T]) -> LLMResult[T]: ...


class LLMError(Exception):
    """Base class of all adapter errors. Messages are redacted: never prompt content."""


class LLMRefusal(LLMError):
    def __init__(self, category: str | None) -> None:
        super().__init__(f"model refused the request (category={category})")
        self.category = category


class LLMMalformedOutput(LLMError):
    """The model's reply did not match the output schema, even after one repair call."""


class BudgetExceeded(LLMError):
    """The daily model-spend ceiling would be exceeded."""


class LLMTransientError(LLMError):
    """Rate limit, 5xx or network failure; the caller may retry."""
