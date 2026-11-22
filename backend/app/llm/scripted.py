"""Scripted LLM client for tests and offline demos: replays canned outputs, zero cost."""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from app.llm.client import LLMCall, LLMResult, Purpose, T, Usage


class ScriptedLLMClient:
    def __init__(self, scripts: Mapping[str, Mapping[Purpose, Sequence[dict[str, Any]]]]) -> None:
        self._queues: dict[tuple[str, Purpose], list[dict[str, Any]]] = {
            (key, purpose): list(outputs)
            for key, by_purpose in scripts.items()
            for purpose, outputs in by_purpose.items()
        }

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        queue = self._queues.get((call.script_key or "", call.purpose))
        if not queue:
            raise LookupError(
                f"no scripted output left for script_key={call.script_key!r}, "
                f"purpose={call.purpose!r}"
            )
        output = call.output_model.model_validate(queue.pop(0))
        return LLMResult(output=output, usage=Usage(), model="scripted", cost_usd=Decimal("0"))
