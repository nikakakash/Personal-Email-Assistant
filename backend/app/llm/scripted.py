"""Scripted LLM client for tests, offline demos and the E2E stack: canned outputs, zero cost.

Scripts map `script_key` (the triggering message's provider id) to outputs per purpose. By
default each call pops the next output of its (key, purpose) queue.

`by_step=True` (the running app in `LLM_PROVIDER=scripted` mode) picks a call that carries a
`script_step` (the planner) by that step number instead, repeating the last output past the
end. The step number is checkpointed graph state, so a restarted worker that resumes a run asks
for the same step again and gets the same answer; a process-local queue would start over.

`fixture_scripts` reads the `script` of every fixture in a directory, keyed by the fixture's
last message (the one whose workflow the script drives). Strings may contain `{{workday}}`,
replaced by `next_workday(today)` in ISO form, so meeting dates stay in the future.
"""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from app.llm.client import LLMCall, LLMResult, Purpose, T, Usage
from app.providers.fixtures import load_fixture

WORKDAY_TOKEN = "{{workday}}"


def next_workday(today: date) -> date:
    """The first Monday-to-Friday date at least two days after `today`."""
    day = today + timedelta(days=2)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _fill(value: Any, workday: str) -> Any:
    if isinstance(value, str):
        return value.replace(WORKDAY_TOKEN, workday)
    if isinstance(value, list):
        return [_fill(v, workday) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, workday) for k, v in value.items()}
    return value


type Scripts = dict[str, dict[Purpose, list[dict[str, Any]]]]


def fixture_scripts(fixtures_dir: Path, today: date) -> Scripts:
    """`{last message id: {purpose: outputs}}` for every fixture that has a script."""
    workday = next_workday(today).isoformat()
    scripts: Scripts = {}
    for path in sorted(fixtures_dir.glob("*.json")):
        fixture = load_fixture(path.stem, fixtures_dir)
        if fixture.script:
            key = fixture.thread[-1].provider_message_id
            # Fixture keys are purpose names; an unknown one is simply never asked for.
            scripts[key] = cast(dict[Purpose, list[dict[str, Any]]], _fill(fixture.script, workday))
    return scripts


class ScriptedLLMClient:
    def __init__(
        self,
        scripts: Mapping[str, Mapping[Purpose, Sequence[dict[str, Any]]]],
        *,
        by_step: bool = False,
    ) -> None:
        self._by_step = by_step
        self._scripts: dict[tuple[str, Purpose], list[dict[str, Any]]] = {
            (key, purpose): list(outputs)
            for key, by_purpose in scripts.items()
            for purpose, outputs in by_purpose.items()
        }
        self._queues = {k: list(v) for k, v in self._scripts.items()}

    def keys(self) -> set[str]:
        """The script keys this client has outputs for."""
        return {key for key, _ in self._scripts}

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        key = (call.script_key or "", call.purpose)
        if self._by_step and call.script_step is not None and self._scripts.get(key):
            outputs = self._scripts[key]
            raw = outputs[min(call.script_step, len(outputs) - 1)]
        else:
            queue = self._queues.get(key)
            if not queue:
                raise LookupError(
                    f"no scripted output left for script_key={call.script_key!r}, "
                    f"purpose={call.purpose!r}"
                )
            raw = queue.pop(0)
        output = call.output_model.model_validate(raw)
        return LLMResult(output=output, usage=Usage(), model="scripted", cost_usd=Decimal("0"))
