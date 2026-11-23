"""Scripted mode for the running app: fixture scripts, step-indexed plans, the date token."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.graphs.schemas import NextStep, TriageDecision
from app.llm.client import LLMCall
from app.llm.factory import make_llm_client
from app.llm.scripted import ScriptedLLMClient, fixture_scripts, next_workday


def _call(purpose: str, model: Any, key: str = "m1", step: int | None = None) -> LLMCall[Any]:
    return LLMCall(
        purpose=purpose,  # type: ignore[arg-type]
        system="s",
        cached_context="",
        user_content="u",
        output_model=model,
        script_key=key,
        script_step=step,
    )


def _plan(action: str, **extra: Any) -> dict[str, Any]:
    return {"action": action, "reason": action, **extra}


def _write(dir_: Path, fixture_id: str, mid: str, script: dict[str, Any]) -> None:
    message = {
        "provider_message_id": mid,
        "provider_thread_id": f"thr-{fixture_id}",
        "rfc_message_id": f"<{mid}@example.net>",
        "sender": "alice@example.org",
        "to": ["owner@example.com"],
        "cc": [],
        "subject": "Hi",
        "body_text": "Hello",
        "body_html": None,
        "received_at": "2026-11-19T08:30:00+00:00",
        "headers": {},
    }
    fixture = {
        "id": fixture_id,
        "thread": [message],
        "label": {"classification": "respond"},
        "script": script,
    }
    (dir_ / f"{fixture_id}.json").write_text(json.dumps(fixture), encoding="utf-8")


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 11, 23), date(2026, 11, 25)),  # Monday -> Wednesday
        (date(2026, 11, 26), date(2026, 11, 30)),  # Thursday -> Saturday -> next Monday
        (date(2026, 11, 28), date(2026, 11, 30)),  # Saturday -> Monday
    ],
)
def test_next_workday_is_two_or_more_days_ahead_on_a_weekday(today: date, expected: date) -> None:
    assert next_workday(today) == expected


async def test_by_step_plan_output_follows_step_count_not_call_order() -> None:
    client = ScriptedLLMClient(
        {"m1": {"plan": [_plan("draft_reply"), _plan("finish")]}}, by_step=True
    )
    # A restarted worker asks for step 1 first: it gets step 1's output, not the first one.
    later = await client.structured(_call("plan", NextStep, step=1))
    first = await client.structured(_call("plan", NextStep, step=0))
    beyond = await client.structured(_call("plan", NextStep, step=7))
    assert (later.output.action, first.output.action, beyond.output.action) == (
        "finish",
        "draft_reply",
        "finish",  # past the end: the last output repeats
    )


async def test_fixture_scripts_key_by_trigger_message_and_fill_workday(tmp_path: Path) -> None:
    triage = {
        "classification": "respond",
        "urgency": "normal",
        "reason": "r",
        "evidence_refs": ["message:fx-m1"],
        "needs_clarification": False,
    }
    plan = _plan("get_freebusy", freebusy={"start_date": "{{workday}}", "end_date": "{{workday}}"})
    _write(tmp_path, "fx", "fx-m1", {"triage": [triage], "plan": [plan]})
    _write(tmp_path, "empty", "empty-m1", {})

    scripts = fixture_scripts(tmp_path, today=date(2026, 11, 23))

    assert set(scripts) == {"fx-m1"}
    assert scripts["fx-m1"]["plan"][0]["freebusy"] == {
        "start_date": "2026-11-25",
        "end_date": "2026-11-25",
    }
    client = ScriptedLLMClient(scripts, by_step=True)
    out = await client.structured(_call("triage", TriageDecision, key="fx-m1"))
    assert out.output.classification == "respond"


def test_factory_scripted_mode_loads_fixture_scripts(base_env: None, tmp_path: Path) -> None:
    _write(tmp_path, "fx", "fx-m1", {"plan": [_plan("finish")]})
    settings = Settings(_env_file=None, llm_provider="scripted", fixtures_dir=tmp_path)  # type: ignore[call-arg]
    client = make_llm_client(settings, None)  # type: ignore[arg-type]
    assert isinstance(client, ScriptedLLMClient)
    assert client.keys() == {"fx-m1"}
