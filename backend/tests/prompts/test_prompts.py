"""Tests for the versioned prompt loader."""

import pytest

from app.prompts import PROMPT_VERSIONS, load_prompt

NAMES = ["triage", "plan", "draft_reply", "draft_meeting"]


def test_prompts_have_versions_and_injection_rule() -> None:
    assert set(PROMPT_VERSIONS) == set(NAMES)
    for name in NAMES:
        p = load_prompt(name)
        assert p.version == f"{name}_v1" == PROMPT_VERSIONS[name]
        assert p.name == name
        assert "untrusted" in p.system.lower()
        assert "message:" in p.system and "chain-of-thought" in p.system
        assert not p.system.startswith("---")


def test_draft_prompts_encode_time_rules() -> None:
    for name in ("draft_reply", "draft_meeting"):
        s = load_prompt(name).system.lower()
        assert "never guess" in s and "time zone" in s
        assert "proposed" in s and "confirmed" in s
        assert "availability" in s


def test_plan_prompt_describes_actions() -> None:
    s = load_prompt("plan").system
    for action in (
        "read_thread",
        "get_freebusy",
        "ask_owner",
        "draft_reply",
        "draft_meeting",
        "finish",
    ):
        assert action in s


def test_load_prompt_caches_and_rejects_unknown() -> None:
    assert load_prompt("triage") is load_prompt("triage")
    with pytest.raises(KeyError):
        load_prompt("nope")
