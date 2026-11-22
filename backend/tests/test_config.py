"""Settings: safe defaults, placeholder rejection, alias parsing."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def make() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_outbound_enabled_rejects_placeholder_key(
    base_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTBOUND_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("SESSION_SECRET", "replace-me")
    with pytest.raises(ValidationError, match="placeholder"):
        make()


def test_outbound_enabled_rejects_placeholder_api_key(
    base_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTBOUND_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "replace-me")
    with pytest.raises(ValidationError, match="placeholder"):
        make()


def test_placeholder_allowed_while_outbound_disabled(
    base_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SESSION_SECRET", "replace-me")
    assert make().outbound_execution_enabled is False


def test_defaults_are_safe(base_env: None) -> None:
    s = make()
    assert s.outbound_execution_enabled is False
    assert s.model_triage == "claude-opus-5-5" and s.effort_triage == "low"
    assert s.owner_aliases == []


def test_owner_aliases_parse_comma_separated(
    base_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OWNER_ALIASES", "a@x.com, b@x.com")
    assert make().owner_aliases == ["a@x.com", "b@x.com"]
