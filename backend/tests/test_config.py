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


def test_api_bind_defaults_to_loopback_and_is_configurable(
    base_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import Settings

    default = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (default.api_host, default.api_port) == ("127.0.0.1", 8000)
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "9000")
    container = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (container.api_host, container.api_port) == ("0.0.0.0", 9000)


def test_rag_is_off_by_default_with_local_embeddings(base_env: None) -> None:
    s = make()
    assert s.rag_enabled is False
    assert s.embeddings_provider == "fastembed"
    assert s.embed_model == "bge-small-en-v1.5"
    assert (s.retrieval_mode, s.retrieval_top_k) == ("hybrid", 5)
    assert s.rag_rerank is False and s.rag_contextualize is False
    assert s.voyage_api_key is None


def test_outbound_enabled_rejects_placeholder_voyage_key(
    base_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTBOUND_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("VOYAGE_API_KEY", "replace-me")
    with pytest.raises(ValidationError, match="placeholder"):
        make()
