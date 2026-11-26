"""Fixture loader: every shipped fixture parses and covers the required scenarios."""

from pathlib import Path

import pytest

from app.config import Settings
from app.providers.fixtures import list_fixtures, load_fixture


def test_all_fixtures_load_with_aware_dates_and_unique_ids() -> None:
    summaries = list_fixtures()
    assert len(summaries) >= 40
    ids: set[str] = set()
    for s in summaries:
        fx = load_fixture(s.id)
        assert fx.id == s.id and fx.thread
        assert fx.script == {}
        for m in fx.thread:
            assert m.received_at.utcoffset() is not None
            assert m.provider_message_id not in ids
            ids.add(m.provider_message_id)


def test_fixture_set_covers_required_scenarios() -> None:
    fixtures = [load_fixture(s.id) for s in list_fixtures()]
    assert any(len(f.thread) == 12 for f in fixtures)
    assert any(f.thread[-1].body_text is None and f.thread[-1].body_html for f in fixtures)
    assert any("List-Id" in f.thread[-1].headers for f in fixtures)
    assert any("Auto-Submitted" in f.thread[-1].headers for f in fixtures)
    assert any(f.label.needs_question for f in fixtures)
    assert {f.label.classification for f in fixtures} == {"respond", "notify", "ignore"}


def test_load_unknown_fixture_raises() -> None:
    with pytest.raises(KeyError):
        load_fixture("does-not-exist")


def test_fixtures_dir_setting_default_and_env_override(
    base_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default = Settings(_env_file=None).fixtures_dir  # type: ignore[call-arg]
    assert default.name == "fixtures" and default.parent.name == "evals"
    monkeypatch.setenv("FIXTURES_DIR", str(tmp_path))
    assert Settings(_env_file=None).fixtures_dir == tmp_path  # type: ignore[call-arg]
