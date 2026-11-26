"""Loader for the eval/demo email fixtures in `evals/fixtures/<id>.json`.

Each fixture is a thread, the expected triage label and (filled in by later tasks) the
scripted model outputs per purpose. The directory comes from `Settings.fixtures_dir`
(env `FIXTURES_DIR`); callers without a Settings object get the same resolution.
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from app.config import DEFAULT_FIXTURES_DIR
from app.providers.interfaces import NormalizedMessage


class FixtureLabel(BaseModel):
    classification: Literal["respond", "notify", "ignore"]
    needs_question: bool = False
    urgency: Literal["normal", "high"] = "normal"


class Fixture(BaseModel):
    id: str
    thread: list[NormalizedMessage]
    label: FixtureLabel
    script: dict[str, list[dict[str, Any]]] = {}


class FixtureSummary(BaseModel):
    id: str
    classification: str
    message_count: int
    subject: str


def _dir(fixtures_dir: Path | None) -> Path:
    if fixtures_dir is not None:
        return fixtures_dir
    env = os.environ.get("FIXTURES_DIR")
    return Path(env) if env else DEFAULT_FIXTURES_DIR


def load_fixture(fixture_id: str, fixtures_dir: Path | None = None) -> Fixture:
    path = _dir(fixtures_dir) / f"{fixture_id}.json"
    if not path.is_file():
        raise KeyError(fixture_id)
    return Fixture.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_fixtures(fixtures_dir: Path | None = None) -> list[FixtureSummary]:
    summaries = []
    for path in sorted(_dir(fixtures_dir).glob("*.json")):
        fx = load_fixture(path.stem, fixtures_dir)
        summaries.append(
            FixtureSummary(
                id=fx.id,
                classification=fx.label.classification,
                message_count=len(fx.thread),
                subject=fx.thread[-1].subject,
            )
        )
    return summaries
