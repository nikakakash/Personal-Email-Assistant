"""evals/run_triage_eval.py without spending money: refusal, splits, metrics, scripted run."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from app.llm.scripted import ScriptedLLMClient

SCRIPT = Path(__file__).resolve().parents[2] / "evals" / "run_triage_eval.py"
FIXTURES = SCRIPT.parent / "fixtures"


@pytest.fixture(scope="module")
def ev() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_triage_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_triage_eval"] = module
    spec.loader.exec_module(module)
    return module


async def test_refuses_without_yes_and_prints_estimate(
    ev: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await ev.main(["--split", "all"]) == 2
    out = capsys.readouterr().out
    assert "Estimated cost: $" in out
    assert "pass --yes" in out


def test_heldout_is_a_stable_subset(ev: ModuleType) -> None:
    every = ev.load_split(FIXTURES, "all")
    held = ev.load_split(FIXTURES, "heldout")
    assert 0 < len(held) < len(every)
    assert {f.id for f in held} <= {f.id for f in every}
    assert [f.id for f in held] == [f.id for f in ev.load_split(FIXTURES, "heldout")]


def test_metrics_accuracy_respond_recall_and_confusion(ev: ModuleType) -> None:
    rows: list[dict[str, Any]] = [
        {"expected": "respond", "predicted": "respond"},
        {"expected": "respond", "predicted": "notify"},
        {"expected": "notify", "predicted": "notify"},
        {"expected": "ignore", "predicted": None},
    ]
    m = ev.metrics(rows)
    assert m["accuracy"] == 0.5
    assert m["respond_recall"] == 0.5
    assert m["errors"] == 1
    assert m["confusion"]["respond"] == {"respond": 1, "notify": 1, "ignore": 0, "error": 0}
    assert m["confusion"]["ignore"]["error"] == 1


async def test_classify_runs_the_triage_node(ev: ModuleType) -> None:
    fixtures = ev.load_split(FIXTURES, "all")[:2]
    scripts = {
        f.thread[-1].provider_message_id: {
            "triage": [
                {
                    "classification": f.label.classification,
                    "urgency": "normal",
                    "reason": "r",
                    "evidence_refs": [f"message:{f.thread[-1].provider_message_id}"],
                    "needs_clarification": False,
                }
            ]
        }
        for f in fixtures
    }
    rows = await ev.classify(fixtures, ScriptedLLMClient(scripts))
    assert [r["predicted"] for r in rows] == [f.label.classification for f in fixtures]
    assert ev.metrics(rows)["accuracy"] == 1.0
