"""evals/run_rag_draft_eval.py without spending money: refusal, scoring, a scripted run."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.scripted import ScriptedLLMClient
from app.retrieval.embeddings import HashEmbeddingsClient

SCRIPT = Path(__file__).resolve().parents[2] / "evals" / "run_rag_draft_eval.py"


@pytest.fixture(scope="module")
def ev() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_rag_draft_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_rag_draft_eval"] = module
    spec.loader.exec_module(module)
    return module


async def test_refuses_without_yes_and_prints_estimate(
    ev: ModuleType, capsys: pytest.CaptureFixture[str], base_env: None
) -> None:
    assert await ev.main([]) == 2
    out = capsys.readouterr().out
    assert "Estimated cost: $" in out and "--yes" in out


def test_expected_facts_belong_to_known_queries(ev: ModuleType) -> None:
    from evals.retrieval.corpus import QUERIES

    assert set(ev.EXPECTED) <= {q[0] for q in QUERIES}
    assert len(ev.EXPECTED) >= 20


def test_scoring(ev: ModuleType) -> None:
    assert ev.fact_hit("The total is $5,200.", ["5,200", "5200"])
    assert not ev.fact_hit("I will check the amount.", ["5,200"])
    assert ev.citation_scores(["history:a", "history:b", "message:m"], ["a", "b"], ["a"]) == (
        0.5,
        0,
    )
    assert ev.citation_scores(["history:z"], ["a"], ["a"]) == (0.0, 1)
    assert ev.citation_scores(["message:m"], ["a"], ["a"]) == (None, 0)


async def test_scripted_run_scores_rag_on_and_off(
    ev: ModuleType, base_env: None, db_url: str, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    def reply(body: str, refs: list[str]) -> dict[str, Any]:
        return {
            "to": ["alex@brightweb.example"],
            "cc": [],
            "body_text": body,
            "evidence_refs": refs,
        }

    verdict = {"faithful": True, "unsupported": [], "reason": "ok"}
    llm = ScriptedLLMClient(
        {
            "query-q01": {
                "draft_reply": [
                    reply("We agreed on $5,200.", ["message:query-q01", "history:bw-3"]),
                    reply("Let me check the amount and get back to you.", ["message:query-q01"]),
                ],
                "judge": [verdict, verdict],
            }
        }
    )
    settings = ev.eval_settings(db_url, ev.Decimal("1"), llm_provider="scripted")
    results = await ev.run(sessionmaker, settings, llm, HashEmbeddingsClient(), ["q01"])
    on, off = results
    assert on.rag and "bw-3" in on.retrieved and on.fact_hit
    assert on.citation_precision == 1.0 and on.invalid_citations == 0
    assert not off.rag and off.retrieved == [] and not off.fact_hit
    summary = ev.summarize(results)
    assert summary["rag_on"]["fact_accuracy"] == 1.0
    assert summary["rag_off"]["fact_accuracy"] == 0.0
