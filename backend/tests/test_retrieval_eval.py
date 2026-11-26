"""evals/run_retrieval_eval.py: corpus integrity, metric math, baseline check, an end-to-end run
on the test database with the hash embedder (free, no model download)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evals" / "run_retrieval_eval.py"


@pytest.fixture(scope="module")
def ev() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_retrieval_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_retrieval_eval"] = module
    spec.loader.exec_module(module)
    return module


def _parties(message: tuple[object, ...]) -> set[str]:
    _, _, sender, to, *_ = message
    return {str(sender), *map(str, to)}  # type: ignore[call-overload]


def test_corpus_is_consistent(ev: ModuleType) -> None:
    from evals.retrieval.corpus import MESSAGES, OWNER, QUERIES

    ids = [m[0] for m in MESSAGES]
    assert len(ids) == len(set(ids)) and len(ids) >= 140
    assert len({q[0] for q in QUERIES}) == len(QUERIES) >= 60
    by_id = {m[0]: m for m in MESSAGES}
    for qid, sender, _subject, _body, relevant, category in QUERIES:
        assert relevant, qid
        assert category in {
            "paraphrase",
            "superseded",
            "identifier",
            "collision",
            "multi",
            "near-duplicate",
        }, qid
        for rid in relevant:
            assert rid in by_id, (qid, rid)
            # The participant scope must be able to find every answer.
            assert sender in _parties(by_id[rid]), (qid, rid)
        assert sender != OWNER


def test_metrics(ev: ModuleType) -> None:
    assert ev.recall_at(["a", "b"], ["a", "x", "b"], 2) == 0.5
    assert ev.recall_at(["a", "b"], ["a", "x", "b"], 3) == 1.0
    assert ev.reciprocal_rank(["b"], ["a", "b"]) == 0.5
    assert ev.reciprocal_rank(["z"], ["a", "b"]) == 0.0
    assert ev.ndcg_at(["a"], ["a"], 10) == 1.0
    assert ev.ndcg_at(["a", "b"], ["b", "x", "a"], 10) == pytest.approx(
        (1 + 1 / 2) / (1 + 1 / 1.5849625007211562)
    )
    assert ev.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert ev.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_check_baseline(ev: ModuleType) -> None:
    report = ev.Report("hash", "hybrid", "participants", 1, 0.8, 0.9, 0.7, 0.75, 1.0, 2.0)
    ok = [
        {
            "embedder": "hash",
            "mode": "hybrid",
            "scope": "participants",
            "metric": "recall_at_5",
            "min": 0.75,
        }
    ]
    assert ev.check_baseline([report], ok) == []
    assert ev.check_baseline([report], [{**ok[0], "min": 0.85}])
    assert ev.check_baseline([report], [{**ok[0], "mode": "vector"}])


async def test_paid_runs_refuse_without_yes(
    ev: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await ev.main(["--embedders", "voyage"]) == 2
    assert "--yes" in capsys.readouterr().out


async def test_end_to_end_with_hash_embedder(
    ev: ModuleType, db_url: str, tmp_path: Path, sessionmaker: object
) -> None:
    out = tmp_path / "r.json"
    baseline = tmp_path / "b.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "embedder": "hash",
                    "mode": "hybrid",
                    "scope": "participants",
                    "metric": "recall_at_5",
                    "min": 0.5,
                }
            ]
        )
    )
    code = await ev.main(
        [
            "--embedders",
            "hash",
            "--modes",
            "keyword,vector,hybrid",
            "--scope",
            "participants",
            "--db-url",
            db_url,
            "--out",
            str(out),
            "--check-baseline",
            str(baseline),
        ]
    )
    assert code == 0
    reports = {r["mode"]: r for r in json.loads(out.read_text())["reports"]}
    assert set(reports) == {"keyword", "vector", "hybrid"}
    assert reports["hybrid"]["queries"] == 60
    assert 0.0 <= reports["hybrid"]["recall_at_5"] <= 1.0
