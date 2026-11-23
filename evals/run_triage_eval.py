"""Triage evaluation: run the real triage node over the labeled fixtures with Claude.

Usage (from backend/, with the database up and ANTHROPIC_API_KEY set):

    uv run python ../evals/run_triage_eval.py --split all --yes
    uv run python ../evals/run_triage_eval.py --split heldout --model claude-opus-5-5 --yes

It costs money, so without `--yes` it only prints the estimate and exits. Spend goes through
the app's daily budget guard (`DAILY_BUDGET_USD`) like any other model call; `--daily-budget`
raises the cap for this run only.

Reports accuracy, the recall of "respond" (spec §16 target: >= 0.95 on the held-out set), a
confusion matrix and the actual cost, and writes `evals/results/<date>-<model>.json`.

Splits: `heldout` is a fixed ~25% of fixtures chosen by a hash of the fixture id, so it never
changes as fixtures are added elsewhere; `all` is every fixture. A triage result of
`needs_clarification` counts as `notify`, because that is what the app does with it.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from app.llm.client import LLMCall, LLMClient, LLMResult, T  # noqa: E402
from app.providers.fixtures import Fixture, load_fixture  # noqa: E402
from app.providers.interfaces import ThreadSnapshot  # noqa: E402

LABELS = ("respond", "notify", "ignore")
RESPOND_RECALL_TARGET = 0.95
# Rough per-call size for the estimate: the triage prompt plus the thread, ~4 chars a token.
SYSTEM_TOKENS = 1_500
OUTPUT_TOKENS = 300


def in_heldout(fixture_id: str) -> bool:
    return hashlib.sha256(fixture_id.encode()).digest()[0] % 4 == 0


def load_split(fixtures_dir: Path, split: str) -> list[Fixture]:
    fixtures = [load_fixture(p.stem, fixtures_dir) for p in sorted(fixtures_dir.glob("*.json"))]
    return [f for f in fixtures if split == "all" or in_heldout(f.id)]


def thread_state(fixture: Fixture) -> dict[str, Any]:
    """The graph state triage reads: the thread snapshot and the triggering message."""
    snapshot = ThreadSnapshot(
        provider_thread_id=fixture.thread[0].provider_thread_id,
        messages=fixture.thread,
        version="eval",
    )
    return {
        "workflow_id": str(uuid.uuid4()),
        "provider_message_id": fixture.thread[-1].provider_message_id,
        "thread": snapshot.model_dump(mode="json"),
        "preferences": {},
    }


def estimate_usd(fixtures: Sequence[Fixture], model: str) -> Decimal:
    from app.llm.pricing import price_for

    price = price_for(model)
    total = Decimal(0)
    for f in fixtures:
        chars = sum(len(m.body_text or "") + len(m.body_html or "") + 200 for m in f.thread)
        tokens_in = SYSTEM_TOKENS + chars // 4
        total += (tokens_in * price.input + OUTPUT_TOKENS * price.output) / Decimal(1_000_000)
    return total.quantize(Decimal("0.0001"))


class _Unattributed:
    """Spend counts toward the daily budget but is not tied to a workflow run (there is none)."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.cost = Decimal(0)

    async def structured(self, call: LLMCall[T]) -> LLMResult[T]:
        result = await self.inner.structured(dataclasses.replace(call, workflow_id=None))
        self.cost += result.cost_usd
        return result


async def classify(fixtures: Sequence[Fixture], llm: LLMClient) -> list[dict[str, Any]]:
    from app.graphs.nodes.triage import triage

    runtime = SimpleNamespace(context=SimpleNamespace(llm=llm))
    rows = []
    for f in fixtures:
        try:
            out = await triage(thread_state(f), runtime)  # type: ignore[arg-type]
            predicted, error = out.get("classification"), out.get("error_code")
        except Exception as exc:  # one failed call must not lose the whole run
            predicted, error = None, type(exc).__name__
        rows.append(
            {
                "fixture_id": f.id,
                "expected": f.label.classification,
                "predicted": predicted,
                "error": error,
                "heldout": in_heldout(f.id),
            }
        )
        print(f"  {f.id:<32} expected={f.label.classification:<8} predicted={predicted}")
    return rows


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["predicted"] in LABELS]
    correct = sum(r["predicted"] == r["expected"] for r in scored)
    respond = [r for r in rows if r["expected"] == "respond"]
    hits = sum(r["predicted"] == "respond" for r in respond)
    pairs = Counter((r["expected"], r["predicted"] or "error") for r in rows)
    return {
        "total": len(rows),
        "errors": len(rows) - len(scored),
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "respond_recall": round(hits / len(respond), 4) if respond else None,
        "respond_recall_target": RESPOND_RECALL_TARGET,
        "confusion": {e: {p: pairs[(e, p)] for p in (*LABELS, "error")} for e in LABELS},
    }


def print_report(m: dict[str, Any], cost: Decimal) -> None:
    print(f"\nFixtures: {m['total']}  errors: {m['errors']}")
    print(f"Accuracy: {m['accuracy']}")
    recall = m["respond_recall"]
    verdict = (
        ""
        if recall is None
        else ("  (meets target)" if recall >= RESPOND_RECALL_TARGET else "  (BELOW target)")
    )
    print(f"Respond recall: {recall}  target >= {RESPOND_RECALL_TARGET}{verdict}")
    print("\nConfusion (rows = expected, columns = predicted)")
    cols = (*LABELS, "error")
    print(" " * 10 + "".join(f"{c:>9}" for c in cols))
    for e in LABELS:
        print(f"{e:<10}" + "".join(f"{m['confusion'][e][c]:>9}" for c in cols))
    print(f"\nActual cost: ${cost:.4f}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", choices=("all", "heldout"), default="heldout")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--fixtures-dir", type=Path, default=REPO / "evals" / "fixtures")
    p.add_argument("--results-dir", type=Path, default=REPO / "evals" / "results")
    p.add_argument("--daily-budget", type=Decimal, default=None, help="override DAILY_BUDGET_USD")
    p.add_argument("--yes", action="store_true", help="actually call the model (costs money)")
    return p.parse_args(argv)


async def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    fixtures = load_split(args.fixtures_dir, args.split)
    print(f"{len(fixtures)} fixtures ({args.split}), model {args.model}")
    print(f"Estimated cost: ${estimate_usd(fixtures, args.model)} (rough estimate)")
    if not args.yes:
        print("Not run: pass --yes to call the model.")
        return 2

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.llm.anthropic_client import AnthropicLLMClient, build_anthropic
    from app.llm.budget import BudgetGuard

    settings = get_settings().model_copy(update={"model_triage": args.model})
    engine = create_async_engine(settings.database_url)
    try:
        budget = BudgetGuard(
            async_sessionmaker(engine), args.daily_budget or settings.daily_budget_usd
        )
        llm = _Unattributed(AnthropicLLMClient(build_anthropic(settings), settings, budget))
        rows = await classify(fixtures, llm)
    finally:
        await engine.dispose()

    m = metrics(rows)
    print_report(m, llm.cost)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{datetime.now(UTC).date().isoformat()}-{args.model}.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "split": args.split,
                "cost_usd": str(llm.cost),
                **m,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # psycopg async
    sys.exit(asyncio.run(main(sys.argv[1:])))
