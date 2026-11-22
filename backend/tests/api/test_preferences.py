"""Preferences (versioned per namespace), flags and usage."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AppFlags, LlmUsage, PreferenceVersion

SCHEDULING = {
    "time_zone": "Europe/Berlin",
    "work_start": "08:30",
    "work_end": "17:00",
    "work_days": [0, 1, 2, 3, 4],
    "buffer_minutes": 15,
    "default_duration_minutes": 30,
}


async def test_preferences_version_conflict_409(authed: AsyncClient, session: AsyncSession) -> None:
    assert (await authed.get("/api/v1/preferences")).json() == {}
    url = "/api/v1/preferences/writing"
    first = await authed.patch(url, json={"value": {"tone": "brief"}, "expected_revision": 0})
    assert first.status_code == 200
    assert first.json()["revision"] == 1 and first.json()["value"]["tone"] == "brief"

    second = await authed.patch(url, json={"value": {"tone": "warm"}, "expected_revision": 1})
    assert second.status_code == 200 and second.json()["revision"] == 2

    stale = await authed.patch(url, json={"value": {"tone": "formal"}, "expected_revision": 1})
    assert stale.status_code == 409
    assert stale.json() == {"error": "conflict", "reason": "revision_mismatch"}

    rows = (
        (await session.execute(select(PreferenceVersion).order_by(PreferenceVersion.revision)))
        .scalars()
        .all()
    )
    assert [(r.revision, r.status, r.previous_revision) for r in rows] == [
        (1, "superseded", None),
        (2, "active", 1),
    ]
    current = (await authed.get("/api/v1/preferences")).json()
    assert current["writing"]["revision"] == 2 and current["writing"]["value"]["tone"] == "warm"


async def test_scheduling_preferences_validation_422(authed: AsyncClient) -> None:
    url = "/api/v1/preferences/scheduling"
    ok = await authed.patch(url, json={"value": SCHEDULING, "expected_revision": 0})
    assert ok.status_code == 200, ok.text
    bad_values = [
        {"time_zone": "Mars/Olympus"},
        {"work_start": "9am"},
        {"work_start": "18:00", "work_end": "09:00"},
        {"work_days": [0, 7]},
        {"work_days": []},
        {"buffer_minutes": -5},
        {"default_duration_minutes": 0},
        {"default_duration_minutes": 100000},
        {"unknown_field": 1},
    ]
    for bad in bad_values:
        r = await authed.patch(url, json={"value": {**SCHEDULING, **bad}, "expected_revision": 1})
        assert r.status_code == 422, bad
    assert (
        await authed.patch(
            "/api/v1/preferences/secrets", json={"value": {}, "expected_revision": 0}
        )
    ).status_code == 404


async def test_flags_get_and_patch(authed: AsyncClient, session: AsyncSession) -> None:
    assert (await authed.get("/api/v1/flags")).json() == {
        "ingestion_paused": False,
        "outbound_paused": False,
    }
    r = await authed.patch("/api/v1/flags", json={"outbound_paused": True})
    assert r.json() == {"ingestion_paused": False, "outbound_paused": True}
    r = await authed.patch("/api/v1/flags", json={"ingestion_paused": True})
    assert r.json() == {"ingestion_paused": True, "outbound_paused": True}
    flags = await session.get_one(AppFlags, 1)
    assert flags.ingestion_paused and flags.outbound_paused


async def test_usage_daily_totals_by_model(authed: AsyncClient, session: AsyncSession) -> None:
    now = datetime.now(UTC)
    session.add_all(
        [
            LlmUsage(
                purpose="triage",
                model="m1",
                input_tokens=10,
                output_tokens=5,
                cost_usd=Decimal("0.10"),
                status="settled",
            ),
            LlmUsage(
                purpose="plan",
                model="m1",
                input_tokens=20,
                output_tokens=5,
                cost_usd=Decimal("0.20"),
                status="settled",
            ),
            LlmUsage(
                purpose="draft_reply",
                model="m2",
                input_tokens=1,
                output_tokens=1,
                cost_usd=Decimal("0.05"),
                status="settled",
            ),
            LlmUsage(
                purpose="triage",
                model="m1",
                input_tokens=7,
                output_tokens=0,
                cost_usd=Decimal("1.00"),
                status="settled",
                created_at=now - timedelta(days=2),
            ),
        ]
    )
    await session.commit()
    body = (await authed.get("/api/v1/usage")).json()
    today = now.date().isoformat()
    todays = {(d["model"]): d for d in body["daily"] if d["date"] == today}
    assert todays["m1"]["input_tokens"] == 30 and todays["m1"]["output_tokens"] == 10
    assert Decimal(todays["m1"]["cost_usd"]) == Decimal("0.30")
    assert Decimal(todays["m2"]["cost_usd"]) == Decimal("0.05")
    assert len(body["daily"]) == 3
    assert Decimal(body["today"]["cost_usd"]) == Decimal("0.35")
    assert Decimal(body["today"]["budget_usd"]) == Decimal("2.00")
