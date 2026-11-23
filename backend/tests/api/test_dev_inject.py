"""Dev-only routes: fixtures, inject, faults and outbox; absent in production."""

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import EmailMessage, Job, MockOutbox
from app.main import create_app
from app.providers.mock_mail import MockFaults


async def test_inject_enqueues_start_job(authed: AsyncClient, session: AsyncSession) -> None:
    listed = await authed.get("/api/v1/dev/fixtures")
    assert listed.status_code == 200
    assert "fyi-bank-statement" in {f["id"] for f in listed.json()}

    r = await authed.post("/api/v1/dev/inject", json={"fixture_id": "fyi-bank-statement"})
    assert r.status_code == 200
    [result] = r.json()["results"]
    assert result["provider_message_id"] == "fyi-bank-statement-m1"
    assert result["status"] == "queued"
    job = (await session.execute(select(Job).where(Job.kind == "start_workflow"))).scalar_one()
    assert job.payload == {
        "account_id": "mock-account",
        "provider_message_id": "fyi-bank-statement-m1",
        "automated": False,
    }
    assert str(job.id) == result["workflow_job_id"]
    assert await session.scalar(select(EmailMessage.subject)) == "Your monthly statement is ready"

    again = await authed.post("/api/v1/dev/inject", json={"fixture_id": "fyi-bank-statement"})
    assert again.json()["results"][0]["status"] == "duplicate"


async def test_inject_unknown_or_unsafe_fixture(authed: AsyncClient) -> None:
    assert (await authed.post("/api/v1/dev/inject", json={"fixture_id": "nope"})).status_code == 404
    unsafe = await authed.post("/api/v1/dev/inject", json={"fixture_id": "../secrets"})
    assert unsafe.status_code == 422


async def test_faults_and_outbox(authed: AsyncClient, app: FastAPI, session: AsyncSession) -> None:
    r = await authed.patch("/api/v1/dev/faults", json={"send": "timeout_after_accept"})
    assert r.status_code == 200
    assert r.json() == {"send": "timeout_after_accept", "calendar_create": "ok"}
    faults: MockFaults = app.state.faults
    assert faults.send == "timeout_after_accept"
    assert (await authed.patch("/api/v1/dev/faults", json={"send": "explode"})).status_code == 422

    session.add(MockOutbox(operation_id="op-1", payload={"to": ["alice@example.com"]}))
    await session.commit()
    outbox = (await authed.get("/api/v1/dev/outbox")).json()
    assert [o["operation_id"] for o in outbox] == ["op-1"]
    assert outbox[0]["payload"] == {"to": ["alice@example.com"]}


async def test_dev_routes_absent_in_production(
    base_env: None, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    settings = Settings(_env_file=None, app_env="production")  # type: ignore[call-arg]
    app = create_app(settings, sessionmaker=sessionmaker)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        login = await c.post("/api/v1/auth/login", json={"passphrase": "a-real-passphrase"})
        assert login.status_code == 200
        assert "secure" in login.headers["set-cookie"].lower()
        token = login.json()["csrf_token"]
        # Secure cookies are not sent over plain http; pass them explicitly.
        cookies = {"session": login.cookies["session"], "csrf_token": token}
        headers = {
            "X-CSRF-Token": token,
            "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items()),
        }
        for method, path in [
            ("GET", "/api/v1/dev/fixtures"),
            ("POST", "/api/v1/dev/inject"),
            ("PATCH", "/api/v1/dev/faults"),
            ("GET", "/api/v1/dev/outbox"),
        ]:
            r = await c.request(method, path, json={}, headers=headers)
            assert r.status_code == 404, path
        # The same session is accepted elsewhere, so the 404 is the missing router.
        assert (await c.get("/api/v1/auth/me", headers=headers)).status_code == 200
        # `once` is a dev/test flag only.
        assert (
            await c.get("/api/v1/events", params={"once": "true"}, headers=headers)
        ).status_code == 422
