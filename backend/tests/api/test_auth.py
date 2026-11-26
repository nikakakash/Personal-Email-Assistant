"""Owner login, session cookie, CSRF and origin checks, and the failed-login throttle."""

import uuid

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import CSRF_HEADER
from app.db.models import Owner
from tests.api.conftest import PASSPHRASE


async def test_unauthenticated_401(anon: AsyncClient) -> None:
    for path in ("/api/v1/auth/me", "/api/v1/reviews", f"/api/v1/workflows/{uuid.uuid4()}"):
        r = await anon.get(path)
        assert r.status_code == 401, path
        assert r.json() == {"error": "unauthenticated"}


async def test_tampered_session_cookie_401(anon: AsyncClient) -> None:
    anon.cookies.set("session", f"{uuid.uuid4()}.forged.signature")
    assert (await anon.get("/api/v1/auth/me")).status_code == 401


async def test_login_wrong_passphrase_401(anon: AsyncClient) -> None:
    r = await anon.post("/api/v1/auth/login", json={"passphrase": "wrong"})
    assert r.status_code == 401
    assert r.json() == {"error": "invalid_credentials"}
    assert "session" not in r.cookies


async def test_login_sets_cookies_and_creates_single_owner(
    anon: AsyncClient, session: AsyncSession
) -> None:
    for _ in range(2):
        r = await anon.post("/api/v1/auth/login", json={"passphrase": PASSPHRASE})
        assert r.status_code == 200
    token = r.json()["csrf_token"]
    set_cookies = r.headers.get_list("set-cookie")
    session_cookie = next(c for c in set_cookies if c.startswith("session="))
    csrf_cookie = next(c for c in set_cookies if c.startswith("csrf_token="))
    assert "httponly" in session_cookie.lower() and "samesite=lax" in session_cookie.lower()
    assert "secure" not in session_cookie.lower()  # not production
    assert "httponly" not in csrf_cookie.lower() and token in csrf_cookie
    assert await session.scalar(select(func.count()).select_from(Owner)) == 1

    me = await anon.get("/api/v1/auth/me")
    assert me.status_code == 200 and me.json()["email"] == "owner@example.com"

    anon.headers[CSRF_HEADER] = token
    assert (await anon.post("/api/v1/auth/logout")).status_code == 200
    assert (await anon.get("/api/v1/auth/me")).status_code == 401


async def test_login_throttle_429(anon: AsyncClient) -> None:
    for _ in range(5):
        r = await anon.post("/api/v1/auth/login", json={"passphrase": "wrong"})
        assert r.status_code == 401
    # Throttled: even the right passphrase is refused until the window passes.
    r = await anon.post("/api/v1/auth/login", json={"passphrase": PASSPHRASE})
    assert r.status_code == 429
    assert r.json() == {"error": "too_many_attempts"}


async def test_mutation_without_csrf_403(authed: AsyncClient) -> None:
    url = f"/api/v1/workflows/{uuid.uuid4()}/cancel"
    body = {"request_id": "r1"}
    token = authed.headers.pop(CSRF_HEADER)
    r = await authed.post(url, json=body)
    assert r.status_code == 403 and r.json() == {"error": "csrf_failed"}
    r = await authed.post(url, json=body, headers={CSRF_HEADER: token + "x"})
    assert r.status_code == 403
    # With the token the request gets past CSRF (and finds no such workflow).
    r = await authed.post(url, json=body, headers={CSRF_HEADER: token})
    assert r.status_code == 404


async def test_mutation_with_foreign_origin_403(authed: AsyncClient) -> None:
    url = f"/api/v1/workflows/{uuid.uuid4()}/cancel"
    body = {"request_id": "r1"}
    r = await authed.post(url, json=body, headers={"Origin": "http://evil.test"})
    assert r.status_code == 403 and r.json() == {"error": "csrf_failed"}
    r = await authed.post(url, json=body, headers={"Origin": "http://localhost:8000"})
    assert r.status_code == 404


async def test_login_with_foreign_origin_403(anon: AsyncClient) -> None:
    r = await anon.post(
        "/api/v1/auth/login",
        json={"passphrase": PASSPHRASE},
        headers={"Origin": "http://evil.test"},
    )
    assert r.status_code == 403
