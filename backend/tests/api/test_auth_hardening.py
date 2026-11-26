"""Auth hardening: expired and forged sessions, hostile origins, unconfigured secrets, the
throttle window, docs in production, and a guard over every mutating route in the app."""

import re
import time
import uuid
from datetime import timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from itsdangerous import TimestampSigner
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.auth import CSRF_HEADER, LoginThrottle
from app.api.deps import SESSION_MAX_AGE, signer
from app.config import Settings
from app.main import create_app
from tests.api.conftest import PASSPHRASE


class _BackdatedSigner(TimestampSigner):
    """Signs as if `age` ago, to produce a session cookie of a given age."""

    def __init__(self, secret: str, age: timedelta) -> None:
        super().__init__(secret, salt="owner-session")
        self._age = age

    def get_timestamp(self) -> int:
        return int(time.time() - self._age.total_seconds())


async def test_expired_session_cookie_401(authed: AsyncClient, settings: Settings) -> None:
    owner_id = (await authed.get("/api/v1/auth/me")).json()["owner_id"]
    secret = settings.session_secret.get_secret_value()
    fresh = _BackdatedSigner(secret, SESSION_MAX_AGE - timedelta(minutes=1)).sign(owner_id)
    authed.cookies.set("session", fresh.decode())
    assert (await authed.get("/api/v1/auth/me")).status_code == 200
    expired = _BackdatedSigner(secret, SESSION_MAX_AGE + timedelta(minutes=1)).sign(owner_id)
    authed.cookies.set("session", expired.decode())
    assert (await authed.get("/api/v1/auth/me")).status_code == 401


async def test_signed_cookie_for_unknown_owner_401(anon: AsyncClient, settings: Settings) -> None:
    anon.cookies.set("session", signer(settings).sign(str(uuid.uuid4())).decode())
    r = await anon.get("/api/v1/auth/me")
    assert r.status_code == 401 and r.json() == {"error": "unauthenticated"}


@pytest.mark.parametrize(
    "origin",
    ["null", "http://evil.localhost:8000", "http://localhost:8001", "https://localhost:8000"],
)
async def test_hostile_origins_403(authed: AsyncClient, origin: str) -> None:
    r = await authed.post(
        f"/api/v1/workflows/{uuid.uuid4()}/cancel",
        json={"request_id": "r1"},
        headers={"Origin": origin},
    )
    assert r.status_code == 403 and r.json() == {"error": "csrf_failed"}


@pytest.mark.parametrize(
    "value",
    ["replace-me", "", "   ", "short-11ch!"],
    ids=["placeholder", "empty", "blank", "short"],
)
@pytest.mark.parametrize("name", ["OWNER_PASSPHRASE", "SESSION_SECRET"])
async def test_login_refuses_unconfigured_secrets_503(
    base_env: None,
    sessionmaker: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    app = create_app(Settings(_env_file=None), sessionmaker=sessionmaker)  # type: ignore[call-arg]
    # Even sending the configured value itself is refused: a default never claims ownership.
    guess = value if name == "OWNER_PASSPHRASE" else PASSPHRASE
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/auth/login", json={"passphrase": guess})
    assert r.status_code == 503 and r.json() == {"error": "not_configured"}
    assert "session" not in r.cookies


@pytest.mark.parametrize("value", ["replace-me", "", "short-11ch!"])
@pytest.mark.parametrize("name", ["OWNER_PASSPHRASE", "SESSION_SECRET"])
def test_production_settings_reject_weak_secrets(
    base_env: None, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=name.lower()):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_production_settings_accept_real_secrets(base_env: None) -> None:
    settings = Settings(_env_file=None, app_env="production")  # type: ignore[call-arg]
    assert settings.app_env == "production"


def test_throttle_window_expires() -> None:
    now = [1000.0]
    throttle = LoginThrottle(max_failures=5, window=60, clock=lambda: now[0])
    for _ in range(5):
        throttle.record_failure()
    assert throttle.blocked()
    now[0] += 59.9
    assert throttle.blocked()
    now[0] += 0.2  # the oldest failure has left the window
    assert not throttle.blocked()


async def test_login_allowed_again_after_throttle_window(app: FastAPI, anon: AsyncClient) -> None:
    now = [0.0]
    app.state.login_throttle = LoginThrottle(clock=lambda: now[0])
    for _ in range(5):
        await anon.post("/api/v1/auth/login", json={"passphrase": "wrong"})
    login = {"passphrase": PASSPHRASE}
    assert (await anon.post("/api/v1/auth/login", json=login)).status_code == 429
    now[0] += 61
    assert (await anon.post("/api/v1/auth/login", json=login)).status_code == 200


async def test_docs_disabled_in_production(
    base_env: None, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    prod_settings = Settings(_env_file=None, app_env="production")  # type: ignore[call-arg]
    dev_settings = Settings(_env_file=None)  # type: ignore[call-arg]
    for settings, expected in ((prod_settings, 404), (dev_settings, 200)):
        app = create_app(settings, sessionmaker=sessionmaker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for path in ("/docs", "/redoc", "/openapi.json"):
                assert (await c.get(path)).status_code == expected, (path, settings.app_env)


# ------------------------------------------------------ every mutating route is guarded

_EXEMPT = {("POST", "/api/v1/auth/login")}  # no session or CSRF token exists before login
_PATH_PARAMS = {"namespace": "writing"}


def _mutating_routes() -> list[tuple[str, str]]:
    """Every non-GET route of a development app (which includes the dev router).

    Routes with `include_in_schema=False` would be missed; none exist, and adding one should
    extend this list by hand.
    """
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="development",
        app_base_url="http://localhost:8000",
        database_url="postgresql+psycopg://u:p@localhost/x",
        checkpoint_database_url="postgresql://u:p@localhost/x",
        owner_email="owner@example.com",
        owner_passphrase=PASSPHRASE,
        session_secret="a-real-session-secret",
    )
    # The OpenAPI schema lists every mounted route; FastAPI keeps included routers nested
    # in `app.routes`, so walking that list would miss them.
    paths = create_app(settings).openapi()["paths"]
    return sorted(
        (method.upper(), path)
        for path, operations in paths.items()
        for method in operations
        if method.upper() not in {"GET", "HEAD", "OPTIONS"}
    )


MUTATING = _mutating_routes()


def _concrete(path: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: _PATH_PARAMS.get(m[1], str(uuid.uuid4())), path)


def test_route_inventory_is_complete() -> None:
    """Guards the parametrized test below against silently covering nothing."""
    assert ("POST", "/api/v1/workflows/{workflow_id}/cancel") in MUTATING
    assert ("PATCH", "/api/v1/dev/faults") in MUTATING
    assert ("POST", "/api/v1/auth/logout") in MUTATING
    assert _EXEMPT <= set(MUTATING)
    assert len(MUTATING) >= 10


@pytest.mark.parametrize(("method", "path"), [r for r in MUTATING if r not in _EXEMPT])
async def test_every_mutating_route_requires_csrf_and_session(
    authed: AsyncClient, anon: AsyncClient, method: str, path: str
) -> None:
    url = _concrete(path)
    token = authed.headers.pop(CSRF_HEADER)
    no_csrf = await authed.request(method, url, json={})
    assert no_csrf.status_code == 403, (method, path, no_csrf.text)

    anon.cookies.set("csrf_token", token)
    no_session = await anon.request(method, url, json={}, headers={CSRF_HEADER: token})
    assert no_session.status_code == 401, (method, path, no_session.text)
