"""Owner login, logout and `me`.

There is exactly one way to become the owner: know `OWNER_PASSPHRASE`. The first successful
login creates the single `Owner` row (email = `OWNER_EMAIL`); later logins reuse it. A visitor
without the passphrase cannot claim the app (spec §13).

Failed logins are throttled in memory: after `MAX_FAILURES` failures within `WINDOW` every
login attempt (even a correct one) gets 429 until the oldest failure leaves the window. The
counter is global, not per client address, so rotating addresses does not help a guesser; the
cost is that a flood of bad attempts also locks the owner out for up to a minute.
"""

import hmac
import secrets
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text

from app.api.deps import (
    CSRF_COOKIE,
    CSRF_HEADER,
    OWNER_ONLY,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    OwnerId,
    SessionDep,
    SettingsDep,
    check_origin,
    signer,
)
from app.config import Settings
from app.db.models import Owner

__all__ = ["CSRF_HEADER", "LoginThrottle", "router"]

MAX_FAILURES = 5
WINDOW_SECONDS = 60.0
_OWNER_LOCK_KEY = 0x6F776E72  # "ownr": serializes the first owner creation


class LoginThrottle:
    """Counts failed logins in a sliding window (in memory, per process)."""

    def __init__(
        self,
        max_failures: int = MAX_FAILURES,
        window: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max, self._window, self._clock = max_failures, window, clock
        self._failures: deque[float] = deque()

    def _prune(self) -> None:
        cutoff = self._clock() - self._window
        while self._failures and self._failures[0] <= cutoff:
            self._failures.popleft()

    def blocked(self) -> bool:
        self._prune()
        return len(self._failures) >= self._max

    def record_failure(self) -> None:
        self._failures.append(self._clock())


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passphrase: str = Field(max_length=1024)


class LoginOut(BaseModel):
    csrf_token: str


class MeOut(BaseModel):
    owner_id: uuid.UUID
    email: str


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _set_cookies(response: Response, settings: Settings, owner_id: uuid.UUID, csrf: str) -> None:
    secure = settings.app_env == "production"
    max_age = int(SESSION_MAX_AGE.total_seconds())
    session_value = signer(settings).sign(str(owner_id)).decode()
    common: dict[str, Any] = {"max_age": max_age, "samesite": "lax", "secure": secure, "path": "/"}
    response.set_cookie(SESSION_COOKIE, session_value, httponly=True, **common)
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, **common)


async def _ensure_owner(session: SessionDep, settings: Settings) -> Owner:
    """The single owner row, created on the first successful login."""
    async with session.begin():
        await session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _OWNER_LOCK_KEY})
        owner = (
            await session.execute(select(Owner).order_by(Owner.created_at).limit(1))
        ).scalar_one_or_none()
        if owner is None:
            owner = Owner(email=settings.owner_email)
            session.add(owner)
            await session.flush()
    return owner


@router.post("/login")
async def login(
    body: LoginIn, request: Request, response: Response, session: SessionDep, settings: SettingsDep
) -> LoginOut:
    check_origin(request, settings)  # no CSRF token exists yet; the origin check still applies
    throttle: LoginThrottle = request.app.state.login_throttle
    if throttle.blocked():
        raise HTTPException(status_code=429, detail="too_many_attempts")
    expected = settings.owner_passphrase.get_secret_value().encode()
    if not hmac.compare_digest(body.passphrase.encode(), expected):
        throttle.record_failure()
        raise HTTPException(status_code=401, detail="invalid_credentials")
    owner = await _ensure_owner(session, settings)
    csrf = secrets.token_urlsafe(32)
    _set_cookies(response, settings, owner.id, csrf)
    return LoginOut(csrf_token=csrf)


@router.post("/logout", dependencies=OWNER_ONLY)
async def logout(response: Response) -> dict[str, str]:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    return {"status": "ok"}


@router.get("/me")
async def me(owner_id: OwnerId, session: SessionDep) -> MeOut:
    owner = await session.get_one(Owner, owner_id)
    return MeOut(owner_id=owner.id, email=owner.email)
