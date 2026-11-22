"""Shared FastAPI dependencies: settings, DB sessions, the owner session and CSRF checks.

Auth model (single owner):
- `session` cookie: the owner id signed with `itsdangerous.TimestampSigner(session_secret)`,
  valid for `SESSION_MAX_AGE`. HttpOnly, so page scripts cannot read it.
- `csrf_token` cookie: a random token page scripts *can* read; every non-GET request must
  echo it in the `X-CSRF-Token` header (double-submit). A cross-site page can make the
  browser send the cookies, but it cannot read the token to set the header.
- `Origin`: when the browser sends it on a mutation, it must equal `app_base_url`.

`require_owner` opens its own short session instead of using the request one, so a long
response (the SSE stream) does not keep a transaction open.
"""

import hmac
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from itsdangerous import BadSignature, TimestampSigner
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import Owner
from app.providers.mock_mail import MockFaults, MockMailProvider

SESSION_COOKIE = "session"
CSRF_COOKIE = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"
SESSION_MAX_AGE = timedelta(days=7)
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_sessionmaker(request: Request) -> async_sessionmaker[AsyncSession]:
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    return maker


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionmakerDep = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)]


async def get_session(maker: SessionmakerDep) -> AsyncIterator[AsyncSession]:
    """One session per request, for reads (and writes the handler commits itself)."""
    async with maker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_faults(request: Request) -> MockFaults:
    faults: MockFaults = request.app.state.faults
    return faults


def get_mail(
    maker: SessionmakerDep, faults: Annotated[MockFaults, Depends(get_faults)]
) -> MockMailProvider:
    return MockMailProvider(maker, faults)


def signer(settings: Settings) -> TimestampSigner:
    return TimestampSigner(settings.session_secret.get_secret_value(), salt="owner-session")


def session_owner_id(settings: Settings, cookie: str | None) -> uuid.UUID | None:
    """The owner id in a valid, unexpired session cookie, else None."""
    if not cookie:
        return None
    try:
        raw = signer(settings).unsign(cookie, max_age=int(SESSION_MAX_AGE.total_seconds()))
        return uuid.UUID(raw.decode())
    except (BadSignature, ValueError):
        return None


def unauthenticated() -> HTTPException:
    return HTTPException(status_code=401, detail="unauthenticated")


async def require_owner(
    request: Request, settings: SettingsDep, maker: SessionmakerDep
) -> uuid.UUID:
    """The logged-in owner's id; 401 without a valid session for an existing owner."""
    owner_id = session_owner_id(settings, request.cookies.get(SESSION_COOKIE))
    if owner_id is None:
        raise unauthenticated()
    async with maker() as session:
        if await session.get(Owner, owner_id) is None:
            raise unauthenticated()
    return owner_id


OwnerId = Annotated[uuid.UUID, Depends(require_owner)]


def check_origin(request: Request, settings: Settings) -> None:
    """403 when an `Origin` header is present and is not the app's own origin."""
    origin = request.headers.get("origin")
    if origin is not None and origin.rstrip("/") != settings.app_base_url.rstrip("/"):
        raise HTTPException(status_code=403, detail="csrf_failed")


def require_csrf(request: Request, settings: SettingsDep) -> None:
    """For every non-GET request: same origin (if stated) and header token == cookie token."""
    if request.method in _SAFE_METHODS:
        return
    check_origin(request, settings)
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get(CSRF_HEADER)
    if not cookie or not header or not hmac.compare_digest(cookie.encode(), header.encode()):
        raise HTTPException(status_code=403, detail="csrf_failed")


# Router-level dependencies for every owner-only route.
OWNER_ONLY = [Depends(require_owner), Depends(require_csrf)]
