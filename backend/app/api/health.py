"""Liveness and readiness endpoints (no auth: they reveal nothing but up or down)."""

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.deps import SessionmakerDep

log = logging.getLogger(__name__)

READY_TIMEOUT_SECONDS = 3.0

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", response_model=None)
async def ready(maker: SessionmakerDep) -> dict[str, str] | JSONResponse:
    """`SELECT 1` against the application database; 503 if it fails or takes too long."""
    try:
        async with asyncio.timeout(READY_TIMEOUT_SECONDS), maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        log.warning("readiness check failed", exc_info=True)
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return {"status": "ok"}
