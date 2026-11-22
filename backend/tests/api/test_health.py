"""Health endpoint."""

from httpx import AsyncClient


async def test_live(client: AsyncClient) -> None:
    r = await client.get("/api/v1/health/live")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
