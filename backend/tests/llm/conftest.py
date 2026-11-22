"""Fixtures for LLM tests.

The Anthropic SDK 2.x talks through `httpx2` (its own fork of httpx), which respx cannot patch
globally. So respx's router is used for route matching and call recording, and a small bridge
transport hands every `httpx2` request to it.
"""

import httpx
import httpx2
import pytest
import respx
from anthropic import AsyncAnthropic


@pytest.fixture
def api() -> respx.MockRouter:
    """respx router standing in for api.anthropic.com."""
    return respx.MockRouter(assert_all_called=False)


@pytest.fixture
def make_sdk(api: respx.MockRouter):  # type: ignore[no-untyped-def]
    async def bridge(request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        mirrored = httpx.Request(
            request.method,
            str(request.url),
            headers=list(request.headers.raw),
            content=request.content,
        )
        try:
            response = await api.async_handler(mirrored)
        except httpx.ConnectError as exc:
            raise httpx2.ConnectError(str(exc), request=request) from None
        await response.aread()
        return httpx2.Response(
            response.status_code, headers=list(response.headers.raw), content=response.content
        )

    def make() -> AsyncAnthropic:
        http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(bridge))
        return AsyncAnthropic(api_key="test-key", max_retries=0, http_client=http_client)

    return make
