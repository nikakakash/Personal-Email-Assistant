"""`POST /proposals/{id}/decision`: 202 and idempotent repeats, 409 on stale, 422 on bad input."""

from collections.abc import Awaitable, Callable
from typing import Any

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job, Proposal
from tests.api.conftest import SeededProposal

SeedProposal = Callable[[], Awaitable[SeededProposal]]


def _body(seeded: SeededProposal, **extra: Any) -> dict[str, Any]:
    return {
        "decision": "approve",
        "proposal_revision": seeded.revision,
        "payload_hash": seeded.payload_hash,
        "interrupt_id": seeded.interrupt_id,
        "request_id": "req-1",
        **extra,
    }


async def test_decision_202_then_duplicate_same_outcome(
    authed: AsyncClient, session: AsyncSession, seed_proposal: SeedProposal
) -> None:
    seeded = await seed_proposal()
    url = f"/api/v1/proposals/{seeded.proposal_id}/decision"

    first = await authed.post(url, json=_body(seeded))
    assert first.status_code == 202
    out = first.json()
    assert out["status"] == "accepted" and out["revision"] == seeded.revision
    job = (
        await session.execute(select(Job).where(Job.dedup_key == f"resume:{seeded.interrupt_id}"))
    ).scalar_one()
    assert out["job_id"] == str(job.id)
    proposal = await session.get_one(Proposal, seeded.proposal_id)
    assert proposal.status == "approved"

    again = await authed.post(url, json=_body(seeded))
    assert again.status_code == 202 and again.json() == out
    other = await authed.post(url, json=_body(seeded, request_id="req-2"))
    assert other.status_code == 409 and other.json() == {
        "error": "conflict",
        "reason": "not_pending",
    }


async def test_decision_stale_409(authed: AsyncClient, seed_proposal: SeedProposal) -> None:
    seeded = await seed_proposal()
    url = f"/api/v1/proposals/{seeded.proposal_id}/decision"
    r = await authed.post(url, json=_body(seeded, proposal_revision=seeded.revision + 1))
    assert r.status_code == 409
    assert r.json() == {"error": "conflict", "reason": "stale_revision"}
    r = await authed.post(url, json=_body(seeded, payload_hash="other"))
    assert r.json() == {"error": "conflict", "reason": "hash_mismatch"}


async def test_decision_validation_422(authed: AsyncClient, seed_proposal: SeedProposal) -> None:
    seeded = await seed_proposal()
    url = f"/api/v1/proposals/{seeded.proposal_id}/decision"
    r = await authed.post(url, json=_body(seeded, unexpected="x"))
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    # edit_approve needs an edited payload (model validator): still a 422, no raw input echoed.
    r = await authed.post(url, json=_body(seeded, decision="edit_approve"))
    assert r.status_code == 422
    assert all("input" not in e for e in r.json()["errors"])


async def test_decision_invalid_edit_422(authed: AsyncClient, seed_proposal: SeedProposal) -> None:
    seeded = await seed_proposal()
    r = await authed.post(
        f"/api/v1/proposals/{seeded.proposal_id}/decision",
        json=_body(seeded, decision="edit_approve", edited_payload={"subject": "changed"}),
    )
    assert r.status_code == 422
    assert r.json() == {"error": "invalid_edit", "errors": ["field_not_editable:subject"]}
