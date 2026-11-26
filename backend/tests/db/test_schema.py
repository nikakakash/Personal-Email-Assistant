"""Schema tests: migration output, uniqueness guarantees and CHECK constraints."""

from collections.abc import Awaitable, Callable

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Approval, Base, Job, Owner, ProposalRevision, WorkflowRun

RevisionFactory = Callable[[], Awaitable[ProposalRevision]]
WorkflowFactory = Callable[[], Awaitable[WorkflowRun]]


async def test_migration_creates_tables(session: AsyncSession) -> None:
    names = set(
        (
            await session.execute(text("select tablename from pg_tables where schemaname='public'"))
        ).scalars()
    )
    assert {t.name for t in Base.metadata.sorted_tables} <= names
    assert {
        "jobs",
        "proposals",
        "proposal_revisions",
        "approvals",
        "actions",
        "review_items",
        "workflow_events",
        "llm_usage",
        "preference_versions",
    } <= names


async def test_one_accepted_approval_per_revision(
    session: AsyncSession, revision_factory: RevisionFactory, owner: Owner
) -> None:
    rev = await revision_factory()
    session.add(
        Approval(proposal_revision_id=rev.id, owner_id=owner.id, decision="approve", request_id="a")
    )
    await session.commit()
    session.add(
        Approval(
            proposal_revision_id=rev.id, owner_id=owner.id, decision="edit_approve", request_id="b"
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_rejections_do_not_block_approval(
    session: AsyncSession, revision_factory: RevisionFactory, owner: Owner
) -> None:
    rev = await revision_factory()
    for i, decision in enumerate(["reject", "feedback", "approve"]):
        session.add(
            Approval(
                proposal_revision_id=rev.id,
                owner_id=owner.id,
                decision=decision,
                request_id=f"r{i}",
            )
        )
    await session.commit()


async def test_approval_request_id_unique(
    session: AsyncSession, revision_factory: RevisionFactory, owner: Owner
) -> None:
    rev = await revision_factory()
    session.add(
        Approval(proposal_revision_id=rev.id, owner_id=owner.id, decision="reject", request_id="x")
    )
    await session.commit()
    session.add(
        Approval(
            proposal_revision_id=rev.id, owner_id=owner.id, decision="feedback", request_id="x"
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_job_dedup_key_unique(session: AsyncSession) -> None:
    session.add(Job(kind="ingest", dedup_key="k1"))
    await session.commit()
    session.add(Job(kind="ingest", dedup_key="k1"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_job_state_check_constraint(session: AsyncSession) -> None:
    session.add(Job(kind="ingest", dedup_key="k2", state="bogus"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_workflow_status_paused_allowed(
    session: AsyncSession, workflow_factory: WorkflowFactory
) -> None:
    run = await workflow_factory()
    run.status = "paused"
    await session.commit()
    run.status = "nonsense"
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_server_defaults_and_timestamps_are_tz_aware(
    session: AsyncSession, workflow_factory: WorkflowFactory
) -> None:
    run = await workflow_factory()
    await session.refresh(run)
    assert run.id is not None
    assert run.status == "running"
    assert run.step_count == 0
    assert run.created_at.tzinfo is not None


# Created by `AsyncPostgresSaver.setup()` (langgraph-checkpoint-postgres), not by Alembic.
CHECKPOINT_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)


async def test_models_match_migration(db_url: str) -> None:
    """Autogenerate finds no difference between the models and the migrated database.

    LangGraph's checkpoint tables (created by `AsyncPostgresSaver.setup()`, not Alembic) may
    share the database and are ignored.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    def ours(
        obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
    ) -> bool:
        table = name if type_ == "table" else getattr(getattr(obj, "table", None), "name", None)
        return table not in CHECKPOINT_TABLES

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn, opts={"include_object": ours})
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()
    assert diff == []
