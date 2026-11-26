"""SQLAlchemy models for the application schema (everything except LangGraph's checkpoint tables).

Conventions: UUID primary keys come from Postgres (`gen_random_uuid()`), every timestamp is
`timestamptz`, and enum-like columns are `text` with a named CHECK constraint (not a Postgres
ENUM) so later migrations can widen the allowed values with a simple constraint swap.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    cast,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

WORKFLOW_STATUSES = (
    "running",
    "waiting_review",
    "paused",
    "done_no_action",
    "done_with_actions",
    "failed",
    "cancelled",
)
JOB_STATES = ("queued", "running", "done", "failed")
REVIEW_KINDS = ("notification", "question", "proposal")
REVIEW_STATUSES = ("pending", "resolved", "stale", "cancelled")
PROPOSAL_KINDS = ("send_reply", "create_event")
PROPOSAL_STATUSES = (
    "awaiting_approval",
    "approved",
    "rejected",
    "stale",
    "superseded",
    "executed",
    "cancelled_before_execution",
)
REVISION_AUTHORS = ("model", "owner")
APPROVAL_DECISIONS = ("approve", "edit_approve", "reject", "feedback")
ACTION_STATUSES = ("executing", "succeeded", "failed_definite", "outcome_unknown")
USAGE_STATUSES = ("reserved", "settled")


def _in(column: str, values: tuple[str, ...], name: str) -> CheckConstraint:
    allowed = ", ".join(f"'{v}'" for v in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=name)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Base(DeclarativeBase):
    pass


class Owner(Base):
    __tablename__ = "owners"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(Text, nullable=False)
    time_zone: Mapped[str] = mapped_column(Text, nullable=False, server_default="UTC")
    created_at: Mapped[datetime] = _ts()


class EmailThread(Base):
    __tablename__ = "email_threads"
    __table_args__ = (UniqueConstraint("account_id", "provider_thread_id"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    provider_thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    latest_message_id: Mapped[str | None] = mapped_column(Text)
    source_version: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = _ts()


class EmailMessage(Base):
    __tablename__ = "email_messages"
    __table_args__ = (UniqueConstraint("account_id", "provider_message_id"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    thread_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("email_threads.id"), nullable=False)
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    provider_message_id: Mapped[str] = mapped_column(Text, nullable=False)
    rfc_message_id: Mapped[str | None] = mapped_column(Text)
    sender: Mapped[str] = mapped_column(Text, nullable=False)
    to_addrs: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'")
    )
    cc_addrs: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'")
    )
    subject: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    body_text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    body_html: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)


# Embedding models with an HNSW index (migration 0004): model name -> vector dimension.
# Adding a model means a migration with one more partial index, then `python -m app reindex`.
EMBEDDING_DIMENSIONS = {"bge-small-en-v1.5": 384, "voyage-3.5-lite": 512}


def embedding_index_name(model: str) -> str:
    return "ix_chunk_embeddings_hnsw_" + model.replace("-", "_").replace(".", "_")


class MessageChunk(Base):
    """A retrievable piece of one email. Deleting the message deletes its chunks (spec §11)."""

    __tablename__ = "message_chunks"
    __table_args__ = (
        UniqueConstraint("message_id", "chunk_index"),
        Index("ix_message_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_message_chunks_account_received", "account_id", "received_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("email_messages.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("email_threads.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional one-line situating context written by a model (contextual retrieval).
    context_prefix: Mapped[str | None] = mapped_column(Text)
    sender: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(context_prefix, '') || ' ' || text)",
            persisted=True,
        ),
    )
    created_at: Mapped[datetime] = _ts()


class ChunkEmbedding(Base):
    """One embedding of a chunk per model; the column has no fixed dimension so several
    models coexist, each with its own partial HNSW index (see EMBEDDING_DIMENSIONS)."""

    __tablename__ = "chunk_embeddings"
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("message_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(Text, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = _ts()


for _model, _dim in EMBEDDING_DIMENSIONS.items():
    _label = f"embedding_{_dim}"
    Index(
        embedding_index_name(_model),
        cast(ChunkEmbedding.embedding, Vector(_dim)).label(_label),
        postgresql_using="hnsw",
        postgresql_ops={_label: "vector_cosine_ops"},
        postgresql_where=ChunkEmbedding.model == _model,
    )


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    __table_args__ = (_in("status", WORKFLOW_STATUSES, "ck_workflow_runs_status"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    graph_thread_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("email_messages.id"), nullable=False)
    provider_thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    graph_version: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_versions: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="running")
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_code: Mapped[str | None] = mapped_column(Text)
    # Written by the `triage` node: ignore / notify / respond and its short reason.
    classification: Mapped[str | None] = mapped_column(Text)
    triage_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _ts()


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        _in("state", JOB_STATES, "ck_jobs_state"),
        Index("ix_jobs_state_scheduled_at", "state", "scheduled_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    dedup_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    serial_key: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    state: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    scheduled_at: Mapped[datetime] = _ts()
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    lease_owner: Mapped[str | None] = mapped_column(Text)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Proposal(Base):
    __tablename__ = "proposals"
    __table_args__ = (
        _in("kind", PROPOSAL_KINDS, "ck_proposals_kind"),
        _in("status", PROPOSAL_STATUSES, "ck_proposals_status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="awaiting_approval")
    active_revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    # Set by `create_proposal` so a re-run graph node gets the existing proposal back.
    idempotency_key: Mapped[str | None] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = _ts()


class ProposalRevision(Base):
    __tablename__ = "proposal_revisions"
    __table_args__ = (
        UniqueConstraint("proposal_id", "revision"),
        _in("author", REVISION_AUTHORS, "ck_proposal_revisions_author"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    proposal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("proposals.id"), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    source_version: Mapped[str | None] = mapped_column(Text)
    preference_version: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _ts()


class ReviewItem(Base):
    __tablename__ = "review_items"
    __table_args__ = (
        _in("kind", REVIEW_KINDS, "ck_review_items_kind"),
        _in("status", REVIEW_STATUSES, "ck_review_items_status"),
        Index("ix_review_items_status_created_at", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    interrupt_id: Mapped[str | None] = mapped_column(Text)
    proposal_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("proposals.id"))
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = _ts()
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Approval(Base):
    __tablename__ = "approvals"
    __table_args__ = (
        _in("decision", APPROVAL_DECISIONS, "ck_approvals_decision"),
        # At most one accepting decision per revision, however many request ids race.
        Index(
            "uq_approvals_one_accepted_per_revision",
            "proposal_revision_id",
            unique=True,
            postgresql_where=text("decision IN ('approve', 'edit_approve')"),
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    proposal_revision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("proposal_revisions.id"), nullable=False
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("owners.id"), nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    request_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    interrupt_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _ts()


class Action(Base):
    __tablename__ = "actions"
    __table_args__ = (_in("status", ACTION_STATUSES, "ck_actions_status"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    operation_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    approval_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("approvals.id"), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="executing")
    provider_ref: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = _ts()


class PreferenceVersion(Base):
    __tablename__ = "preference_versions"
    __table_args__ = (UniqueConstraint("owner_id", "namespace", "revision"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("owners.id"), nullable=False)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    previous_revision: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = _ts()


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    workflow_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("workflow_runs.id"))
    action_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("actions.id"))
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = _ts()


class WorkflowEvent(Base):
    """Append-only event log; the source for server-sent events."""

    __tablename__ = "workflow_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = _ts()


class LlmUsage(Base):
    __tablename__ = "llm_usage"
    __table_args__ = (_in("status", USAGE_STATUSES, "ck_llm_usage_status"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    workflow_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("workflow_runs.id"))
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cache_write_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="reserved")
    created_at: Mapped[datetime] = _ts()


class AppFlags(Base):
    """Single-row table (id = 1) holding global pause switches."""

    __tablename__ = "app_flags"
    __table_args__ = (CheckConstraint("id = 1", name="ck_app_flags_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, server_default="1")
    ingestion_paused: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    outbound_paused: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )


class MockOutbox(Base):
    __tablename__ = "mock_outbox"

    id: Mapped[uuid.UUID] = _uuid_pk()
    operation_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _ts()


class MockCalendarEvent(Base):
    __tablename__ = "mock_calendar_events"

    event_id: Mapped[str] = mapped_column(Text, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _ts()
