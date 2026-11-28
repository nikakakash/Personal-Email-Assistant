"""Retrieval index: pgvector, message_chunks (with a generated tsvector) and chunk_embeddings.

Revision ID: 0004
Revises: 0003

`chunk_embeddings.embedding` has no fixed dimension so several embedding models can coexist;
each model gets a partial HNSW index over a cast to its dimension (the pattern pgvector
documents for mixed dimensions). Deleting an email deletes its chunks and embeddings.
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

# Frozen copy (migrations must not import app code): model name -> dimension.
_MODELS = {"bge-small-en-v1.5": 384, "voyage-3.5-lite": 512}


def _index_name(model: str) -> str:
    return "ix_chunk_embeddings_hnsw_" + model.replace("-", "_").replace(".", "_")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "message_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("account_id", sa.Text, nullable=False),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("email_messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "thread_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("email_threads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("context_prefix", sa.Text),
        sa.Column("sender", sa.Text, nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.Text, nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR,
            sa.Computed(
                "to_tsvector('english', coalesce(context_prefix, '') || ' ' || text)",
                persisted=True,
            ),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("message_id", "chunk_index"),
    )
    op.create_index("ix_message_chunks_tsv", "message_chunks", ["tsv"], postgresql_using="gin")
    op.create_index(
        "ix_message_chunks_account_received", "message_chunks", ["account_id", "received_at"]
    )
    op.create_table(
        "chunk_embeddings",
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("message_chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model", sa.Text, primary_key=True),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    for model, dim in _MODELS.items():
        op.execute(
            f"CREATE INDEX {_index_name(model)} ON chunk_embeddings "
            f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops) "
            f"WHERE model = '{model}'"
        )


def downgrade() -> None:
    op.drop_table("chunk_embeddings")
    op.drop_table("message_chunks")
    # The extension is left installed: other database objects may depend on it.
