"""Add proposals.idempotency_key (nullable, unique).

Revision ID: 0002
Revises: 0001

`create_proposal` stores the caller's idempotency key here (rather than in the payload, so the
payload hash covers the payload only) and returns the existing proposal when a graph node
re-runs with the same key.
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("proposals", sa.Column("idempotency_key", sa.Text, nullable=True))
    op.create_unique_constraint("proposals_idempotency_key_key", "proposals", ["idempotency_key"])


def downgrade() -> None:
    op.drop_constraint("proposals_idempotency_key_key", "proposals", type_="unique")
    op.drop_column("proposals", "idempotency_key")
