"""Add workflow_runs.classification and workflow_runs.triage_reason (nullable).

Revision ID: 0003
Revises: 0002

The `triage` node stores its outcome here so the review queue and workflow lists can show
why an email was routed the way it was without reading LangGraph checkpoints.
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workflow_runs", sa.Column("classification", sa.Text, nullable=True))
    op.add_column("workflow_runs", sa.Column("triage_reason", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("workflow_runs", "triage_reason")
    op.drop_column("workflow_runs", "classification")
