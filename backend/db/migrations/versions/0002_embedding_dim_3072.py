"""change embedding vector dimension from 768 to 3072

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

OLD_DIM = 768
NEW_DIM = 3072


def upgrade() -> None:
    # Drop the old 768-dim column and recreate it at 3072.
    # Any existing chunk rows will be deleted first — ingest must be re-run.
    op.execute("DELETE FROM chunks")
    op.drop_column("chunks", "embedding")
    op.add_column("chunks", sa.Column("embedding", Vector(NEW_DIM), nullable=False))


def downgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.drop_column("chunks", "embedding")
    op.add_column("chunks", sa.Column("embedding", Vector(OLD_DIM), nullable=False))
