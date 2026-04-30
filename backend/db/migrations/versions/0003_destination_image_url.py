"""Add image_url to destinations table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "destinations",
        sa.Column("image_url", sa.String(1024), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("destinations", "image_url")
