"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    # Enable pgvector extension — idempotent, safe to run multiple times
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "agent_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("total_input_tokens", sa.Integer, default=0, nullable=False),
        sa.Column("total_output_tokens", sa.Integer, default=0, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"])

    op.create_table(
        "tool_calls",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tool_name", sa.String(64), nullable=False),
        sa.Column("input_json", JSONB, nullable=False),
        sa.Column("output_json", JSONB, nullable=False),
        sa.Column("duration_ms", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_tool_calls_run_id", "tool_calls", ["run_id"])

    op.create_table(
        "destinations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("country", sa.String(128)),
        sa.Column("climate_zone", sa.String(8), nullable=False),
        sa.Column("avg_temp_peak_season_c", sa.Float, nullable=False),
        sa.Column("peak_season_length_months", sa.Integer, nullable=False),
        sa.Column("terrain_primary", sa.String(32), nullable=False),
        sa.Column("coastal_access", sa.Boolean, nullable=False),
        sa.Column("unesco_sites_count", sa.Integer, nullable=False),
        sa.Column("outdoor_activity_score", sa.Integer, nullable=False),
        sa.Column("daily_cost_bucket", sa.Integer, nullable=False),
        sa.Column("accommodation_range", sa.String(32), nullable=False),
        sa.Column("visa_difficulty", sa.Integer, nullable=False),
        sa.Column("english_prevalence", sa.Integer, nullable=False),
        sa.Column("tourism_maturity", sa.String(32), nullable=False),
        sa.Column("hand_label", sa.String(32), nullable=False),
        sa.Column("predicted_label", sa.String(32)),
    )
    op.create_index("ix_destinations_name", "destinations", ["name"])

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("destination_id", UUID(as_uuid=True), sa.ForeignKey("destinations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_url", sa.String(1024), nullable=False),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_documents_destination_id", "documents", ["destination_id"])

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("section_group", sa.String(32)),
        sa.Column("styles", sa.String(128)),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("destinations")
    op.drop_table("tool_calls")
    op.drop_table("agent_runs")
    op.drop_table("users")
