"""SQLAlchemy 2.x ORM models — the entire relational schema in one file.

All tables share a single `Base` and metadata. Models are dataclass-style
(`Mapped`/`mapped_column`) for full type hint support.

Six tables:
  - users          authenticated accounts
  - agent_runs     one row per /agent/query call
  - tool_calls     one row per tool invocation inside an agent run
  - destinations   the 150-row hand-labeled training set + features
  - documents      raw destination prose (Wikivoyage etc.)
  - chunks         tokenized chunks with pgvector embeddings
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Embedding dimension matches gemini-embedding-001 output.
# Changing this requires a migration + full re-embed of all chunks.
EMBEDDING_DIM = 3072


class Base(DeclarativeBase):
    """Declarative base. All ORM models inherit from this."""


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# Users
# ──────────────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = _created_at()

    runs: Mapped[list[AgentRun]] = relationship(back_populates="user", cascade="all, delete-orphan")


# ──────────────────────────────────────────────────────────────────────────────
# Agent runs and tool calls
# ──────────────────────────────────────────────────────────────────────────────
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    user: Mapped[User] = relationship(back_populates="runs")
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    input_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    run: Mapped[AgentRun] = relationship(back_populates="tool_calls")


# ──────────────────────────────────────────────────────────────────────────────
# Destinations (12 features + label)
# ──────────────────────────────────────────────────────────────────────────────
class Destination(Base):
    __tablename__ = "destinations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    country: Mapped[str | None] = mapped_column(String(128))

    # ── 12 features (locked in CLAUDE.md) ─────────────────────────────────────
    climate_zone: Mapped[str] = mapped_column(String(8), nullable=False)             # Köppen
    avg_temp_peak_season_c: Mapped[float] = mapped_column(Float, nullable=False)
    peak_season_length_months: Mapped[int] = mapped_column(Integer, nullable=False)
    terrain_primary: Mapped[str] = mapped_column(String(32), nullable=False)
    coastal_access: Mapped[bool] = mapped_column(Boolean, nullable=False)
    unesco_sites_count: Mapped[int] = mapped_column(Integer, nullable=False)
    outdoor_activity_score: Mapped[int] = mapped_column(Integer, nullable=False)     # 1-3
    daily_cost_bucket: Mapped[int] = mapped_column(Integer, nullable=False)          # 1-4
    accommodation_range: Mapped[str] = mapped_column(String(32), nullable=False)
    visa_difficulty: Mapped[int] = mapped_column(Integer, nullable=False)            # 1-3
    english_prevalence: Mapped[int] = mapped_column(Integer, nullable=False)         # 1-3
    tourism_maturity: Mapped[str] = mapped_column(String(32), nullable=False)

    # ── Labels ────────────────────────────────────────────────────────────────
    hand_label: Mapped[str] = mapped_column(String(32), nullable=False)              # ground truth
    predicted_label: Mapped[str | None] = mapped_column(String(32))                  # classifier output

    # ── Media ─────────────────────────────────────────────────────────────────
    image_url: Mapped[str | None] = mapped_column(String(1024))                      # representative photo

    documents: Mapped[list[Document]] = relationship(
        back_populates="destination", cascade="all, delete-orphan"
    )


# ──────────────────────────────────────────────────────────────────────────────
# RAG documents and chunks
# ──────────────────────────────────────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    destination_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("destinations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    destination: Mapped[Destination] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    # Semantic group this chunk belongs to (overview/activities/practical/logistics/stay/safety/other)
    section_group: Mapped[str | None] = mapped_column(String(32))
    # Comma-separated travel styles for this destination, e.g. "Adventure,Family"
    styles: Mapped[str | None] = mapped_column(String(128))

    document: Mapped[Document] = relationship(back_populates="chunks")
