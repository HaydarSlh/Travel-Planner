"""Pydantic models for the agent HTTP boundary and internal state.

AgentQueryRequest   — what the client sends to /agent/query
NeedsMoreInfoResponse — returned when TripPreference.is_sufficient() is False
ToolCallRecord      — one tool invocation (persisted + returned in the response)
AgentRunResponse    — final response after a complete run
AgentState          — internal LangGraph state passed between graph nodes
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.trip import TripPreference


# ── HTTP request / response ───────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    webhook_url: str | None = Field(default=None, description="Override the global webhook URL for this run.")


class NeedsMoreInfoResponse(BaseModel):
    """Returned when the extracted preferences are not sufficient to proceed."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["needs_more_info"] = "needs_more_info"
    message: str = Field(..., description="Friendly message asking the user for the missing info.")
    missing_fields: list[str]
    partial_preferences: TripPreference


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any]
    duration_ms: int


class DestinationMeta(BaseModel):
    """Image URL and source link for a single destination, surfaced from the RAG DB."""

    model_config = ConfigDict(extra="forbid")

    name: str
    image_url: str | None = None
    source_url: str | None = None


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    answer: str
    tool_calls: list[ToolCallRecord] = []
    styles_predicted: list[str] = Field(
        default_factory=list,
        description="1-2 travel styles predicted by the heavy LLM for this query.",
    )
    destination_metadata: list[DestinationMeta] = Field(
        default_factory=list,
        description="Image URLs and source links for destinations mentioned in the answer.",
    )
    token_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-step token counts keyed by stage name.",
    )
    created_at: datetime


# ── Internal LangGraph state ──────────────────────────────────────────────────

class AgentState(BaseModel):
    """Mutable state passed between LangGraph nodes.

    Each node reads what it needs and writes its outputs back into this object.
    Using a Pydantic model (not a plain TypedDict) gives us validation at every
    node boundary so bugs surface immediately rather than silently downstream.
    """

    model_config = ConfigDict(extra="forbid")

    # Set by the entry node from the HTTP request
    query: str = ""
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)

    # Set by the intent-parse node
    preferences: TripPreference | None = None

    # Set by the style-prediction node (heavy LLM)
    # 1-2 styles the user is looking for, e.g. ["Adventure", "Budget"]
    styles_wanted: list[str] = Field(default_factory=list)

    # Set by the RAG + classifier nodes
    rag_chunks: list[str] = Field(default_factory=list)
    rag_sources: list[str] = Field(default_factory=list)
    # chunks grouped by style — used to build structured synthesis prompt
    rag_chunks_by_style: dict[str, list[str]] = Field(default_factory=dict)
    # image + source URL per unique destination (keyed by name)
    destination_metadata: dict[str, Any] = Field(default_factory=dict)

    # Set by the live-conditions node
    live_conditions: dict[str, Any] = Field(default_factory=dict)

    # Accumulated tool call records for persistence
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    # Token usage accumulated across all LLM calls in this run
    token_usage: dict[str, Any] = Field(default_factory=dict)

    # Final synthesised answer — set by the synthesis node
    answer: str = ""
