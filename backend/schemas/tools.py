"""Pydantic schemas for all three agent tools.

Input schemas are validated before the tool executes — bad LLM-supplied values
raise ValidationError, which the agent loop catches and feeds back to the LLM
as a ToolError for retry.

Output schemas give the agent loop a typed contract on what each tool returns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolError(BaseModel):
    """Structured error returned by any tool — never raises into the agent loop."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    error: str


# ── ClassifyDestination ───────────────────────────────────────────────────────

# Exactly the 12 destination features. No destination name, no country —
# those are identifiers that would leak label information to the model.
_CLIMATE_ZONES = Literal["Af", "Am", "Aw", "BSh", "BSk", "BWh", "BWk", "Cfa", "Cfb", "Cfc", "Csa", "Csb", "Csc", "Cwa", "Cwb", "Dfa", "Dfb", "Dfc", "ET"]
_TERRAIN = Literal["Urban", "Coastal", "Mountain", "Desert", "Jungle", "Island", "Mixed"]
_ACCOMMODATION = Literal["Hostel-Mid", "Mid-Luxury", "Full Range", "Luxury Only"]
_TOURISM_MATURITY = Literal["Developing", "Established", "Overtouristed"]


class ClassifyDestinationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    climate_zone: _CLIMATE_ZONES
    avg_temp_peak_season_c: float = Field(..., ge=-30, le=55)
    peak_season_length_months: int = Field(..., ge=1, le=12)
    terrain_primary: _TERRAIN
    coastal_access: int = Field(..., ge=0, le=1, description="1 = has coast, 0 = landlocked")
    unesco_sites_count: int = Field(..., ge=0)
    outdoor_activity_score: int = Field(..., ge=0, le=5)
    daily_cost_bucket: int = Field(..., ge=1, le=5, description="1=budget … 5=ultra-luxury")
    accommodation_range: _ACCOMMODATION
    visa_difficulty: int = Field(..., ge=1, le=3, description="1=easy, 2=moderate, 3=hard")
    english_prevalence: int = Field(..., ge=1, le=3, description="1=low, 2=moderate, 3=high")
    tourism_maturity: _TOURISM_MATURITY


class ClassifyDestinationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_style: Literal["Adventure", "Relaxation", "Culture", "Budget", "Luxury", "Family"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    per_class_probs: dict[str, float]


# ── RAGRetrieve ───────────────────────────────────────────────────────────────

class RAGRetrieveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    travel_style: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGRetrieveOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[str]
    source_documents: list[str]
    # Parallel to source_documents — image URL for each chunk's destination (may be None)
    image_urls: list[str | None] = Field(default_factory=list)
    # Parallel to source_documents — Wikivoyage / source URL for each chunk (may be None)
    source_urls: list[str | None] = Field(default_factory=list)


# ── LiveConditions ────────────────────────────────────────────────────────────

class LiveConditionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_name: str = Field(..., min_length=1)
    travel_month: int = Field(..., ge=1, le=12)


class LiveConditionsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    avg_temp_c: float | None = None
    precipitation_mm: float | None = None
    weather_summary: str = ""
    flight_available: bool = False
    estimated_flight_cost_usd: float | None = None
