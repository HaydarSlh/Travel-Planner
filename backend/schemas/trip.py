"""Pydantic schema for the structured trip preference extracted from free-text user input.

This is the output of the cheap-LLM intent-parsing step. Validated before the agent
loop proceeds — if the LLM returns garbage, Pydantic raises and the agent retries.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TripPreference(BaseModel):
    """Structured representation of what the user wants from their trip."""

    model_config = ConfigDict(extra="forbid")

    budget_usd: float | None = Field(
        default=None,
        ge=0,
        description="Total trip budget in USD. None when the user did not specify.",
    )
    duration_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Trip length in days (e.g. '2 weeks' -> 14). None when unspecified.",
    )
    travel_month: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Calendar month 1-12. None when unspecified.",
    )
    climate_pref: Literal["warm", "cold", "mild", "any"] = Field(
        default="any",
        description="User's climate preference. 'any' when not mentioned.",
    )
    style_keywords: list[str] = Field(
        default_factory=list,
        description="Free-form descriptors like 'hiking', 'not touristy'.",
    )
    group_type: Literal["solo", "couple", "family", "group", "unknown"] = Field(
        default="unknown",
        description="Travel group composition. 'unknown' when not mentioned.",
    )

    def is_sufficient(self) -> bool:
        """Return True when the preference has enough signal to proceed.

        The agent needs at least budget OR duration, plus at least one style
        signal (climate_pref != 'any' OR non-empty style_keywords). Without
        these the heavy LLM cannot make a meaningful style prediction and the
        RAG filter will be too broad.
        """
        has_logistics = self.budget_usd is not None or self.duration_days is not None
        has_style_signal = self.climate_pref != "any" or len(self.style_keywords) > 0
        return has_logistics and has_style_signal

    def missing_fields(self) -> list[str]:
        """Return human-readable names of the fields still needed."""
        missing = []
        if self.budget_usd is None and self.duration_days is None:
            missing.append("budget or trip duration")
        if self.climate_pref == "any" and not self.style_keywords:
            missing.append("what kind of trip you want (e.g. hiking, relaxation, culture)")
        return missing
