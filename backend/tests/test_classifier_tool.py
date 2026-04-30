"""Tests for the classify_destination tool.

The sklearn Pipeline is stubbed with a minimal fake so tests are fast and
hermetic — they exercise our wrapping logic and schema validation without
needing a trained model or any ML dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from agent.tools.classify_destination import classify_destination
from schemas.tools import ClassifyDestinationInput, ClassifyDestinationOutput, ToolError

# ── Stub pipeline ─────────────────────────────────────────────────────────────

CLASSES = ["Adventure", "Budget", "Culture", "Family", "Luxury", "Relaxation"]


class StubPipeline:
    """Minimal sklearn-compatible stub: returns canned predict_proba output."""

    classes_ = np.array(CLASSES)

    def __init__(self, proba: list[float]) -> None:
        assert len(proba) == len(CLASSES)
        assert abs(sum(proba) - 1.0) < 1e-6
        self._proba = np.array([proba])

    def predict_proba(self, X):  # noqa: N803
        return self._proba


class RaisingPipeline:
    """Simulates a broken pipeline that raises on predict_proba."""

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):  # noqa: N803
        raise RuntimeError("model exploded")


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_INPUT = ClassifyDestinationInput(
    climate_zone="Cfa",
    avg_temp_peak_season_c=28.0,
    peak_season_length_months=4,
    terrain_primary="Coastal",
    coastal_access=1,
    unesco_sites_count=1,
    outdoor_activity_score=3,
    daily_cost_bucket=3,
    accommodation_range="Mid-Luxury",
    visa_difficulty=1,
    english_prevalence=2,
    tourism_maturity="Established",
)

# High-confidence proba: Adventure wins at 0.80
HIGH_CONF_PROBA = [0.80, 0.04, 0.04, 0.04, 0.04, 0.04]

# Low-confidence proba: spread across classes, best at 0.25
LOW_CONF_PROBA = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestClassifyDestination:
    def test_high_confidence_returns_output(self) -> None:
        pipeline = StubPipeline(HIGH_CONF_PROBA)
        result = classify_destination(VALID_INPUT, pipeline, threshold=0.60)
        assert isinstance(result, ClassifyDestinationOutput)
        assert result.predicted_style == "Adventure"
        assert abs(result.confidence - 0.80) < 1e-6

    def test_per_class_probs_has_all_classes(self) -> None:
        pipeline = StubPipeline(HIGH_CONF_PROBA)
        result = classify_destination(VALID_INPUT, pipeline, threshold=0.60)
        assert isinstance(result, ClassifyDestinationOutput)
        assert set(result.per_class_probs.keys()) == set(CLASSES)

    def test_per_class_probs_sum_to_one(self) -> None:
        pipeline = StubPipeline(HIGH_CONF_PROBA)
        result = classify_destination(VALID_INPUT, pipeline, threshold=0.60)
        assert isinstance(result, ClassifyDestinationOutput)
        assert abs(sum(result.per_class_probs.values()) - 1.0) < 1e-5

    def test_low_confidence_still_returns_output(self) -> None:
        """Low confidence is a signal for the agent to use LLM fallback — the
        tool itself still returns a valid output; the threshold check is the
        agent's responsibility, not the tool's."""
        pipeline = StubPipeline(LOW_CONF_PROBA)
        result = classify_destination(VALID_INPUT, pipeline, threshold=0.60)
        assert isinstance(result, ClassifyDestinationOutput)
        assert result.confidence < 0.60

    def test_broken_pipeline_returns_tool_error(self) -> None:
        result = classify_destination(VALID_INPUT, RaisingPipeline(), threshold=0.60)
        assert isinstance(result, ToolError)
        assert result.tool == "classify_destination"
        assert "model exploded" in result.error

    def test_confidence_is_max_proba(self) -> None:
        proba = [0.10, 0.10, 0.50, 0.10, 0.10, 0.10]  # Culture wins
        pipeline = StubPipeline(proba)
        result = classify_destination(VALID_INPUT, pipeline, threshold=0.60)
        assert isinstance(result, ClassifyDestinationOutput)
        assert result.predicted_style == "Culture"
        assert abs(result.confidence - 0.50) < 1e-6


class TestClassifyDestinationInput:
    def test_rejects_invalid_climate_zone(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyDestinationInput(
                climate_zone="INVALID",  # type: ignore[arg-type]
                avg_temp_peak_season_c=25,
                peak_season_length_months=3,
                terrain_primary="Urban",
                coastal_access=0,
                unesco_sites_count=0,
                outdoor_activity_score=1,
                daily_cost_bucket=2,
                accommodation_range="Mid-Luxury",
                visa_difficulty=1,
                english_prevalence=2,
                tourism_maturity="Established",
            )

    def test_rejects_out_of_range_temp(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyDestinationInput(
                climate_zone="Cfa",
                avg_temp_peak_season_c=100,  # > 55 limit
                peak_season_length_months=3,
                terrain_primary="Urban",
                coastal_access=0,
                unesco_sites_count=0,
                outdoor_activity_score=1,
                daily_cost_bucket=2,
                accommodation_range="Mid-Luxury",
                visa_difficulty=1,
                english_prevalence=2,
                tourism_maturity="Established",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyDestinationInput(
                climate_zone="Cfa",
                avg_temp_peak_season_c=25,
                peak_season_length_months=3,
                terrain_primary="Urban",
                coastal_access=0,
                unesco_sites_count=0,
                outdoor_activity_score=1,
                daily_cost_bucket=2,
                accommodation_range="Mid-Luxury",
                visa_difficulty=1,
                english_prevalence=2,
                tourism_maturity="Established",
                destination_name="Paris",  # must be rejected — leakage guard  # type: ignore[call-arg]
            )

    def test_rejects_invalid_terrain(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyDestinationInput(
                climate_zone="Cfa",
                avg_temp_peak_season_c=25,
                peak_season_length_months=3,
                terrain_primary="Swamp",  # type: ignore[arg-type]
                coastal_access=0,
                unesco_sites_count=0,
                outdoor_activity_score=1,
                daily_cost_bucket=2,
                accommodation_range="Mid-Luxury",
                visa_difficulty=1,
                english_prevalence=2,
                tourism_maturity="Established",
            )
