"""Pydantic schema tests — valid and invalid inputs at every boundary.

If our schemas are wrong, our agent is wrong. These tests are cheap and
high-value: they verify that bad data raises ValidationError at the edge,
not deep inside business logic.
"""

import pytest
from pydantic import ValidationError

from schemas.trip import TripPreference


class TestTripPreference:
    def test_full_valid_input(self) -> None:
        prefs = TripPreference(
            budget_usd=1500,
            duration_days=14,
            travel_month=7,
            climate_pref="warm",
            style_keywords=["hiking", "not touristy"],
            group_type="couple",
        )
        assert prefs.budget_usd == 1500
        assert prefs.duration_days == 14
        assert prefs.travel_month == 7
        assert prefs.climate_pref == "warm"
        assert prefs.style_keywords == ["hiking", "not touristy"]
        assert prefs.group_type == "couple"

    def test_minimal_input_uses_defaults(self) -> None:
        prefs = TripPreference()
        assert prefs.budget_usd is None
        assert prefs.duration_days is None
        assert prefs.travel_month is None
        assert prefs.climate_pref == "any"
        assert prefs.style_keywords == []
        assert prefs.group_type == "unknown"

    def test_partial_input_keeps_unspecified_as_none(self) -> None:
        prefs = TripPreference(travel_month=7, climate_pref="warm")
        assert prefs.travel_month == 7
        assert prefs.climate_pref == "warm"
        assert prefs.budget_usd is None
        assert prefs.duration_days is None

    @pytest.mark.parametrize("bad_month", [0, 13, -1, 100])
    def test_invalid_travel_month_rejected(self, bad_month: int) -> None:
        with pytest.raises(ValidationError):
            TripPreference(travel_month=bad_month)

    @pytest.mark.parametrize("bad_duration", [0, -1, -100])
    def test_invalid_duration_rejected(self, bad_duration: int) -> None:
        with pytest.raises(ValidationError):
            TripPreference(duration_days=bad_duration)

    def test_excessive_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TripPreference(duration_days=400)

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TripPreference(budget_usd=-50)

    def test_invalid_climate_pref_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TripPreference(climate_pref="freezing")  # type: ignore[arg-type]

    def test_invalid_group_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TripPreference(group_type="business")  # type: ignore[arg-type]

    def test_extra_fields_rejected(self) -> None:
        # extra="forbid" must catch typos / hallucinated fields from the LLM
        with pytest.raises(ValidationError):
            TripPreference.model_validate(
                {"budget_usd": 1000, "destination": "Paris"}
            )

    def test_string_budget_coerced_to_float(self) -> None:
        prefs = TripPreference.model_validate({"budget_usd": "1500"})
        assert prefs.budget_usd == 1500.0


class TestTripPreferenceSufficiency:
    def test_sufficient_with_budget_and_style_keywords(self) -> None:
        prefs = TripPreference(budget_usd=1500, style_keywords=["hiking"])
        assert prefs.is_sufficient() is True

    def test_sufficient_with_duration_and_climate(self) -> None:
        prefs = TripPreference(duration_days=14, climate_pref="warm")
        assert prefs.is_sufficient() is True

    def test_insufficient_with_no_logistics(self) -> None:
        # Has style signal but no budget or duration
        prefs = TripPreference(style_keywords=["hiking"], climate_pref="warm")
        assert prefs.is_sufficient() is False

    def test_insufficient_with_no_style_signal(self) -> None:
        # Has budget but climate is "any" and no keywords
        prefs = TripPreference(budget_usd=1500)
        assert prefs.is_sufficient() is False

    def test_completely_empty_is_insufficient(self) -> None:
        assert TripPreference().is_sufficient() is False

    def test_missing_fields_lists_logistics_when_absent(self) -> None:
        prefs = TripPreference(style_keywords=["beach"])
        missing = prefs.missing_fields()
        assert any("budget" in m or "duration" in m for m in missing)

    def test_missing_fields_lists_style_when_absent(self) -> None:
        prefs = TripPreference(budget_usd=1000)
        missing = prefs.missing_fields()
        assert any("kind of trip" in m for m in missing)

    def test_missing_fields_empty_when_sufficient(self) -> None:
        prefs = TripPreference(budget_usd=1500, style_keywords=["culture"])
        assert prefs.missing_fields() == []
