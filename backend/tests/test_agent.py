"""Agent-level tests — currently focused on the intent parser.

The Gemini client is fully stubbed: we control exactly what `generate_content`
returns, so these tests exercise our retry logic and parsing without touching
the network or burning tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from agent.router import parse_intent, predict_styles
from schemas.trip import TripPreference


@dataclass
class FakeUsage:
    prompt_token_count: int = 100
    candidates_token_count: int = 50
    total_token_count: int = 150


@dataclass
class FakeResponse:
    text: str
    usage_metadata: FakeUsage = field(default_factory=FakeUsage)


class FakeAioModels:
    """Stub for `client.aio.models`. Returns canned responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self, *, model: str, contents: str, config: Any
    ) -> FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._responses:
            raise RuntimeError("FakeAioModels exhausted — no more canned responses")
        return FakeResponse(text=self._responses.pop(0))


@dataclass
class FakeAio:
    models: FakeAioModels


@dataclass
class FakeClient:
    aio: FakeAio


def make_client(responses: list[str]) -> FakeClient:
    return FakeClient(aio=FakeAio(models=FakeAioModels(responses)))


VALID_JSON = (
    '{"budget_usd": 1500, "duration_days": 14, "travel_month": 7, '
    '"climate_pref": "warm", "style_keywords": ["hiking", "not touristy"], '
    '"group_type": "couple"}'
)

INVALID_JSON_BAD_MONTH = (
    '{"budget_usd": 1500, "duration_days": 14, "travel_month": 13, '
    '"climate_pref": "warm", "style_keywords": ["hiking"], '
    '"group_type": "unknown"}'
)


class TestParseIntent:
    async def test_valid_first_response_returns_typed_preference(self) -> None:
        client = make_client([VALID_JSON])
        result = await parse_intent(client, "gemini-1.5-flash", "anything")
        assert isinstance(result, TripPreference)
        assert result.budget_usd == 1500
        assert result.travel_month == 7
        assert result.style_keywords == ["hiking", "not touristy"]
        assert len(client.aio.models.calls) == 1

    async def test_invalid_first_then_valid_second_succeeds(self) -> None:
        client = make_client([INVALID_JSON_BAD_MONTH, VALID_JSON])
        result = await parse_intent(client, "gemini-1.5-flash", "anything")
        assert isinstance(result, TripPreference)
        assert len(client.aio.models.calls) == 2

    async def test_two_invalid_responses_raise(self) -> None:
        client = make_client([INVALID_JSON_BAD_MONTH, INVALID_JSON_BAD_MONTH])
        with pytest.raises(ValidationError):
            await parse_intent(client, "gemini-1.5-flash", "anything")
        assert len(client.aio.models.calls) == 2

    async def test_malformed_json_then_valid_succeeds(self) -> None:
        client = make_client(["{not valid json", VALID_JSON])
        result = await parse_intent(client, "gemini-1.5-flash", "anything")
        assert isinstance(result, TripPreference)
        assert len(client.aio.models.calls) == 2

    async def test_retry_includes_error_in_system_prompt(self) -> None:
        """The retry prompt must surface the validation error to the model."""
        client = make_client([INVALID_JSON_BAD_MONTH, VALID_JSON])
        await parse_intent(client, "gemini-1.5-flash", "anything")
        # Second call's system_instruction should mention the error
        second_config = client.aio.models.calls[1]["config"]
        system_text = second_config.system_instruction
        assert "previous response failed validation" in system_text


# Scores within 0.15 → both styles kept; scores far apart → only top style kept
VALID_STYLES_JSON = '{"styles": ["Adventure", "Budget"], "scores": [0.80, 0.75]}'
SINGLE_STYLE_JSON = '{"styles": ["Luxury"], "scores": [0.95]}'
INVALID_STYLE_JSON = '{"styles": ["Hiking"]}'        # "Hiking" is not a valid style
TOO_MANY_STYLES_JSON = '{"styles": ["Adventure", "Budget", "Culture"]}'  # 3 is too many


class TestPredictStyles:
    async def test_valid_two_styles(self) -> None:
        client = make_client([VALID_STYLES_JSON])
        result = await predict_styles(client, "gemini-2.5-flash", TripPreference(
            budget_usd=1200, duration_days=14, style_keywords=["hiking", "cheap"]
        ))
        assert result == ["Adventure", "Budget"]
        assert len(client.aio.models.calls) == 1

    async def test_valid_single_style(self) -> None:
        client = make_client([SINGLE_STYLE_JSON])
        result = await predict_styles(client, "gemini-2.5-flash", TripPreference(
            budget_usd=8000, duration_days=7, style_keywords=["luxury resort"]
        ))
        assert result == ["Luxury"]

    async def test_invalid_style_retries_and_succeeds(self) -> None:
        client = make_client([INVALID_STYLE_JSON, VALID_STYLES_JSON])
        result = await predict_styles(client, "gemini-2.5-flash", TripPreference(
            budget_usd=1200, style_keywords=["hiking"]
        ))
        assert result == ["Adventure", "Budget"]
        assert len(client.aio.models.calls) == 2

    async def test_too_many_styles_retries(self) -> None:
        client = make_client([TOO_MANY_STYLES_JSON, SINGLE_STYLE_JSON])
        result = await predict_styles(client, "gemini-2.5-flash", TripPreference(
            budget_usd=1000, style_keywords=["culture"]
        ))
        assert result == ["Luxury"]
        assert len(client.aio.models.calls) == 2

    async def test_two_invalid_responses_raise(self) -> None:
        client = make_client([INVALID_STYLE_JSON, INVALID_STYLE_JSON])
        with pytest.raises(ValueError):
            await predict_styles(client, "gemini-2.5-flash", TripPreference(
                budget_usd=1000, style_keywords=["hiking"]
            ))
        assert len(client.aio.models.calls) == 2

    async def test_retry_prompt_mentions_error(self) -> None:
        client = make_client([INVALID_STYLE_JSON, VALID_STYLES_JSON])
        await predict_styles(client, "gemini-2.5-flash", TripPreference(
            budget_usd=1000, style_keywords=["hiking"]
        ))
        second_config = client.aio.models.calls[1]["config"]
        assert "unknown styles" in second_config.system_instruction
