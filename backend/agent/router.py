"""Gemini client factory and the intent-parsing agent node.

Two separate concerns live here:
  1. `build_gemini_client(api_key)` — pure factory used by the FastAPI lifespan
     handler to construct exactly one client per process. Never called per-request.
  2. `parse_intent(...)` — the cheap-LLM step that turns free text into a
     validated `TripPreference`. One automatic retry on validation failure
     (the error is fed back to the model). Token usage is logged via structlog
     so the README cost section has its data point.

Rate-limit / overload handling
───────────────────────────────
Gemini raises `ServerError(503)` for "high demand" and `ClientError(429)` for
quota exhaustion.  Both are transient and safe to retry.

Strategy:
  1. tenacity retries the raw HTTP call up to 4 times with exponential backoff
     + full jitter (0.5 s → 1 s → 2 s → 4 s, ± random).
  2. If all retries on the primary model fail we fall back to the next model in
     the fallback chain before giving up.  The chain is:
       strong: gemini-2.5-flash  → gemini-2.5-flash-lite
       cheap:  gemini-2.5-flash-lite → gemini-2.5-flash-lite  (same, no cheaper option)
  3. 4xx errors other than 429 (bad request, auth, etc.) are NOT retried — they
     won't fix themselves.
"""

from __future__ import annotations

import json

import structlog
import tenacity
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from agent.prompts import (
    INTENT_PARSER_PROMPT,
    INTENT_PARSER_RETRY_PROMPT,
    STYLE_PREDICTOR_PROMPT,
    STYLE_PREDICTOR_RETRY_PROMPT,
)
from schemas.trip import TripPreference

_VALID_STYLES = {"Adventure", "Relaxation", "Culture", "Budget", "Luxury", "Family"}

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Fallback model chains — ordered from strongest to most available
# ---------------------------------------------------------------------------
_STRONG_FALLBACK_CHAIN = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
_CHEAP_FALLBACK_CHAIN  = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient Gemini errors that are worth retrying."""
    if isinstance(exc, genai_errors.ServerError):
        # 503 UNAVAILABLE, 500 INTERNAL — always transient
        return True
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        # 429 RESOURCE_EXHAUSTED — quota, retryable with backoff
        return True
    return False


def _make_retry() -> tenacity.AsyncRetrying:
    """Return a tenacity retry policy for a single Gemini call."""
    return tenacity.AsyncRetrying(
        retry=tenacity.retry_if_exception(_is_retryable),
        wait=tenacity.wait_exponential_jitter(initial=0.5, max=8.0),
        stop=tenacity.stop_after_attempt(4),
        reraise=True,
        before_sleep=lambda rs: log.warning(
            "gemini.retry",
            attempt=rs.attempt_number,
            wait=round(rs.next_action.sleep, 2) if rs.next_action else None,
            error=str(rs.outcome.exception()) if rs.outcome else None,
        ),
    )


def build_gemini_client(api_key: str) -> genai.Client:
    """Construct a Gemini client. Call this once per process from the lifespan."""
    return genai.Client(api_key=api_key)


def _build_config(system_instruction: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=0.0,
    )


async def _call_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """Call Gemini once, retrying on transient errors with exponential backoff."""
    async for attempt in _make_retry():
        with attempt:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response
    # tenacity re-raises on exhaustion; this line is unreachable but satisfies type checker
    raise RuntimeError("Retry loop exited without returning")  # pragma: no cover


async def _call_with_fallback(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    fallback_chain: list[str],
) -> tuple[types.GenerateContentResponse, str]:
    """Try each model in the fallback chain until one succeeds.

    Returns (response, model_that_succeeded).
    """
    # Build the full chain: requested model first, then fallbacks (deduped)
    chain = [model] + [m for m in fallback_chain if m != model]

    last_exc: BaseException | None = None
    for candidate in chain:
        try:
            resp = await _call_with_retry(client, candidate, contents, config)
            if candidate != model:
                log.warning("gemini.fallback_used", original=model, used=candidate)
            return resp, candidate
        except (genai_errors.ServerError, genai_errors.ClientError) as exc:
            if getattr(exc, "code", None) not in (429, 500, 503):
                raise  # non-transient — don't try fallbacks
            log.warning("gemini.model_unavailable", model=candidate, code=getattr(exc, "code", "?"))
            last_exc = exc

    raise last_exc or RuntimeError("All models in fallback chain failed")


async def _generate_json(
    client: genai.Client,
    model: str,
    user_query: str,
    system_instruction: str,
    fallback_chain: list[str] | None = None,
) -> tuple[str, types.GenerateContentResponse]:
    """Call Gemini and return (raw_text, response).

    Retries transient errors and walks the fallback chain if needed.
    """
    config = _build_config(system_instruction)
    chain = fallback_chain or _CHEAP_FALLBACK_CHAIN
    response, used_model = await _call_with_fallback(client, model, user_query, config, chain)
    if response.text is None:
        raise RuntimeError("Gemini returned an empty response")
    return response.text, response


def _log_usage(stage: str, response: types.GenerateContentResponse, model: str) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    log.info(
        "llm.usage",
        stage=stage,
        model=model,
        prompt_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
        total_tokens=getattr(usage, "total_token_count", None),
    )


def _parse(text: str) -> TripPreference:
    data = json.loads(text)
    return TripPreference.model_validate(data)


def _parse_styles(text: str) -> list[str]:
    data = json.loads(text)
    styles = data.get("styles", [])
    if not isinstance(styles, list) or not styles:
        raise ValueError("'styles' must be a non-empty list")
    if len(styles) > 2:
        raise ValueError(f"at most 2 styles allowed, got {len(styles)}")
    invalid = [s for s in styles if s not in _VALID_STYLES]
    if invalid:
        raise ValueError(f"unknown styles: {invalid}")
    return styles


def _parse_styles_with_scores(text: str) -> list[tuple[str, float]]:
    """Decode style-prediction JSON and return (style, confidence) pairs."""
    data = json.loads(text)
    styles = data.get("styles", [])
    if not isinstance(styles, list) or not styles:
        raise ValueError("'styles' must be a non-empty list")
    if len(styles) > 2:
        raise ValueError(f"at most 2 styles allowed, got {len(styles)}")
    invalid = [s for s in styles if s not in _VALID_STYLES]
    if invalid:
        raise ValueError(f"unknown styles: {invalid}")

    raw_scores: list = data.get("scores", [])
    if len(raw_scores) == len(styles):
        try:
            scores = [float(s) for s in raw_scores]
        except (TypeError, ValueError):
            scores = [1.0] + [0.0] * (len(styles) - 1)
    else:
        scores = [1.0] + [0.0] * (len(styles) - 1)

    return list(zip(styles, scores))


_STYLE_PROXIMITY_THRESHOLD = 0.15


async def predict_styles(
    client: genai.Client,
    model: str,
    preferences: TripPreference,
) -> list[str]:
    """Predict 1-2 travel styles. Two styles only when scores are within 0.15."""
    scored = await predict_styles_scored(client, model, preferences)
    if len(scored) == 2:
        _, s1 = scored[0]
        _, s2 = scored[1]
        if abs(s1 - s2) > _STYLE_PROXIMITY_THRESHOLD:
            log.info(
                "style.predict.dropped_weak_style",
                kept=scored[0][0],
                dropped=scored[1][0],
                gap=round(abs(s1 - s2), 3),
            )
            return [scored[0][0]]
    return [s for s, _ in scored]


async def predict_styles_scored(
    client: genai.Client,
    model: str,
    preferences: TripPreference,
) -> list[tuple[str, float]]:
    """Like predict_styles but returns (style, confidence) pairs.

    Retries once on validation failure. Raises ValueError after two failures.
    """
    prefs_text = preferences.model_dump_json(indent=2)
    system = STYLE_PREDICTOR_PROMPT.format(preferences=prefs_text)

    log.info("style.predict.start", model=model)

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.0,
    )

    # Attempt 1 — with retry + fallback
    try:
        resp, used = await _call_with_fallback(
            client, model, "Predict the travel styles.", config, _STRONG_FALLBACK_CHAIN
        )
        _log_usage("style.predict.attempt1", resp, used)
        result = _parse_styles_with_scores(resp.text or "")
        log.info("style.predict.success", attempt=1, styles=[s for s, _ in result])
        return result
    except (ValueError, json.JSONDecodeError) as first_error:
        log.warning("style.predict.invalid", attempt=1, error=str(first_error))
        retry_system = system + "\n\n" + STYLE_PREDICTOR_RETRY_PROMPT.format(error=str(first_error))

    # Attempt 2
    retry_config = types.GenerateContentConfig(
        system_instruction=retry_system,
        response_mime_type="application/json",
        temperature=0.0,
    )
    resp, used = await _call_with_fallback(
        client, model, "Predict the travel styles.", retry_config, _STRONG_FALLBACK_CHAIN
    )
    _log_usage("style.predict.attempt2", resp, used)
    result = _parse_styles_with_scores(resp.text or "")
    log.info("style.predict.success", attempt=2, styles=[s for s, _ in result])
    return result


async def parse_intent(
    client: genai.Client,
    model: str,
    user_query: str,
) -> TripPreference:
    """Extract a TripPreference from a user's free-text query.

    Retries once on ValidationError (feeds error back to model).
    503/429 are handled inside _generate_json via tenacity + fallback chain.
    """
    log.info("intent.parse.start", model=model, query_length=len(user_query))

    # Attempt 1
    try:
        text, response = await _generate_json(
            client, model, user_query, INTENT_PARSER_PROMPT, _CHEAP_FALLBACK_CHAIN
        )
        _log_usage("intent.parse.attempt1", response, model)
        result = _parse(text)
        log.info("intent.parse.success", attempt=1)
        return result
    except (ValidationError, json.JSONDecodeError) as first_error:
        log.warning("intent.parse.invalid", attempt=1, error=str(first_error))
        retry_system_prompt = (
            INTENT_PARSER_PROMPT
            + "\n\n"
            + INTENT_PARSER_RETRY_PROMPT.format(error=str(first_error))
        )

    # Attempt 2 — validation error fed back into prompt
    text, response = await _generate_json(
        client, model, user_query, retry_system_prompt, _CHEAP_FALLBACK_CHAIN
    )
    _log_usage("intent.parse.attempt2", response, model)
    try:
        result = _parse(text)
        log.info("intent.parse.success", attempt=2)
        return result
    except (ValidationError, json.JSONDecodeError) as second_error:
        log.error("intent.parse.failed", attempts=2, error=str(second_error))
        raise
