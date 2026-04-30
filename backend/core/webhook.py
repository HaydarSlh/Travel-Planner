"""Async fire-and-forget webhook delivery to Slack.

The Slack Incoming Webhook URL is set via WEBHOOK_URL in .env.
If the URL is blank the function returns immediately — this lets the app run
in development without a Slack workspace.

Delivery contract
─────────────────
• The payload is a Slack Block Kit message (pretty, not raw JSON dump).
• POSTed with httpx.AsyncClient inside a background asyncio.Task so it never
  blocks the user-facing response.
• Tenacity retries up to WEBHOOK_MAX_RETRIES times on transient errors
  (network errors, 429, 5xx) with exponential backoff + jitter.
• After retries are exhausted the error is logged and silently swallowed —
  webhook failure must NEVER surface to the user.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
import tenacity

from core.config import get_settings
from schemas.webhook import WebhookPayload

log = structlog.get_logger()


# ── Slack Block Kit builder ───────────────────────────────────────────────────

def _build_slack_message(payload: WebhookPayload) -> dict:
    """Convert a WebhookPayload into a Slack Block Kit body."""
    styles_text = ", ".join(payload.styles_predicted) if payload.styles_predicted else "—"
    destinations_text = ", ".join(payload.destinations) if payload.destinations else "—"
    tools_text = "\n".join(f"• {t}" for t in payload.tool_summary) if payload.tool_summary else "—"

    # Truncate the answer preview to keep the Slack message readable
    answer_preview = payload.answer[:400].rstrip()
    if len(payload.answer) > 400:
        answer_preview += "…"

    return {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "✈️ New Travel Plan Generated",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*User ID*\n`{payload.user_id}`"},
                    {"type": "mrkdwn", "text": f"*Run ID*\n`{payload.run_id}`"},
                    {"type": "mrkdwn", "text": f"*Styles*\n{styles_text}"},
                    {"type": "mrkdwn", "text": f"*Destinations*\n{destinations_text}"},
                    {"type": "mrkdwn", "text": f"*Tokens*\nin {payload.total_input_tokens} / out {payload.total_output_tokens}"},
                    {"type": "mrkdwn", "text": f"*Timestamp*\n{payload.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Query*\n>{payload.query}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Answer preview*\n{answer_preview}"},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Tools used*\n{tools_text}"},
            },
        ]
    }


# ── Retry policy ──────────────────────────────────────────────────────────────

def _is_webhook_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _make_webhook_retry(max_retries: int) -> tenacity.AsyncRetrying:
    return tenacity.AsyncRetrying(
        retry=tenacity.retry_if_exception(_is_webhook_retryable),
        wait=tenacity.wait_exponential_jitter(initial=1.0, max=30.0),
        stop=tenacity.stop_after_attempt(max_retries),
        reraise=True,
        before_sleep=lambda rs: log.warning(
            "webhook.retry",
            attempt=rs.attempt_number,
            wait=round(rs.next_action.sleep, 2) if rs.next_action else None,
            error=str(rs.outcome.exception()) if rs.outcome else None,
        ),
    )


# ── Core delivery ─────────────────────────────────────────────────────────────

async def _deliver(payload: WebhookPayload, url: str, timeout: int, max_retries: int) -> None:
    """POST the Slack message. Retries on transient errors, swallows on exhaustion."""
    body = _build_slack_message(payload)
    try:
        async for attempt in _make_webhook_retry(max_retries):
            with attempt:
                async with httpx.AsyncClient(timeout=float(timeout)) as http:
                    resp = await http.post(url, json=body)
                    resp.raise_for_status()
        log.info("webhook.delivered", run_id=str(payload.run_id))
    except Exception as exc:  # noqa: BLE001
        log.error(
            "webhook.failed",
            run_id=str(payload.run_id),
            error=str(exc),
        )


# ── Public API ────────────────────────────────────────────────────────────────

def fire_and_forget(payload: WebhookPayload) -> None:
    """Schedule webhook delivery as a background asyncio task.

    Returns immediately. The task runs concurrently and never raises — any
    failure is logged and swallowed so the caller's response is unaffected.
    Call this after the agent run is persisted to the DB.
    """
    settings = get_settings()
    url = settings.webhook_url.strip()
    if not url:
        log.debug("webhook.skipped", reason="WEBHOOK_URL not configured")
        return

    loop = asyncio.get_event_loop()
    loop.create_task(
        _deliver(
            payload,
            url=url,
            timeout=settings.webhook_timeout_seconds,
            max_retries=settings.webhook_max_retries,
        )
    )
