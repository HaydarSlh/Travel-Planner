"""Outbound webhook payload schema.

WebhookPayload is validated before every HTTP POST fires so the receiver
always gets a well-formed body. The Slack Block Kit message is built
separately in core.webhook — this schema only captures the raw data.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class WebhookPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    user_id: uuid.UUID
    query: str
    answer: str
    styles_predicted: list[str] = Field(default_factory=list)
    destinations: list[str] = Field(
        default_factory=list,
        description="Destination names extracted from the run (from destination_metadata).",
    )
    tool_summary: list[str] = Field(
        default_factory=list,
        description="One line per tool call: '<tool_name> completed in <ms>ms'.",
    )
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    timestamp: datetime
