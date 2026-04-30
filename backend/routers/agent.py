"""Agent routes.

POST /agent/query        — run the travel planner agent, returns JSON response
POST /agent/query/stream — same, but streams SSE tokens during synthesis
GET  /agent/history      — list past AgentRuns for the current user
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from agent.graph import run_agent, run_agent_stream
from core.deps import CheapLlmDep, CurrentUserDep, DbDep, SettingsDep, StrongLlmDep
from core.webhook import fire_and_forget as webhook_fire
from db.models import AgentRun, ToolCall
from schemas.agent import (
    AgentQueryRequest,
    AgentRunResponse,
    AgentState,
    DestinationMeta,
    NeedsMoreInfoResponse,
    ToolCallRecord,
)
from schemas.webhook import WebhookPayload

router = APIRouter(prefix="/agent", tags=["agent"])
log = structlog.get_logger()


def _build_webhook_payload(
    run_id: uuid.UUID,
    user_id,
    query: str,
    answer: str,
    tool_calls: list[ToolCallRecord],
    token_usage: dict,
    styles_predicted: list[str],
    destination_names: list[str],
) -> WebhookPayload:
    total_input = sum(
        v.get("prompt_tokens") or 0
        for v in token_usage.values()
        if isinstance(v, dict)
    )
    total_output = sum(
        v.get("output_tokens") or 0
        for v in token_usage.values()
        if isinstance(v, dict)
    )
    return WebhookPayload(
        run_id=run_id,
        user_id=user_id,
        query=query,
        answer=answer,
        styles_predicted=styles_predicted,
        destinations=destination_names,
        tool_summary=[f"{tc.tool_name} completed in {tc.duration_ms}ms" for tc in tool_calls],
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        timestamp=datetime.now(timezone.utc),
    )


@router.post(
    "/query",
    response_model=AgentRunResponse | NeedsMoreInfoResponse,
)
async def query(
    body: AgentQueryRequest,
    request: Request,
    user: CurrentUserDep,
    db: DbDep,
    cheap_client: CheapLlmDep,
    strong_client: StrongLlmDep,
    settings: SettingsDep,
) -> AgentRunResponse | NeedsMoreInfoResponse:
    log.info("agent.query.start", user_id=str(user.id), query_len=len(body.query))

    result = await run_agent(
        query=body.query,
        cheap_client=cheap_client,
        strong_client=strong_client,
        cheap_model=settings.cheap_model,
        strong_model=settings.strong_model,
        db=db,
        embedding_model=settings.embedding_model,
        top_k=settings.rag_top_k,
    )

    # Not enough info — return early without persisting a run
    if isinstance(result, NeedsMoreInfoResponse):
        log.info("agent.query.needs_more_info", user_id=str(user.id))
        return result

    # Persist AgentRun
    total_input = sum(
        v.get("prompt_tokens") or 0
        for v in result.token_usage.values()
        if isinstance(v, dict)
    )
    total_output = sum(
        v.get("output_tokens") or 0
        for v in result.token_usage.values()
        if isinstance(v, dict)
    )

    run = AgentRun(
        id=result.run_id,
        user_id=user.id,
        query=body.query,
        answer=result.answer,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )
    db.add(run)

    for tc in result.tool_calls:
        db.add(ToolCall(
            id=uuid.uuid4(),
            run_id=result.run_id,
            tool_name=tc.tool_name,
            input_json=tc.input,
            output_json=tc.output,
            duration_ms=tc.duration_ms,
        ))

    await db.commit()
    log.info(
        "agent.query.done",
        run_id=str(result.run_id),
        answer_len=len(result.answer),
        tool_calls=len(result.tool_calls),
    )

    dest_meta = [
        DestinationMeta(
            name=name,
            image_url=meta.get("image_url"),
            source_url=meta.get("source_url"),
        )
        for name, meta in result.destination_metadata.items()
    ]

    webhook_fire(_build_webhook_payload(
        run_id=result.run_id,
        user_id=user.id,
        query=body.query,
        answer=result.answer,
        tool_calls=result.tool_calls,
        token_usage=result.token_usage,
        styles_predicted=result.styles_wanted,
        destination_names=list(result.destination_metadata.keys()),
    ))

    return AgentRunResponse(
        run_id=result.run_id,
        answer=result.answer,
        tool_calls=result.tool_calls,
        styles_predicted=result.styles_wanted,
        destination_metadata=dest_meta,
        token_usage=result.token_usage,
        created_at=datetime.now(timezone.utc),
    )


async def _persist_run(
    db,
    run_id: uuid.UUID,
    user_id,
    query: str,
    answer: str,
    tool_calls: list[ToolCallRecord],
    token_usage: dict,
) -> None:
    """Write AgentRun + ToolCall rows after a streaming run completes."""
    total_input = sum(
        v.get("prompt_tokens") or 0
        for v in token_usage.values()
        if isinstance(v, dict)
    )
    total_output = sum(
        v.get("output_tokens") or 0
        for v in token_usage.values()
        if isinstance(v, dict)
    )
    run = AgentRun(
        id=run_id,
        user_id=user_id,
        query=query,
        answer=answer,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )
    db.add(run)
    for tc in tool_calls:
        db.add(ToolCall(
            id=uuid.uuid4(),
            run_id=run_id,
            tool_name=tc.tool_name,
            input_json=tc.input,
            output_json=tc.output,
            duration_ms=tc.duration_ms,
        ))
    await db.commit()


@router.post("/query/stream")
async def query_stream(
    body: AgentQueryRequest,
    request: Request,
    user: CurrentUserDep,
    db: DbDep,
    cheap_client: CheapLlmDep,
    strong_client: StrongLlmDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Stream the agent response as Server-Sent Events.

    Events emitted (each line: `data: <json>\\n\\n`):
      {"type": "tool_call", "tool_name": ..., "input": ..., "output": ..., "duration_ms": ...}
      {"type": "token", "text": "..."}
      {"type": "done", "run_id": ..., "answer": ..., "styles_predicted": [...], ...}
      {"type": "needs_more_info", "message": ..., "missing_fields": [...]}
      {"type": "error", "message": "..."}
    """
    log.info("agent.stream.start", user_id=str(user.id), query_len=len(body.query))

    async def event_generator():
        done_payload = None
        async for chunk in run_agent_stream(
            query=body.query,
            cheap_client=cheap_client,
            strong_client=strong_client,
            cheap_model=settings.cheap_model,
            strong_model=settings.strong_model,
            db=db,
            embedding_model=settings.embedding_model,
            top_k=settings.rag_top_k,
        ):
            yield chunk
            # Capture the done event so we can persist after streaming
            if chunk.startswith("data: "):
                try:
                    payload = json.loads(chunk[6:])
                    if payload.get("type") == "done":
                        done_payload = payload
                except Exception:
                    pass

        # Persist the run and fire webhook after the stream is fully consumed
        if done_payload:
            try:
                tcs = [
                    ToolCallRecord(**tc) for tc in (done_payload.get("tool_calls") or [])
                ]
                run_id = uuid.UUID(done_payload["run_id"])
                answer = done_payload.get("answer", "")
                token_usage = done_payload.get("token_usage") or {}
                styles = done_payload.get("styles_predicted") or []
                dest_names = [d["name"] for d in (done_payload.get("destination_metadata") or [])]

                await _persist_run(
                    db=db,
                    run_id=run_id,
                    user_id=user.id,
                    query=body.query,
                    answer=answer,
                    tool_calls=tcs,
                    token_usage=token_usage,
                )
                log.info("agent.stream.persisted", run_id=done_payload["run_id"])

                webhook_fire(_build_webhook_payload(
                    run_id=run_id,
                    user_id=user.id,
                    query=body.query,
                    answer=answer,
                    tool_calls=tcs,
                    token_usage=token_usage,
                    styles_predicted=styles,
                    destination_names=dest_names,
                ))
            except Exception as exc:
                log.error("agent.stream.persist_error", error=str(exc))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=list[AgentRunResponse])
async def history(user: CurrentUserDep, db: DbDep) -> list[AgentRunResponse]:
    rows = await db.execute(
        select(AgentRun)
        .where(AgentRun.user_id == user.id)
        .order_by(AgentRun.created_at.desc())
        .limit(20)
    )
    runs = rows.scalars().all()

    responses = []
    for run in runs:
        tc_rows = await db.execute(
            select(ToolCall).where(ToolCall.run_id == run.id)
        )
        tool_calls = [
            ToolCallRecord(
                tool_name=tc.tool_name,
                input=tc.input_json,
                output=tc.output_json,
                duration_ms=tc.duration_ms,
            )
            for tc in tc_rows.scalars().all()
        ]
        responses.append(AgentRunResponse(
            run_id=run.id,
            answer=run.answer or "",
            tool_calls=tool_calls,
            styles_predicted=[],
            token_usage={},
            created_at=run.created_at,
        ))

    return responses
