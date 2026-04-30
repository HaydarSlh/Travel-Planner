"""Standalone tool endpoints for testing and debugging.

These endpoints expose the individual agent tools directly so they can be
exercised in Swagger without running the full agent pipeline.

POST /tools/classify  — run the ML classifier on raw destination features
POST /tools/rag       — run a single RAG retrieve call
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from agent.tools.classify_destination import classify_destination
from agent.tools.rag_retrieve import rag_retrieve
from core.deps import CheapLlmDep, CurrentUserDep, DbDep, SettingsDep
from schemas.tools import (
    ClassifyDestinationInput,
    ClassifyDestinationOutput,
    RAGRetrieveInput,
    RAGRetrieveOutput,
    ToolError,
)

router = APIRouter(prefix="/tools", tags=["tools"])
log = structlog.get_logger()


@router.post(
    "/classify",
    response_model=ClassifyDestinationOutput,
    summary="Classify a destination's travel style from its features",
    description=(
        "Runs the ML classifier pipeline against the 12 destination features. "
        "Returns predicted style, confidence, and per-class probabilities. "
        "Requires the model to have been trained (`uv run python -m ml.train`)."
    ),
)
async def classify(
    body: ClassifyDestinationInput,
    request: Request,
    _user: CurrentUserDep,
    settings: SettingsDep,
) -> ClassifyDestinationOutput:
    pipeline = getattr(request.app.state, "classifier", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier not loaded — run `uv run python -m ml.train` first.",
        )

    result = classify_destination(body, pipeline, settings.classifier_confidence_threshold)
    if isinstance(result, ToolError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result.error)

    log.info("tools.classify.done", predicted=result.predicted_style, confidence=result.confidence)
    return result


@router.post(
    "/rag",
    response_model=RAGRetrieveOutput,
    summary="Retrieve destination chunks via pgvector similarity search",
    description=(
        "Embeds the query with Gemini and returns the top-k most similar chunks "
        "from the vector store, optionally filtered by travel style."
    ),
)
async def rag(
    body: RAGRetrieveInput,
    _user: CurrentUserDep,
    db: DbDep,
    cheap_client: CheapLlmDep,
    settings: SettingsDep,
) -> RAGRetrieveOutput:
    result = await rag_retrieve(body, db, cheap_client, settings.embedding_model)
    if isinstance(result, ToolError):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=result.error)

    log.info("tools.rag.done", chunks=len(result.chunks), style=body.travel_style)
    return result
