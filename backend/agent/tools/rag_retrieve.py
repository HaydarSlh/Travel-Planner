"""rag_retrieve tool — embed the query then search pgvector for matching chunks.

Called by the agent after the classifier has determined the travel style.
Filters by style (stored on each chunk) so only relevant destination content
is returned. Returns ToolError on any failure.
"""

from __future__ import annotations

import structlog
from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from rag.store import similarity_search
from schemas.tools import RAGRetrieveInput, RAGRetrieveOutput, ToolError

log = structlog.get_logger()

_TOOL_NAME = "rag_retrieve"
_TASK_TYPE = "RETRIEVAL_QUERY"


async def rag_retrieve(
    inp: RAGRetrieveInput,
    session: AsyncSession,
    client: genai.Client,
    embedding_model: str,
) -> RAGRetrieveOutput | ToolError:
    """Embed inp.query and return the top-k most similar chunks from the DB."""
    try:
        response = await client.aio.models.embed_content(
            model=embedding_model,
            contents=[inp.query],
            config=types.EmbedContentConfig(task_type=_TASK_TYPE),
        )
        query_embedding: list[float] = response.embeddings[0].values

        rows = await similarity_search(
            query_embedding=query_embedding,
            session=session,
            top_k=inp.top_k,
            style_filter=inp.travel_style,
        )

        log.info(
            "rag_retrieve.done",
            query_len=len(inp.query),
            style=inp.travel_style,
            results=len(rows),
        )

        return RAGRetrieveOutput(
            chunks=[r["text"] for r in rows],
            source_documents=[r["destination_name"] for r in rows],
            image_urls=[r.get("image_url") for r in rows],
            source_urls=[r.get("source_url") for r in rows],
        )

    except Exception as exc:  # noqa: BLE001
        log.warning("rag_retrieve.error", error=str(exc))
        return ToolError(tool=_TOOL_NAME, error=str(exc))
