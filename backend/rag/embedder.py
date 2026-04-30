"""Embed chunks with Gemini text-embedding-004.

Processes chunks in small async batches to stay within rate limits.
Returns EmbeddedChunk objects — the same data as Chunk plus an embedding vector.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog
from google import genai
from google.genai import types

from rag.chunker import Chunk

log = structlog.get_logger()

_TASK_TYPE = "RETRIEVAL_DOCUMENT"
_BATCH_SIZE = 10
_BATCH_DELAY = 0.5  # seconds between batches — avoids rate-limit spikes


@dataclass
class EmbeddedChunk:
    destination_name: str
    styles: list[str]
    source: str
    section_group: str
    text: str
    chunk_index: int
    embedding: list[float]


async def embed_chunks(
    chunks: list[Chunk],
    client: genai.Client,
    model: str,
) -> list[EmbeddedChunk]:
    """Embed all chunks and return EmbeddedChunk objects in the same order."""
    embedded: list[EmbeddedChunk] = []

    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _BATCH_SIZE]

        response = await client.aio.models.embed_content(
            model=model,
            contents=[c.text for c in batch],
            config=types.EmbedContentConfig(task_type=_TASK_TYPE),
        )

        for chunk, emb_obj in zip(batch, response.embeddings):
            embedded.append(EmbeddedChunk(
                destination_name=chunk.destination_name,
                styles=chunk.styles,
                source=chunk.source,
                section_group=chunk.section_group,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                embedding=emb_obj.values,
            ))

        log.info(
            "rag.embedder.batch_done",
            batch_start=batch_start,
            batch_size=len(batch),
            total=len(chunks),
        )

        if batch_start + _BATCH_SIZE < len(chunks):
            await asyncio.sleep(_BATCH_DELAY)

    return embedded
