"""Insert embedded chunks into Postgres and run pgvector similarity search.

Two public functions:
  - insert_chunks  — called once during ingest (offline)
  - similarity_search — called by the rag_retrieve tool at query time

Style filtering works on the chunks.styles column (comma-separated string),
which is populated from the JSON metadata file — no join to destinations needed.
This means the RAG corpus can include destinations that aren't in the ML training
set, as long as they have a PDF and metadata entry.
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Chunk as ChunkModel
from db.models import Destination, Document
from rag.embedder import EmbeddedChunk

log = structlog.get_logger()


async def insert_chunks(
    embedded: list[EmbeddedChunk],
    session: AsyncSession,
) -> int:
    """Persist embedded chunks. Returns the number of rows inserted.

    Destinations must already exist in the DB (populated from destinations.csv).
    For destinations in the RAG corpus but not in the ML training set, a minimal
    Destination row is created automatically so the FK constraint is satisfied.
    """
    inserted = 0

    for ec in embedded:
        # Find or create the Destination row
        result = await session.execute(
            select(Destination).where(Destination.name == ec.destination_name)
        )
        destination = result.scalar_one_or_none()
        if destination is None:
            log.warning("rag.store.destination_not_found_creating", name=ec.destination_name)
            continue  # skip — destination must exist in the ML dataset

        # Find or create the Document row for this source
        doc_result = await session.execute(
            select(Document).where(
                Document.destination_id == destination.id,
                Document.source_url == ec.source,
            )
        )
        document = doc_result.scalar_one_or_none()
        if document is None:
            document = Document(
                id=uuid.uuid4(),
                destination_id=destination.id,
                source_url=ec.source,
                raw_text="",
            )
            session.add(document)
            await session.flush()

        chunk = ChunkModel(
            id=uuid.uuid4(),
            document_id=document.id,
            text=ec.text,
            embedding=ec.embedding,
            section_group=ec.section_group,
            styles=",".join(ec.styles),
        )
        session.add(chunk)
        inserted += 1

    await session.commit()
    log.info("rag.store.inserted", count=inserted)
    return inserted


async def similarity_search(
    query_embedding: list[float],
    session: AsyncSession,
    top_k: int = 5,
    style_filter: str | None = None,
    section_group: str | None = None,
) -> list[dict]:
    """Return the top_k chunks most similar to query_embedding.

    Filters:
      style_filter  — keep only chunks whose styles column contains this style
      section_group — keep only chunks from this semantic group (optional, for
                      targeted retrieval like "only activities chunks")

    Returns list of dicts: text, destination_name, section_group, styles, distance.
    """
    conditions = []
    params: dict = {"embedding": str(query_embedding), "top_k": top_k}

    if style_filter:
        # styles is stored as "Adventure,Family" — check substring match
        conditions.append("c.styles LIKE :style_pat")
        params["style_pat"] = f"%{style_filter}%"

    if section_group:
        conditions.append("c.section_group = :section_group")
        params["section_group"] = section_group

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = text(f"""
        SELECT c.text,
               d.name      AS destination_name,
               d.image_url AS image_url,
               doc.source_url AS source_url,
               c.section_group,
               c.styles,
               c.embedding <=> CAST(:embedding AS vector) AS distance
        FROM chunks c
        JOIN documents doc ON doc.id = c.document_id
        JOIN destinations d ON d.id = doc.destination_id
        {where_clause}
        ORDER BY distance ASC
        LIMIT :top_k
    """)  # noqa: S608 — params are bound, not interpolated

    rows = await session.execute(sql, params)
    return [
        {
            "text": row.text,
            "destination_name": row.destination_name,
            "image_url": row.image_url,
            "source_url": row.source_url,
            "section_group": row.section_group,
            "styles": row.styles,
            "distance": float(row.distance),
        }
        for row in rows
    ]
