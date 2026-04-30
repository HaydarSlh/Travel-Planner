"""Ingest script: PDF → sections → chunks → embed → insert into DB.

Run with:
    $env:PYTHONPATH = "."; uv run python -m rag.ingest

Safe to re-run — destinations that already have documents in the DB are skipped,
so a partial run (e.g. interrupted by a rate limit) can be resumed cleanly.
"""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import select

from agent.router import build_gemini_client
from core.config import get_settings
from db.engine import create_engine, make_session_factory
from db.models import Document, Destination
from rag.chunker import chunk_sections
from rag.embedder import embed_chunks
from rag.loader import RawSection, load_documents
from rag.store import insert_chunks

log = structlog.get_logger()


async def _already_ingested(session, destination_name: str) -> bool:
    """Return True if this destination already has at least one document row."""
    result = await session.execute(
        select(Document)
        .join(Destination, Destination.id == Document.destination_id)
        .where(Destination.name == destination_name)
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def run_ingest() -> None:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = make_session_factory(engine)

    log.info("ingest.start")
    all_sections = load_documents()
    log.info("ingest.sections_loaded", count=len(all_sections))

    # Filter out destinations already in the DB so re-runs are safe
    async with session_factory() as session:
        seen: set[str] = set()
        pending_sections: list[RawSection] = []
        for sec in all_sections:
            if sec.destination_name not in seen:
                if await _already_ingested(session, sec.destination_name):
                    seen.add(sec.destination_name)
                    log.info("ingest.skip_existing", destination=sec.destination_name)
                    continue
                seen.add(sec.destination_name)
            pending_sections.append(sec)

    if not pending_sections:
        print("Nothing to ingest — all destinations already in DB.")
        await engine.dispose()
        return

    unique_pending = {s.destination_name for s in pending_sections}
    log.info("ingest.pending", destinations=sorted(unique_pending))

    chunks = chunk_sections(pending_sections)
    log.info("ingest.chunks_created", count=len(chunks))

    client = build_gemini_client(settings.gemini_api_key)
    embedded = await embed_chunks(chunks, client, settings.embedding_model)
    log.info("ingest.embedded", count=len(embedded))

    async with session_factory() as session:
        inserted = await insert_chunks(embedded, session)
    await engine.dispose()

    log.info("ingest.done", inserted=inserted)
    print(f"\nIngest complete: {inserted} chunks inserted.")


if __name__ == "__main__":
    asyncio.run(run_ingest())
