"""Liveness probe for Docker / load balancers."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from core.deps import DbDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(db: DbDep) -> dict[str, str]:
    """Return service status. Confirms the DB is reachable with a SELECT 1."""
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:  # noqa: BLE001 - we intentionally degrade gracefully here
        db_status = "down"
    return {"status": "ok" if db_status == "ok" else "degraded", "db": db_status}
