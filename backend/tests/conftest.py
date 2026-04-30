"""Shared pytest fixtures.

Tests run against an in-memory SQLite (via aiosqlite) so they're fast and
hermetic. Tables that use pgvector (Destination/Document/Chunk) are skipped
in this fixture — they're tested separately against a real Postgres in CI.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

# Set required env vars BEFORE importing anything that loads Settings.
# pydantic-settings has extra="forbid", so every required key must be present.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault(
    "JWT_SECRET", "test-secret-must-be-at-least-32-characters-long-for-validation"
)
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import get_settings  # noqa: E402
from core.deps import get_db  # noqa: E402
from db.models import AgentRun, User  # noqa: E402
from main import create_app  # noqa: E402

# Tables that work on plain SQLite (i.e. don't use pgvector or JSONB).
# ToolCall uses JSONB, Destination/Document/Chunk use pgvector — all skipped here.
# Auth tests only need User; we include AgentRun for completeness as a sanity check.
_SQLITE_SAFE_TABLES = [User.__table__, AgentRun.__table__]


@pytest.fixture(scope="session", autouse=True)
def _clear_settings_cache() -> None:
    """Ensure each test session reads the env we set above, not a stale cache."""
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def engine():
    """Per-test in-memory SQLite engine. Creates the auth-relevant tables only."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        for table in _SQLITE_SAFE_TABLES:
            await conn.run_sync(table.create)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def client(session_factory) -> AsyncIterator[AsyncClient]:
    """An httpx AsyncClient bound to the FastAPI app via ASGI transport.

    The DB dependency is overridden to use our test session factory rather
    than the lifespan-built one (lifespan does not run for ASGITransport).
    """
    app = create_app()
    app.state.session_factory = session_factory  # what get_db reads from request.app.state

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
