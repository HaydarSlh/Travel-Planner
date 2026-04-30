"""Async SQLAlchemy engine and session factory.

Both functions are pure — they take a Settings object and return resources.
The lifespan handler in `main.py` is responsible for creating the engine on
startup, attaching it to `app.state`, and disposing it on shutdown.

There are NO module-level globals here. That's deliberate: importing this module
is cheap and side-effect-free, which is what tests need.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine from Settings.

    `pool_pre_ping=True` weeds out connections that have been killed by the
    database mid-pool — a common cause of mystery 500s in long-running services.
    """
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory bound to the given engine.

    `expire_on_commit=False` keeps loaded ORM objects usable after a commit —
    important for FastAPI handlers that return a recently-committed model.
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
