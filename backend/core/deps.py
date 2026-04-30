"""FastAPI dependency-injection functions.

Every shared resource — DB session, LLM clients, current user, Settings — is
exposed here so routes declare what they need with `Depends()`. No globals.
In tests, `app.dependency_overrides[get_x]` replaces any of these with a fake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from core.security import decode_token
from db.models import User

_bearer_scheme = HTTPBearer(auto_error=True)


def settings_dep() -> Settings:
    """Inject the cached Settings singleton."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession scoped to this request.

    The session is created from the factory attached to `app.state` by the
    lifespan handler. Closing happens automatically when the generator
    finalizes — FastAPI handles that.
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


DbDep = Annotated[AsyncSession, Depends(get_db)]


def get_cheap_llm(request: Request) -> genai.Client:
    """Return the process-wide Gemini client used for cheap calls."""
    return request.app.state.cheap_client


def get_strong_llm(request: Request) -> genai.Client:
    """Return the process-wide Gemini client used for synthesis."""
    return request.app.state.strong_client


CheapLlmDep = Annotated[genai.Client, Depends(get_cheap_llm)]
StrongLlmDep = Annotated[genai.Client, Depends(get_strong_llm)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    db: DbDep,
    settings: SettingsDep,
) -> User:
    token = credentials.credentials
    """Decode the Bearer JWT and load the corresponding User from the DB."""
    payload = decode_token(token, settings)
    user_id_raw = payload.get("sub")
    if not user_id_raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(select(User).where(User.id == user_id_raw))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="user not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def get_classifier(request: Request):
    """Return the joblib-loaded sklearn Pipeline, or None if not yet trained."""
    return getattr(request.app.state, "classifier", None)
