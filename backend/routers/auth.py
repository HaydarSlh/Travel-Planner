"""Authentication routes — registration and login.

Both endpoints validate their request body via Pydantic schemas at the
boundary and return Pydantic response models. No defensive checks inside.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from core.deps import DbDep, SettingsDep
from core.security import create_access_token, hash_password, verify_password
from db.models import User
from schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserPublic

router = APIRouter(prefix="/auth", tags=["auth"])
log = structlog.get_logger()


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
)
async def register(body: RegisterRequest, db: DbDep) -> UserPublic:
    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="email already registered",
        ) from None
    await db.refresh(user)
    log.info("auth.register", user_id=str(user.id), email=user.email)
    return UserPublic.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DbDep, settings: SettingsDep) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, expires_in = create_access_token(str(user.id), settings)
    log.info("auth.login", user_id=str(user.id))
    return TokenResponse(access_token=token, expires_in=expires_in)
