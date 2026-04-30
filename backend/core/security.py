"""Authentication primitives — password hashing and JWTs.

Two concerns isolated from everything else:
  1. Password hashing using bcrypt (via passlib's CryptContext).
  2. JWT creation and decoding using PyJWT.

These functions are pure: they take a Settings object explicitly rather than
importing a global. That keeps the security layer testable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from fastapi import HTTPException, status

from core.config import Settings

# bcrypt has a hard 72-byte limit on the input password. We truncate explicitly
# rather than letting it raise — the truncation is part of the algorithm spec
# and matches what every other bcrypt-using system does.
_BCRYPT_MAX_BYTES = 72


def hash_password(plain: str) -> str:
    """Hash a plain-text password using bcrypt.

    bcrypt embeds the salt in the output, so we don't manage one ourselves.
    The returned string is self-contained and suitable for direct DB storage.
    """
    truncated = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    hashed = bcrypt.hashpw(truncated, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison of a plain password against a stored hash."""
    truncated = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(truncated, hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str, settings: Settings) -> tuple[str, int]:
    """Create a signed JWT for the given subject (typically the user id).

    Returns (token, expires_in_seconds) so the caller can put both in the
    TokenResponse without recomputing the expiry.
    """
    expires_in = settings.jwt_expiry_minutes * 60
    expiry = datetime.now(tz=UTC) + timedelta(seconds=expires_in)
    payload = {
        "sub": subject,
        "exp": int(expiry.timestamp()),
        "iat": int(datetime.now(tz=UTC).timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


def decode_token(token: str, settings: Settings) -> dict[str, Any]:
    """Decode and verify a JWT. Raises HTTPException(401) on any failure.

    Wrapping the PyJWT errors in HTTPException keeps callers (route handlers,
    dependencies) from having to know which exception types PyJWT raises.
    """
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
