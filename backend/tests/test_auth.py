"""End-to-end tests for the auth flow: register -> login -> token usage."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestAuthFlow:
    async def test_register_creates_user(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "hunter2hunter2"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["email"] == "alice@example.com"
        assert "id" in body
        assert "created_at" in body
        assert "password" not in body
        assert "hashed_password" not in body

    async def test_duplicate_email_returns_409(self, client: AsyncClient) -> None:
        await client.post(
            "/auth/register",
            json={"email": "bob@example.com", "password": "hunter2hunter2"},
        )
        response = await client.post(
            "/auth/register",
            json={"email": "bob@example.com", "password": "anotherpass123"},
        )
        assert response.status_code == 409

    async def test_register_rejects_short_password(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/register",
            json={"email": "carol@example.com", "password": "short"},
        )
        assert response.status_code == 422  # pydantic validation

    async def test_register_rejects_invalid_email(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "hunter2hunter2"},
        )
        assert response.status_code == 422

    async def test_login_returns_token(self, client: AsyncClient) -> None:
        await client.post(
            "/auth/register",
            json={"email": "dave@example.com", "password": "hunter2hunter2"},
        )
        response = await client.post(
            "/auth/login",
            json={"email": "dave@example.com", "password": "hunter2hunter2"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"  # noqa: S105
        assert body["expires_in"] > 0

    async def test_login_wrong_password_returns_401(self, client: AsyncClient) -> None:
        await client.post(
            "/auth/register",
            json={"email": "eve@example.com", "password": "hunter2hunter2"},
        )
        response = await client.post(
            "/auth/login",
            json={"email": "eve@example.com", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    async def test_login_unknown_email_returns_401(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "anything12345"},
        )
        assert response.status_code == 401
