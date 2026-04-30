"""HTTP endpoint tests for /tools/classify and /tools/rag.

Tests the FastAPI layer: auth gate (401), input validation (422),
503 when classifier not loaded, 502 on RAG tool error, and happy-path
response shapes. The classifier pipeline and rag_retrieve function are
stubbed so these tests never touch a real model or database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_cheap_llm, get_classifier, get_current_user, get_db
from db.models import User
from main import create_app
from schemas.tools import RAGRetrieveOutput, ToolError

# ── Shared test data ──────────────────────────────────────────────────────────

CLASSES = ["Adventure", "Budget", "Culture", "Family", "Luxury", "Relaxation"]

VALID_CLASSIFY_BODY = {
    "climate_zone": "Csa",
    "avg_temp_peak_season_c": 30.0,
    "peak_season_length_months": 5,
    "terrain_primary": "Coastal",
    "coastal_access": 1,
    "unesco_sites_count": 3,
    "outdoor_activity_score": 2,
    "daily_cost_bucket": 4,
    "accommodation_range": "Mid-Luxury",
    "visa_difficulty": 1,
    "english_prevalence": 2,
    "tourism_maturity": "Established",
}

VALID_RAG_BODY = {
    "query": "warm beach not touristy",
    "travel_style": "Relaxation",
    "top_k": 3,
}

_FAKE_USER = User(
    id="00000000-0000-0000-0000-000000000001",
    email="tools-test@example.com",
    hashed_password="x",
)


# ── Stub classifier pipeline ──────────────────────────────────────────────────

class _StubPipeline:
    """Returns Relaxation at 0.75 confidence."""
    classes_ = np.array(CLASSES)

    def predict_proba(self, X):  # noqa: N803
        proba = [0.05, 0.05, 0.05, 0.05, 0.05, 0.75]
        return np.array([proba])


# ── Per-test app + client ─────────────────────────────────────────────────────
# Each test gets a fresh app so overrides don't bleed between tests.

@pytest_asyncio.fixture
async def tools_client(session_factory) -> AsyncIterator[AsyncClient]:
    """AsyncClient wired to a fresh app instance with DB override applied."""
    app = create_app()
    app.state.session_factory = session_factory

    async def override_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_db] = override_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ac.app = app  # expose app on the client so tests can mutate overrides
        yield ac

    app.dependency_overrides.clear()


# ── /tools/classify ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_happy_path(tools_client: AsyncClient) -> None:
    app = tools_client.app
    app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    # classifier is read from app.state directly in the router, not via Depends
    app.state.classifier = _StubPipeline()

    resp = await tools_client.post("/tools/classify", json=VALID_CLASSIFY_BODY)

    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_style"] == "Relaxation"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["per_class_probs"].keys()) == set(CLASSES)
    assert abs(sum(body["per_class_probs"].values()) - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_classify_503_model_not_loaded(tools_client: AsyncClient) -> None:
    app = tools_client.app
    app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    app.state.classifier = None  # simulates missing model.joblib

    resp = await tools_client.post("/tools/classify", json=VALID_CLASSIFY_BODY)

    assert resp.status_code == 503
    assert "train" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_classify_422_invalid_climate_zone(tools_client: AsyncClient) -> None:
    app = tools_client.app
    app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    app.state.classifier = _StubPipeline()

    resp = await tools_client.post(
        "/tools/classify", json={**VALID_CLASSIFY_BODY, "climate_zone": "INVALID"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_classify_422_out_of_range_temp(tools_client: AsyncClient) -> None:
    app = tools_client.app
    app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    app.state.classifier = _StubPipeline()

    resp = await tools_client.post(
        "/tools/classify", json={**VALID_CLASSIFY_BODY, "avg_temp_peak_season_c": 999}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_classify_401_no_auth(tools_client: AsyncClient) -> None:
    # No auth override — HTTPBearer returns 401 when the Authorization header is absent
    tools_client.app.dependency_overrides.pop(get_current_user, None)
    resp = await tools_client.post("/tools/classify", json=VALID_CLASSIFY_BODY)
    assert resp.status_code == 401


# ── /tools/rag ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_happy_path(tools_client: AsyncClient, monkeypatch) -> None:
    app = tools_client.app
    app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    app.dependency_overrides[get_cheap_llm] = lambda: object()

    async def _fake_rag(inp, db, llm_client, model):
        return RAGRetrieveOutput(
            chunks=["Bali has great surf.", "Quiet beaches in the north."],
            source_documents=["Bali", "Bali"],
            image_urls=[None, None],
            source_urls=["https://example.com/bali", "https://example.com/bali"],
        )

    # Patch where the router imports it, not the origin module
    import routers.tools as tools_router
    monkeypatch.setattr(tools_router, "rag_retrieve", _fake_rag)

    resp = await tools_client.post("/tools/rag", json=VALID_RAG_BODY)

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["chunks"]) == 2
    assert body["source_documents"] == ["Bali", "Bali"]
    assert "image_urls" in body
    assert "source_urls" in body


@pytest.mark.asyncio
async def test_rag_502_on_tool_error(tools_client: AsyncClient, monkeypatch) -> None:
    app = tools_client.app
    app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    app.dependency_overrides[get_cheap_llm] = lambda: object()

    async def _failing_rag(inp, db, llm_client, model):
        return ToolError(tool="rag_retrieve", error="embedding service unavailable")

    import routers.tools as tools_router
    monkeypatch.setattr(tools_router, "rag_retrieve", _failing_rag)

    resp = await tools_client.post("/tools/rag", json=VALID_RAG_BODY)

    assert resp.status_code == 502
    assert "embedding service unavailable" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_rag_422_empty_query(tools_client: AsyncClient) -> None:
    app = tools_client.app
    app.dependency_overrides[get_current_user] = lambda: _FAKE_USER
    app.dependency_overrides[get_cheap_llm] = lambda: object()

    resp = await tools_client.post(
        "/tools/rag", json={**VALID_RAG_BODY, "query": ""}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_rag_401_no_auth(tools_client: AsyncClient) -> None:
    # HTTPBearer returns 401 when Authorization header is absent
    tools_client.app.dependency_overrides.pop(get_current_user, None)
    resp = await tools_client.post("/tools/rag", json=VALID_RAG_BODY)
    assert resp.status_code == 401
