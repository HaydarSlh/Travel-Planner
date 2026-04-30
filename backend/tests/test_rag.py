"""Tests for the RAG pipeline — chunker and rag_retrieve tool.

loader.py is not unit-tested here because it calls pymupdf on real PDFs —
that belongs in an integration test. The chunker and retrieve tool are fully
covered with lightweight stubs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.rag_retrieve import rag_retrieve
from rag.chunker import HEADING_TO_GROUP, Chunk, chunk_sections
from rag.loader import RawSection
from schemas.tools import RAGRetrieveInput, RAGRetrieveOutput, ToolError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(heading: str, text: str, dest: str = "Bali", styles: list[str] | None = None) -> RawSection:
    return RawSection(
        destination_name=dest,
        styles=styles or ["Relaxation"],
        source="https://example.com/bali",
        heading=heading,
        text=text,
    )


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_single_short_section_produces_one_chunk(self) -> None:
        sections = [_section("See", "Visit the rice terraces.")]
        chunks = chunk_sections(sections)
        assert len(chunks) == 1
        assert chunks[0].section_group == "activities"
        assert "Visit the rice terraces." in chunks[0].text

    def test_heading_group_mapping(self) -> None:
        for heading, expected_group in [
            ("Understand", "overview"),
            ("See", "activities"),
            ("Do", "activities"),
            ("Eat", "practical"),
            ("Drink", "practical"),
            ("Buy", "practical"),
            ("Sleep", "stay"),
            ("Get in", "logistics"),
            ("Get around", "logistics"),
            ("Stay safe", "safety"),
        ]:
            assert HEADING_TO_GROUP[heading] == expected_group

    def test_related_headings_merged_into_one_chunk(self) -> None:
        sections = [
            _section("Eat", "Great pho everywhere."),
            _section("Drink", "Try bia hoi on the pavement."),
            _section("Buy", "Silk and lacquerware at the market."),
        ]
        chunks = chunk_sections(sections)
        # All three map to "practical" — merged into one (or more if long)
        practical = [c for c in chunks if c.section_group == "practical"]
        assert len(practical) >= 1
        combined = " ".join(c.text for c in practical)
        assert "pho" in combined
        assert "bia hoi" in combined
        assert "lacquerware" in combined

    def test_long_section_is_sub_chunked(self) -> None:
        long_text = "word " * 400  # ~2000 chars, well above MAX_CHUNK_CHARS=800
        sections = [_section("Understand", long_text)]
        chunks = chunk_sections(sections, max_chunk_chars=800, overlap_chars=100)
        overview = [c for c in chunks if c.section_group == "overview"]
        assert len(overview) > 1

    def test_overlap_means_shared_content(self) -> None:
        long_text = "a" * 900
        sections = [_section("Understand", long_text)]
        chunks = chunk_sections(sections, max_chunk_chars=500, overlap_chars=100)
        overview = [c for c in chunks if c.section_group == "overview"]
        assert len(overview) >= 2
        # Strip the prefix "[Bali] overview: " before comparing raw overlap
        raw0 = overview[0].text.split(": ", 1)[1]
        raw1 = overview[1].text.split(": ", 1)[1]
        # The tail of chunk 0 should appear at the start of chunk 1 (overlap)
        assert raw0[-100:] == raw1[:100]

    def test_prefix_contains_destination_and_group(self) -> None:
        sections = [_section("Do", "Surfing and hiking.", dest="Queenstown")]
        chunks = chunk_sections(sections)
        assert chunks[0].text.startswith("[Queenstown] activities:")

    def test_styles_propagated_to_chunk(self) -> None:
        sections = [_section("See", "Temples.", styles=["Culture", "Family"])]
        chunks = chunk_sections(sections)
        assert chunks[0].styles == ["Culture", "Family"]

    def test_unknown_heading_not_embedded(self) -> None:
        # "Go next" maps to "other" which is excluded from EMBED_GROUPS
        sections = [_section("Go next", "Consider Hoi An next.")]
        chunks = chunk_sections(sections)
        assert chunks == []

    def test_empty_text_section_is_skipped(self) -> None:
        sections = [_section("See", "   "), _section("Do", "Good hiking.")]
        chunks = chunk_sections(sections)
        assert all(c.text.strip() for c in chunks)

    def test_multiple_destinations_independent(self) -> None:
        sections = [
            _section("See", "Temples in Kyoto.", dest="Kyoto", styles=["Culture"]),
            _section("See", "Beaches in Bali.", dest="Bali", styles=["Relaxation"]),
        ]
        chunks = chunk_sections(sections)
        names = {c.destination_name for c in chunks}
        assert "Kyoto" in names and "Bali" in names


# ── rag_retrieve tool tests ───────────────────────────────────────────────────

def _make_embed_client(values: list[float]):
    emb = MagicMock()
    emb.values = values
    resp = MagicMock()
    resp.embeddings = [emb]
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(return_value=resp)
    return client


class TestRagRetrieveTool:
    async def test_returns_output_on_success(self, monkeypatch) -> None:
        fake_rows = [
            {"text": "[Bali] activities: Surf and hike.", "destination_name": "Bali",
             "section_group": "activities", "styles": "Relaxation", "distance": 0.1},
        ]
        monkeypatch.setattr("agent.tools.rag_retrieve.similarity_search", AsyncMock(return_value=fake_rows))

        result = await rag_retrieve(
            RAGRetrieveInput(query="outdoor activities", travel_style="Relaxation", top_k=5),
            session=MagicMock(),
            client=_make_embed_client([0.1] * 768),
            embedding_model="text-embedding-004",
        )

        assert isinstance(result, RAGRetrieveOutput)
        assert result.chunks == ["[Bali] activities: Surf and hike."]
        assert result.source_documents == ["Bali"]

    async def test_embed_failure_returns_tool_error(self, monkeypatch) -> None:
        client = MagicMock()
        client.aio.models.embed_content = AsyncMock(side_effect=RuntimeError("quota exceeded"))

        result = await rag_retrieve(
            RAGRetrieveInput(query="anything", travel_style="Adventure", top_k=3),
            session=MagicMock(),
            client=client,
            embedding_model="text-embedding-004",
        )

        assert isinstance(result, ToolError)
        assert "quota exceeded" in result.error

    async def test_empty_db_results_returns_empty_output(self, monkeypatch) -> None:
        monkeypatch.setattr("agent.tools.rag_retrieve.similarity_search", AsyncMock(return_value=[]))

        result = await rag_retrieve(
            RAGRetrieveInput(query="obscure query", travel_style="Luxury", top_k=5),
            session=MagicMock(),
            client=_make_embed_client([0.0] * 768),
            embedding_model="text-embedding-004",
        )

        assert isinstance(result, RAGRetrieveOutput)
        assert result.chunks == []
