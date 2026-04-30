"""Group RawSections into semantically meaningful chunks and sub-chunk long ones.

Strategy
--------
1. Map every Wikivoyage heading into one of six semantic groups. Headings in the
   same group that come from the same destination are concatenated into one chunk —
   they carry related information and benefit from shared context in the embedding.

2. If a group's combined text is longer than MAX_CHUNK_CHARS, slide a window over
   it with OVERLAP_CHARS of carry-over so no context is lost at boundaries.

3. Each chunk is prefixed with "[DestinationName] [section_group]:" before embedding.
   This gives the embedding model context it wouldn't otherwise have, which
   measurably improves retrieval precision.

4. Every chunk carries metadata: destination_name, styles, section_group, chunk_index.
   The styles list is what enables style-filtered retrieval in the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.loader import RawSection

# ── Semantic groups ────────────────────────────────────────────────────────────
# Each Wikivoyage heading maps to one group. Headings not listed here are ignored
# (they add noise without travel-planning value: Connect, Talk, Go next, etc.).

HEADING_TO_GROUP: dict[str, str] = {
    "Understand":  "overview",
    "Regions":     "overview",
    "Districts":   "overview",
    "Cities":      "overview",
    "See":         "activities",
    "Do":          "activities",
    "Learn":       "activities",
    "Buy":         "practical",
    "Eat":         "practical",
    "Drink":       "practical",
    "Get in":      "logistics",
    "Get around":  "logistics",
    "Sleep":       "stay",
    "Stay safe":   "safety",
    "Respect":     "safety",
}

# Any heading not in HEADING_TO_GROUP is placed here rather than dropped entirely,
# so we don't silently lose content.
_FALLBACK_GROUP = "other"

MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200

# Groups worth embedding for travel planning. logistics/safety/other add noise
# without retrieval value — users ask about what to do and where to stay, not
# how to get a bus from the airport.
EMBED_GROUPS = {"overview", "activities", "practical", "stay"}

# Maximum characters to take from a merged group before sub-chunking.
# Wikivoyage PDFs are very long; capping keeps the total chunk count manageable
# for the embedding API while preserving the most useful content (which appears
# near the top of each section).
MAX_GROUP_CHARS = 6000


@dataclass
class Chunk:
    destination_name: str
    styles: list[str]
    source: str
    section_group: str   # overview | activities | practical | logistics | stay | safety | other
    text: str            # prefixed + cleaned, ready for embedding
    chunk_index: int     # position within this destination+group, 0-based


def _sliding_window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of at most max_chars characters."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end].strip())
        step = max_chars - overlap
        start += max(step, 1)
    return [c for c in chunks if c]


def chunk_sections(
    sections: list[RawSection],
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Convert a flat list of RawSections into Chunks ready for embedding.

    Sections from the same destination + group are merged first, then
    long merged texts are sub-chunked with overlap.
    """
    # Group by (destination_name, section_group) preserving order
    grouped: dict[tuple[str, str], list[RawSection]] = {}
    for sec in sections:
        group = HEADING_TO_GROUP.get(sec.heading, _FALLBACK_GROUP)
        key = (sec.destination_name, group)
        grouped.setdefault(key, []).append(sec)

    chunks: list[Chunk] = []
    for (dest_name, group), secs in grouped.items():
        # Skip groups that don't add retrieval value
        if group not in EMBED_GROUPS:
            continue

        styles = secs[0].styles
        source = secs[0].source

        # Merge all sections in this group, then cap to MAX_GROUP_CHARS
        merged = "\n\n".join(s.text for s in secs if s.text)
        if not merged.strip():
            continue
        merged = merged[:MAX_GROUP_CHARS]

        windows = _sliding_window(merged, max_chunk_chars, overlap_chars)
        for idx, window in enumerate(windows):
            # Prepend destination + group so the embedding model has context
            prefixed = f"[{dest_name}] {group}: {window}"
            chunks.append(Chunk(
                destination_name=dest_name,
                styles=styles,
                source=source,
                section_group=group,
                text=prefixed,
                chunk_index=idx,
            ))

    return chunks
