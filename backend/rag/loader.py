"""Extract text from PDF destination guides and split into labeled sections.

Uses pymupdf to read each PDF and detect section boundaries from font size:
  - size >= 16, bold (flags & 16)  → top-level Wikivoyage heading (Understand, Get in, …)
  - size >= 12, bold               → sub-heading (By plane, Climate, …) — kept as body text

Each section's text is cleaned (climate tables, captions, update tags stripped)
and returned as a RawSection with the heading name and destination metadata attached.

The destinations_styles_headings.json file is the authoritative source for which
styles each destination belongs to.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf

DOCS_DIR = Path(__file__).parent / "docs"
META_FILE = DOCS_DIR / "destinations_styles_headings.json"

# Font-size threshold that distinguishes a Wikivoyage top-level heading from body text.
# Observed in all 13 PDFs: headings like "Get in", "Understand" appear at size 18, bold.
_HEADING_MIN_SIZE = 16.0
_HEADING_BOLD_FLAG = 16  # pymupdf flag bit for bold


@dataclass
class RawSection:
    destination_name: str
    styles: list[str]
    source: str
    heading: str      # e.g. "Get in", "Eat"
    text: str         # cleaned body text for this section


def _is_heading(span: dict) -> bool:
    return span["size"] >= _HEADING_MIN_SIZE and bool(span["flags"] & _HEADING_BOLD_FLAG)


def _extract_sections(pdf_path: Path, known_headings: list[str]) -> list[tuple[str, str]]:
    """Return list of (heading, body_text) pairs extracted from the PDF.

    Detects section boundaries by font size. Paragraphs between two headings
    are joined into a single body string.
    """
    doc = fitz.open(str(pdf_path))
    known_set = {h.lower() for h in known_headings}

    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def _flush():
        if current_heading:
            body = _clean_text(" ".join(current_lines))
            if body:
                sections.append((current_heading, body))

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:  # skip image blocks
                continue
            for line in block["lines"]:
                line_text_parts = []
                line_is_heading = False
                heading_candidate = ""

                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    if _is_heading(span):
                        line_is_heading = True
                        heading_candidate += text
                    else:
                        line_text_parts.append(text)

                if line_is_heading:
                    candidate = heading_candidate.strip()
                    if candidate.lower() in known_set:
                        _flush()
                        current_heading = candidate
                        current_lines = []
                    # heading not in known list → treat as sub-heading, keep as body
                    elif current_heading and candidate:
                        current_lines.append(candidate)
                        current_lines.extend(line_text_parts)
                elif line_text_parts and current_heading:
                    current_lines.extend(line_text_parts)

    _flush()
    doc.close()
    return sections


# Patterns that add noise but no travel value: climate table numbers, update tags,
# image captions (short lines of <= 6 words sandwiched between numbers), URL fragments.
_NOISE_PATTERNS = [
    re.compile(r"\(updated \w+ \d{4}\)", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"\b\d{1,3}\s+\d{1,3}\s+\d{1,3}\b"),   # climate table triplets
    re.compile(r"☏\s*[\+\d\s\-]+"),                    # phone numbers
    re.compile(r"IATA"),
]


def _clean_text(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_documents(docs_dir: Path = DOCS_DIR, meta_file: Path = META_FILE) -> list[RawSection]:
    """Load all destination PDFs and return a flat list of RawSection objects.

    Each RawSection corresponds to one Wikivoyage top-level section
    (Understand, Get in, Eat, …) for one destination.
    """
    meta: list[dict] = json.loads(meta_file.read_text(encoding="utf-8"))
    meta_by_stem: dict[str, dict] = {}
    for entry in meta:
        # Map from PDF filename stem to metadata entry
        source_url = entry["source"]
        stem = source_url.rstrip("/").split("/")[-1]  # e.g. "Hanoi"
        meta_by_stem[stem] = entry

    sections: list[RawSection] = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        stem = pdf_path.stem  # e.g. "Hanoi"
        entry = meta_by_stem.get(stem)
        if entry is None:
            continue  # no metadata for this file — skip

        raw_sections = _extract_sections(pdf_path, entry["headings"])
        for heading, body in raw_sections:
            sections.append(RawSection(
                destination_name=entry["name"],
                styles=entry["styles"],
                source=entry["source"],
                heading=heading,
                text=body,
            ))

    return sections
