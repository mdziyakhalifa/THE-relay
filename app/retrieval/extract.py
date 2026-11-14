"""Page-anchored PDF text extraction with section tracking.

Why this module exists (brief requirement): every retrieved passage must
retain paper, arXiv id + exact version, page number and section heading —
and we must never invent page numbers or sections. This module therefore:

1. Extracts text strictly per PDF page (page index, 1-based). Page numbers in
   citations mean "PDF page index"; the manual inspection step in this work
   package verifies printed page numbers align with the PDF page index and
   records any deviation. We do NOT guess.
2. Detects section headings from font geometry using an explainable
   heuristic (body-size ratio + line-length + punctuation + optional
   numbering pattern). Detection is heuristic: a heading line is flagged
   with a boolean; confidence is judged by human inspection of the output
   lists, not claimed as exact.
3. Filters arXiv's rotated left-margin watermark and pure page-number footer
   lines so they don't pollute chunk text.

Output per paper (feeds the chunker): one JSON file with, per PDF page, an
ordered list of lines: text, ``h`` (is a heading), ``s`` (section in force
at that line). Section state evolves across the whole document so a chunk's
section is the running section, not a page-local guess.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pydantic
import pymupdf

from app.core.logging_setup import get_logger
from app.retrieval.papers_registry import PaperSpec

log = get_logger("relay.retrieval.extract")

# pymupdf span flags bit for bold
_SPAN_FLAG_BOLD = 16
_PAGES_EXT = ".pages.json"

_PURE_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_CAPTION_PREFIX = re.compile(r"^(figure|fig\.|table)\s", re.IGNORECASE)

# Heading-shape patterns (evidence-based refinement, see docs/decision_log.md)
# Components capped at 2 digits so table data ("41.29", "0.743", "1920") cannot
# masquerade as section numbers; bare numbers additionally must sit in the left
# margin column (x0 <= 130pt) where LaTeX puts the section-number column.
_NUMBER_ONLY = re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}$")  # "1", "2.3"
_NUMBERED_TEXT = re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}\s+\S")  # "2.3 Improved ..."
_APPENDIX = re.compile(r"^Appendix\s+[A-Z]\.\s+\S")  # "Appendix A. Additional ablation ..."
_SPECIAL = {"Abstract", "References"}
_SPECIAL_PREFIXES = ("Acknowledgments",)
_ASCII_LETTERS = re.compile(r"[A-Za-z]")
_SPACES_OVER_RATIO = 0.30  # figure glyph strings like "n e w _ p r o" have many spaces


class PageLine(pydantic.BaseModel):
    t: str  # text
    h: bool = False  # heading line per geometric heuristic
    s: str = "unknown"  # section in force after this line


class PageRecord(pydantic.BaseModel):
    page: int  # 1-based PDF page index
    lines: list[PageLine]


class ExtractionReport(pydantic.BaseModel):
    paper_key: str
    versioned_id: str
    n_pages: int
    n_heading_lines: int
    headings_in_order: list[str]
    chars_total: int


def _horizontal(line: dict[str, Any]) -> bool:
    """True unless the line is rotated (arXiv watermark lives 90°-rotated)."""
    direction = line.get("dir", (1.0, 0.0))
    return abs(direction[0]) >= 0.95


def _line_text_and_bold(line: dict[str, Any]) -> tuple[str, bool, float]:
    text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
    sizes = [span.get("size", 0.0) for span in line.get("spans", [])]
    bold = any(span.get("flags", 0) & _SPAN_FLAG_BOLD for span in line.get("spans", []))
    return text, bold, max(sizes) if sizes else 0.0


def _body_font_size(doc: pymupdf.Document) -> float:
    """Most common rounded font size across all horizontal lines = body text
    size. Mode, not mean: headings/captions are minority lines."""
    counter: Counter[int] = Counter()
    for page in doc:
        data = page.get_text("dict", sort=True)
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                if not _horizontal(line):
                    continue
                _, _, size = _line_text_and_bold(line)
                if size > 0:
                    counter[round(size * 2) / 2] += 1  # 0.5-pt buckets
    if not counter:
        return 0.0
    return float(counter.most_common(1)[0][0])


def _is_heading_candidate(
    text: str,
    bold: bool,
    line_size: float,
    body_size: float,
    page_index: int,
    x0: float = 0.0,
) -> bool:
    """Geometric + shape heuristic, tuned on measured font evidence of the
    three real papers (see scripts/_dbg_headings.py output + decision log).

    Precision choices, provable from the evidence table:
    - require ≥2 ASCII letters: kills emoji/math-symbol figure glyphs
    - number/appendix/special patterns anchor detection at ratios as low as
      1.09 (Moshi's true headings are only 1.09× body size, bold)
    - page-1 gate at ratio ≥1.40: author/affiliation blocks are 1.2× and
      must not be flagged, while titles (1.43–1.72×) pass
    - bold ≥1.15× on pages ≥2 keeps true section names; spaced figure glyphs
      (non-bold 1.21×) fall to the ≥1.45× non-bold rule
    """
    if not text or body_size <= 0:
        return False
    ratio = line_size / body_size
    if _NUMBER_ONLY.match(text):
        # bare "1"/"2.3" (LaTeX number column) — merges with the next heading;
        # tested FIRST: single digits fail the general length floor by design.
        # x0 gate: LaTeX puts these in the left margin; table values sit mid-page.
        return (bold or ratio >= 1.05) and x0 <= 130.0
    if len(text) > 90 or len(text) < 2:
        return False
    if text.endswith(".") and not _APPENDIX.match(text):
        return False
    if _CAPTION_PREFIX.match(text):
        return False
    if text.count(" ") / len(text) > _SPACES_OVER_RATIO:
        return False  # spaced figure glyphs

    if len(_ASCII_LETTERS.findall(text)) < 2:
        return False  # emoji, math symbols, "!"
    first_word = text.split()[0]
    if first_word.isdigit() and len(first_word) >= 3:
        return False  # table rows like "1920 5.3 Ablations ..."
    if _NUMBERED_TEXT.match(text) or _APPENDIX.match(text):
        return bold or ratio >= 1.05
    if text in _SPECIAL or text.startswith(_SPECIAL_PREFIXES):
        return bold or ratio >= 1.05
    if page_index == 1:
        return ratio >= 1.40  # title only (author blocks are ~1.2×, non-bold)
    return (bold and ratio >= 1.15) or ratio >= 1.45


def _looks_like_title_continuation(text: str) -> bool:
    """Conservative title-shape test for merging a bare section number with the
    heading text that follows on the NEXT line (attention/hnet render number
    and text as separate lines and sometimes the text line is not flagged by
    the geometric rule)."""
    if not (3 <= len(text) <= 60):
        return False
    if text.endswith((".", ",", ";", ":", "?", "!")):
        return False
    if text.count(" ") / max(len(text), 1) > _SPACES_OVER_RATIO:
        return False
    if len(_ASCII_LETTERS.findall(text)) < 2:
        return False
    return text[0].isupper() or text[0].isdigit()


def _merge_number_split_headings(
    rendered: list[tuple[str, bool]]
) -> list[tuple[str, bool]]:
    """Join a bare number heading with the heading text that follows it
    ("1" + "Introduction" -> "1 Introduction") so section labels keep their
    numbering. Two merge paths, both conservative:
    - the next line is a (geometrically) flagged non-number heading, or
    - the next line is a non-heading but title-shaped short line
    (e.g. "3.1" then "Scaled Dot-Product Attention")."""
    out: list[tuple[str, bool]] = []
    pending: str | None = None
    for text, is_heading in rendered:
        if pending is not None:
            if is_heading and not _NUMBER_ONLY.match(text):
                out.append((f"{pending} {text}", True))
                pending = None
                continue
            if not is_heading and _looks_like_title_continuation(text):
                out.append((f"{pending} {text}", True))
                pending = None
                continue
            out.append((pending, True))  # flush bare number as its own heading
            pending = None
        if is_heading and _NUMBER_ONLY.match(text):
            pending = text
            continue
        out.append((text, is_heading))
    if pending is not None:
        out.append((pending, True))
    return out


def extract_pdf(pdf_path: Path) -> tuple[list[PageRecord], ExtractionReport]:
    """Extract one PDF into page-anchored, section-aware line records."""
    doc = pymupdf.open(pdf_path)
    body_size = _body_font_size(doc)

    pages: list[PageRecord] = []
    current_section = "unknown"
    heading_count = 0
    heading_order: list[str] = []
    chars_total = 0

    for index, page in enumerate(doc, start=1):
        page_height = page.rect.height
        rendered_lines: list[tuple[str, bool]] = []

        data = page.get_text("dict", sort=True)
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                if not _horizontal(line):
                    continue  # rotated watermark text
                text, bold, size = _line_text_and_bold(line)
                if not text:
                    continue
                bbox_top = line["bbox"][1]
                bbox_bottom = line["bbox"][3]
                in_footer = bbox_top >= page_height - 60
                if in_footer and _PURE_PAGE_NUMBER.match(text):
                    continue  # page-number footer noise
                if bbox_bottom <= 40 and _PURE_PAGE_NUMBER.match(text):
                    continue  # page-number header noise
                rendered_lines.append(
                    (
                        text,
                        _is_heading_candidate(
                            text, bold, size, body_size, index, x0=line["bbox"][0]
                        ),
                    )
                )

        rendered_lines = _merge_number_split_headings(rendered_lines)

        page_lines: list[PageLine] = []
        for text, is_heading in rendered_lines:
            if is_heading:
                current_section = text
                heading_count += 1
                heading_order.append(text)
                page_lines.append(PageLine(t=text, h=True, s=current_section))
            else:
                page_lines.append(PageLine(t=text, h=False, s=current_section))
            chars_total += len(text)
        pages.append(PageRecord(page=index, lines=page_lines))

    doc.close()
    report = ExtractionReport(
        paper_key="",
        versioned_id="",
        n_pages=len(pages),
        n_heading_lines=heading_count,
        headings_in_order=heading_order,
        chars_total=chars_total,
    )
    log.info(
        "extraction_done",
        extra={"pages": len(pages), "headings": heading_count, "chars": chars_total},
    )
    return pages, report


def load_page_records(path: Path) -> list[PageRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PageRecord.model_validate(obj) for obj in data["pages"]]


def save_page_records(
    path: Path, records: list[PageRecord], spec: PaperSpec, report: ExtractionReport
) -> None:
    """Write the pages JSON with enough metadata to power citation checks
    (including exact n_pages for later page-existence validation)."""
    report = report.model_copy(update={"paper_key": spec.key, "versioned_id": spec.versioned_id})
    payload = {
        "paper": {
            "key": spec.key,
            "title": spec.title,
            "arxiv_id": spec.arxiv_id,
            "version": spec.version,
            "versioned_id": spec.versioned_id,
            "n_pages": len(records),
        },
        "report": report.model_dump(),
        "pages": [page.model_dump() for page in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
