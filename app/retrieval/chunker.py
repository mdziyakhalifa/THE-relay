"""Page-pinned chunking (decision D4).

A chunk NEVER crosses a PDF page boundary. This is what makes page citations
exact-by-construction; the cost (slightly more chunks, text flow split at
page joins) is accepted and documented.

Token sizing is an ESTIMATE: chars/4. Exact model tokenizers need hosted
vocab files; the properties that actually matter here are page pinning
(exact) and relative chunk size (±15% tolerant). Never present these counts
as exact model tokens.

Overlap carries the tail of a closed chunk into the next chunk ON THE SAME
PAGE only. Overlap cannot cross pages because chunks don't.
"""
from __future__ import annotations

import pydantic

from app.retrieval.extract import PageRecord
from app.retrieval.papers_registry import PaperSpec

CHARS_PER_TOKEN = 4.0  # documented estimate, see module docstring


class ChunkConfig(pydantic.BaseModel):
    chunk_tokens: int = 500
    overlap_tokens: int = 80
    min_chunk_tokens: int = 40
    # a trailing fragment below min merges into the previous same-page chunk
    # only when the merge stays within chunk_tokens * 1.25


class Chunk(pydantic.BaseModel):
    chunk_id: str
    paper_key: str
    title: str
    versioned_id: str  # arXiv id + exact version, e.g. "1706.03762v7"
    page: int  # exact PDF page index (1-based) — pinned, never inferred
    section: str  # section in force at the chunk's first line; else "unknown"
    text: str
    chunk_index_in_page: int
    est_tokens: int


def _token_estimate(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _join_lines(parts: list[str]) -> str:
    out: list[str] = []
    for text in parts:
        if out and out[-1].endswith("-") and text and text[0].islower():
            out.append(text)  # de-hyphenate "tokeniza-" + "tion"
        else:
            if out:
                out.append(" ")
            out.append(text)
    return "".join(out).strip()


def _append_chunk(
    chunks: list[Chunk],
    spec: PaperSpec,
    page: int,
    idx: int,
    lines: list[tuple[str, str]],
    section: str | None,
) -> int:
    text = _join_lines([t for t, _ in lines])
    if not text:
        return idx
    chunks.append(
        Chunk(
            chunk_id=f"{spec.key}|{spec.versioned_id}|p{page}|{idx}",
            paper_key=spec.key,
            title=spec.title,
            versioned_id=spec.versioned_id,
            page=page,
            section=section if section is not None else "unknown",
            text=text,
            chunk_index_in_page=idx,
            est_tokens=_token_estimate(text),
        )
    )
    return idx + 1


def _pick_tail(
    lines: list[tuple[str, str]], overlap_tokens: int
) -> tuple[list[tuple[str, str]], int]:
    """Last lines that fit the overlap budget (at least one line)."""
    tail: list[tuple[str, str]] = []
    used = 0
    for text, section in reversed(lines):
        tokens = _token_estimate(text)
        if tail and used + tokens > overlap_tokens:
            break
        tail.insert(0, (text, section))
        used += tokens
    return tail, used


def _merge_tiny_trailing_chunk(
    chunks: list[Chunk], page: int, config: ChunkConfig, spec: PaperSpec
) -> None:
    """If the last chunk of a page is a tiny fragment, fold it into the
    previous chunk of the SAME page when the combined text still fits."""
    page_chunks = [c for c in chunks if c.page == page]
    if len(page_chunks) < 2:
        return
    tiny, prev = page_chunks[-1], page_chunks[-2]
    if tiny.est_tokens >= config.min_chunk_tokens:
        return
    merged_text = _join_lines([prev.text, tiny.text])
    merged_est = _token_estimate(merged_text)
    if merged_est > config.chunk_tokens * 1.25:
        return
    chunks.remove(prev)
    chunks.remove(tiny)
    chunks.append(
        prev.model_copy(update={"text": merged_text, "est_tokens": merged_est})
    )


def chunk_page_records(
    pages: list[PageRecord], spec: PaperSpec, config: ChunkConfig
) -> list[Chunk]:
    """Page-pinned, section-aware, overlap-carrying chunks for one paper."""
    chunks: list[Chunk] = []
    for page_record in pages:
        page = page_record.page
        idx = 0
        open_lines: list[tuple[str, str]] = []
        open_section: str | None = None

        for text, section in ((line.t, line.s) for line in page_record.lines):
            if not open_lines:
                open_section = section  # section in force BEFORE first line
            open_lines.append((text, section))
            total = sum(_token_estimate(t) for t, _ in open_lines)
            if total > config.chunk_tokens:
                idx = _append_chunk(chunks, spec, page, idx, open_lines, open_section)
                tail, _used = _pick_tail(open_lines, config.overlap_tokens)
                open_lines = tail
                open_section = tail[0][1] if tail else None

        if open_lines:
            idx = _append_chunk(chunks, spec, page, idx, open_lines, open_section)

        _merge_tiny_trailing_chunk(chunks, page, config, spec)

    chunks.sort(key=lambda c: (c.page, c.chunk_index_in_page))
    unique_ids = {c.chunk_id for c in chunks}
    if len(unique_ids) != len(chunks):  # integrity guard for citation lookups
        raise RuntimeError("duplicate chunk_id generated")
    return chunks
