"""Extract pages (page-anchored, section-aware) + chunk preview + library index.

Run:  .venv\\Scripts\\python.exe -m scripts.ingest_papers

Produces per run:
- papers/extracted_text/<key>.pages.json   (one per paper; feeds chunking)
- papers/extracted_text/chunks_preview.jsonl (default-config chunk preview)
- papers/extracted_text/library_index.json   (metadata powering citation checks)
and prints quality-inspection material: heading sequence, token stats, and one
sample chunk per paper.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app.core.logging_setup import configure_logging, get_logger
from app.core.logging_setup import configure_logging, force_utf8_stdio
from app.core.settings import get_settings
from app.retrieval.chunker import ChunkConfig, chunk_page_records
from app.retrieval.download import sha256_of
from app.retrieval.extract import ExtractionReport, extract_pdf, save_page_records
from app.retrieval.papers_registry import PAPERS, PaperSpec

log = get_logger("relay.scripts.ingest")

MAX_HEADINGS_SHOWN = 60


def _print_heading_table(spec: PaperSpec, headings: list[str]) -> None:
    print(f"\n[{spec.key}] detected heading sequence ({len(headings)} total):")
    for i, heading in enumerate(headings[:MAX_HEADINGS_SHOWN], 1):
        print(f"   {i:>3}. {heading}")
    if len(headings) > MAX_HEADINGS_SHOWN:
        print(f"   ... ({len(headings) - MAX_HEADINGS_SHOWN} more)")


def main() -> None:
    force_utf8_stdio()
    configure_logging("INFO")
    settings = get_settings()
    papers_dir = Path(settings.papers_dir)
    extracted_dir = papers_dir / "extracted_text"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    library_index: dict[str, dict] = {}
    all_preview_chunks: list[dict] = []

    # default config = retrieval defaults from settings (config B in Phase 1)
    chunk_config = ChunkConfig(
        chunk_tokens=settings.retrieval.chunk_size_tokens,
        overlap_tokens=settings.retrieval.chunk_overlap_tokens,
    )

    for spec in PAPERS:
        pdf_path = papers_dir / "original_pdfs" / spec.pdf_filename
        if not pdf_path.is_file():
            print(f"MISSING PDF: {pdf_path}\nRun: python -m scripts.download_papers")
            raise SystemExit(1)

        started = time.perf_counter()
        pages, report = extract_pdf(pdf_path)
        extraction_ms = (time.perf_counter() - started) * 1000

        pages_path = extracted_dir / f"{spec.key}.pages.json"
        save_page_records(pages_path, pages, spec, report)
        log.info("pages_saved", extra={"paper": spec.key, "file": str(pages_path)})

        started = time.perf_counter()
        chunks = chunk_page_records(pages, spec, chunk_config)
        chunking_ms = (time.perf_counter() - started) * 1000
        all_preview_chunks.extend(chunk.model_dump() for chunk in chunks)

        token_stats = [c.est_tokens for c in chunks]
        library_index[spec.key] = {
            "key": spec.key,
            "title": spec.title,
            "arxiv_id": spec.arxiv_id,
            "version": spec.version,
            "versioned_id": spec.versioned_id,
            "n_pages": len(pages),
            "n_headings": report.n_heading_lines,
            "n_chunks_default_config": len(chunks),
            "chars_total": report.chars_total,
            "pdf_sha256": sha256_of(pdf_path),
            "pdf_filename": spec.pdf_filename,
        }

        print(f"\n=== {spec.key} — {spec.title} ({spec.versioned_id}) ===")
        print(
            f"pages={len(pages)} chars={report.chars_total:,} "
            f"headings={report.n_heading_lines} chunks(default cfg)={len(chunks)} "
            f"extract={extraction_ms:.0f} ms chunk={chunking_ms:.0f} ms (measured)"
        )
        if token_stats:
            print(
                f"chunk est_tokens: min={min(token_stats)} median={statistics.median(token_stats):.0f} "
                f"max={max(token_stats)}"
            )
        first_chunk = next((c for c in chunks if c.page == 1), None)
        if first_chunk:
            print(f"sample chunk page 1 (section={first_chunk.section!r}):")
            print("   " + first_chunk.text[:400].replace("\n", " "))

        _print_heading_table(spec, report.headings_in_order)

    (extracted_dir / "chunks_preview.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_preview_chunks),
        encoding="utf-8",
    )
    (extracted_dir / "library_index.json").write_text(
        json.dumps(library_index, indent=2), encoding="utf-8"
    )
    print(
        f"\nWrote {len(all_preview_chunks)} preview chunks and library_index.json "
        f"to {extracted_dir}"
    )


if __name__ == "__main__":
    main()
