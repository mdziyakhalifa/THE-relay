"""Build the three Phase-1 retrieval indexes (configs A/B/C).

Run:  .venv\\Scripts\\python.exe -m scripts.build_index

Chunk grammar (documented in the decision log): page-pinned chunks with
(no rerank here — C reranks at query time via the cross-encoder).

  A: 300-token chunks, 50 overlap, top-3
  B: 500-token chunks, 80 overlap, top-5   (the app default)
  C: 800-token chunks, 120 overlap, top-5 + cross-encoder rerank (flag only)

Requires papers/extracted_text/<key>.pages.json from
``python -m scripts.ingest_papers``. The embedding model downloads on first
run (~90 MB, offline afterwards). Output: papers/indexes/config_<a|b|c>/ with
matrix.npy + chunks.jsonl + meta.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.logging_setup import configure_logging, force_utf8_stdio, get_logger
from app.core.settings import get_settings
from app.retrieval.chunker import ChunkConfig, chunk_page_records
from app.retrieval.embedder import Embedder
from app.retrieval.extract import load_page_records
from app.retrieval.indexer import RetrievalIndex
from app.retrieval.papers_registry import PAPERS_BY_KEY

log = get_logger("relay.scripts.build_index")


@dataclass(frozen=True)
class IndexBuildSpec:
    name: str  # "A" | "B" | "C"
    directory: str  # papers/indexes/<directory>
    chunk_tokens: int
    overlap_tokens: int


CONFIGS: tuple[IndexBuildSpec, ...] = (
    IndexBuildSpec("A", "config_a", 300, 50),
    IndexBuildSpec("B", "config_b", 500, 80),
    IndexBuildSpec("C", "config_c", 800, 120),
)


def main() -> None:
    force_utf8_stdio()
    configure_logging("INFO")
    settings = get_settings()
    papers_dir = Path(settings.papers_dir)
    extracted_dir = papers_dir / "extracted_text"
    indexes_dir = papers_dir / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    # integrity: config B must equal the settings-defined app default
    if (settings.retrieval.chunk_size_tokens, settings.retrieval.chunk_overlap_tokens) != (
        500,
        80,
    ):
        log.warning(
            "config_b_drift",
            extra={
                "settings_chunk_tokens": settings.retrieval.chunk_size_tokens,
                "settings_overlap_tokens": settings.retrieval.chunk_overlap_tokens,
            },
        )

    pages_by_key: dict[str, tuple] = {}
    for spec in PAPERS_BY_KEY:
        pages_path = extracted_dir / f"{spec}.pages.json"
        if not pages_path.is_file():
            print(f"MISSING {pages_path}\nRun: python -m scripts.ingest_papers")
            raise SystemExit(1)
        payload = json.loads(pages_path.read_text(encoding="utf-8"))
        stored = payload["paper"]["versioned_id"]
        if stored != PAPERS_BY_KEY[spec].versioned_id:
            raise RuntimeError(
                f"{pages_path}: stored versioned_id {stored!r} != registry "
                f"{PAPERS_BY_KEY[spec].versioned_id!r}"
            )
        pages_by_key[spec] = (load_page_records(pages_path), PAPERS_BY_KEY[spec])

    embedder = Embedder()  # one model load reused by every config
    embedder.dimension  # forces the (possibly first-time) download/load now

    per_paper_counts: dict[str, list[int]] = {}
    for build in CONFIGS:
        chunk_config = ChunkConfig(
            chunk_tokens=build.chunk_tokens, overlap_tokens=build.overlap_tokens
        )
        chunks = []
        per_paper_counts[build.name] = []
        for key, (pages, spec) in pages_by_key.items():
            paper_chunks = chunk_page_records(pages, spec, chunk_config)
            per_paper_counts[build.name].append(len(paper_chunks))
            print(
                f"config {build.name}: {key} -> {len(paper_chunks)} chunks "
                f"({build.chunk_tokens}tok/{build.overlap_tokens}overlap) (measured)"
            )
            chunks.extend(paper_chunks)

        started = time.perf_counter()
        index = RetrievalIndex.build(
            chunks,
            embedder,
            config_name=build.name,
            chunk_tokens=build.chunk_tokens,
            overlap_tokens=build.overlap_tokens,
        )
        embed_ms = (time.perf_counter() - started) * 1000
        out_dir = indexes_dir / build.directory
        index.save(out_dir)
        print(
            f"config {build.name}: built n_chunks={len(chunks)} dim={index.meta.dim} "
            f"embed={embed_ms:.0f} ms (measured) -> {out_dir}"
        )

    print("\nper-paper chunk counts [attention, moshi, hnet] by config:")
    for name, counts in per_paper_counts.items():
        print(f"  config {name}: {counts}  total={sum(counts)}")


if __name__ == "__main__":
    main()
