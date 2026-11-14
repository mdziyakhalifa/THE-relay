"""Qualitative retrieval spot-check (NOT a formal evaluation, WP11 is).

Run:
    .venv\\Scripts\\python.exe -m scripts.retrieval_spotcheck            # config B
    .venv\\Scripts\\python.exe -m scripts.retrieval_spotcheck --config a # or c

Loads the chosen index, fires fixed probe questions, and prints the top-3
hits with score / page / section plus MEASURED latency numbers. This is an
eyeball-quality instrument: it shows what a real request would retrieve; it
does not claim recall or accuracy numbers — the eval harness comes in WP11.

Config C additionally runs the cross-encoder reranker (extra model download
on first use). Its score is RAW LOGITS (unbounded — MEASURED Identity()
activation on this checkpoint), NOT cosine; only ordering is meaningful.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from app.core.logging_setup import configure_logging, force_utf8_stdio
from app.core.settings import get_settings
from app.retrieval.embedder import Embedder
from app.retrieval.indexer import IndexMeta, RetrievalIndex
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever

CONFIG_DIRS = {
    "a": ("config_a", False),
    "b": ("config_b", False),
    "c": ("config_c", True),
}

# handwritten qualitative probes; 3 per paper + 1 cross-paper
PROBES: tuple[tuple[str, str], ...] = (
    ("attention", "How does multi-head attention extend the attention mechanism?"),
    ("attention", "Why is the learning rate warmed up and then decayed?"),
    ("attention", "Which weight initialization does the base Transformer use?"),
    ("moshi", "How does Moshi talk and listen at the same time?"),
    ("moshi", "What are semantic and acoustic audio tokens in Mimi?"),
    ("hnet", "What does dynamic chunking replace in hierarchical models?"),
    ("hnet", "How are chunk boundaries selected robustly?"),
    ("cross", "How are encoder and decoder used differently in these models?"),
)

SNIPPET_CHARS = 110
TOP_TO_PRINT = 3


async def _run(config: str) -> None:
    settings = get_settings()
    dir_name, rerank_enabled = CONFIG_DIRS[config]
    index_dir = Path(settings.papers_dir) / "indexes" / dir_name
    if not (index_dir / "meta.json").is_file():
        print(f"MISSING index at {index_dir}\nRun: python -m scripts.build_index")
        raise SystemExit(1)

    meta = IndexMeta.model_validate_json((index_dir / "meta.json").read_text(encoding="utf-8"))
    print(
        f"index: config {meta.config_name} ({dir_name}) chunks={meta.n_chunks} "
        f"dim={meta.dim} model={meta.model_name}\n"
    )

    embedder = Embedder()  # default model (all-MiniLM-L6-v2), matches built indexes
    reranker = CrossEncoderReranker() if rerank_enabled else None
    retriever = Retriever(index_dir, embedder, reranker=reranker)

    # one warmup so the table below excludes one-time model/index loading
    await retriever.retrieve("warmup probe to force index and model load")

    score_label = "rel_logit" if rerank_enabled else "cos"
    defaults = settings.retrieval
    print(
        f"settings: top_k={defaults.top_k} threshold={defaults.similarity_threshold} "
        f"max_context_tokens={defaults.max_context_tokens} rerank={rerank_enabled}\n"
    )

    for label, query in PROBES:
        started = time.perf_counter()
        result = await retriever.retrieve(query)
        total_ms = (time.perf_counter() - started) * 1000
        print(f"[{label}] {query}")
        if not result.items:
            print("   (no hits above threshold)")
        for rank, item in enumerate(result.items[:TOP_TO_PRINT], 1):
            snippet = item.text[:SNIPPET_CHARS].replace("\n", " ")
            print(
                f"   {rank}. {score_label}={item.score:.3f} {item.paper_key} p.{item.page} "
                f"§{item.section}: {snippet}"
            )
        print(
            f"   hits={len(result.items)} kept_tokens={result.total_est_tokens} "
            f"total={total_ms:.0f} ms (measured)"
        )


def main() -> None:
    force_utf8_stdio()
    configure_logging("WARNING")  # keep the table readable; timings come from the table
    parser = argparse.ArgumentParser(description="retrieval spot-check")
    parser.add_argument("--config", choices=sorted(CONFIG_DIRS), default="b")
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()
