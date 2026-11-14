"""Async retrieval facade — the only retrieval entry point the pipeline uses.

All CPU-bound work (query embedding, index search, reranking) runs in worker
threads via ``asyncio.to_thread`` so a request never blocks the event loop;
the embedder's internal lock serializes the actual torch calls.

Flow per query:
  1. embed the query (thread)
  2. bi-encoder search: top_k hits above similarity_threshold (thread)
  3. optional cross-encoder rerank of those survivors (thread, config C)
  4. greedy trim to ``max_context_tokens`` (cheap, runs inline)
  5. assemble context text with citation-ready provenance blocks

Trim rule: chunks are consumed best-first while the cumulative token
estimate fits the budget. If even the FIRST chunk exceeds the budget it is
still included and a warning is logged (availability over a config error);
the result carries ``first_chunk_over_budget`` so callers can see it.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pydantic

from app.core.logging_setup import get_logger
from app.core.settings import RetrievalDefaults
from app.retrieval.chunker import CHARS_PER_TOKEN, Chunk
from app.retrieval.embedder import Embedder
from app.retrieval.indexer import RetrievalIndex

log = get_logger("relay.retrieval.retriever")


class EvidenceItem(pydantic.BaseModel):
    """One retrieved chunk, carrying every provenance field a citation needs."""

    chunk_id: str
    paper_key: str
    title: str
    versioned_id: str
    page: int
    section: str
    text: str
    est_tokens: int
    score: float  # bi-encoder cosine, or cross-encoder logits if reranked


class RetrievalResult(pydantic.BaseModel):
    query: str
    items: list[EvidenceItem]
    context_text: str  # assembled evidence pack actually handed to the LLM
    total_est_tokens: int
    n_candidates_after_threshold: int  # bi-encoder hits before rerank/trim
    reranked: bool
    first_chunk_over_budget: bool


def _block_text(item: EvidenceItem) -> str:
    return f"[{item.title} — arXiv:{item.versioned_id}, p.{item.page}, §{item.section}]\n{item.text}"


def _est_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


class Retriever:
    def __init__(
        self,
        index_dir: Path,
        embedder: Embedder,
        reranker=None,  # CrossEncoderReranker-compatible, or None
        defaults: RetrievalDefaults | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.reranker = reranker
        self.defaults = defaults or RetrievalDefaults()
        self._index: RetrievalIndex | None = None
        self._load_lock = asyncio.Lock()

    async def _get_index(self) -> RetrievalIndex:
        """Load once; concurrent callers coordinate on an asyncio lock."""
        if self._index is None:
            async with self._load_lock:
                if self._index is None:
                    started = time.perf_counter()
                    index = await asyncio.to_thread(RetrievalIndex.load, self.index_dir)
                    self._index = index
                    log.info(
                        "index_loaded",
                        extra={
                            "dir": str(self.index_dir),
                            "n_chunks": index.meta.n_chunks,
                            "config": index.meta.config_name,
                            "ms": round((time.perf_counter() - started) * 1000, 1),
                        },
                    )
        return self._index

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        max_context_tokens: int | None = None,
    ) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("empty query cannot be retrieved")
        top_k = self.defaults.top_k if top_k is None else top_k
        threshold = self.defaults.similarity_threshold if threshold is None else threshold
        max_context_tokens = (
            self.defaults.max_context_tokens
            if max_context_tokens is None
            else max_context_tokens
        )

        index = await self._get_index()

        started = time.perf_counter()
        query_vector = await asyncio.to_thread(self.embedder.encode_query, query)
        embed_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        pairs = await asyncio.to_thread(index.search, query_vector, top_k, threshold)
        search_ms = (time.perf_counter() - started) * 1000
        n_candidates = len(pairs)

        rerank_ms = 0.0
        reranked = self.reranker is not None
        if reranked:
            started = time.perf_counter()
            pairs = await asyncio.to_thread(self.reranker.rerank, query, pairs, top_k)
            rerank_ms = (time.perf_counter() - started) * 1000

        items = [
            EvidenceItem(
                chunk_id=chunk.chunk_id,
                paper_key=chunk.paper_key,
                title=chunk.title,
                versioned_id=chunk.versioned_id,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                est_tokens=chunk.est_tokens,
                score=score,
            )
            for score, chunk in pairs
        ]

        kept: list[EvidenceItem] = []
        total_tokens = 0
        first_chunk_over_budget = False
        for position, item in enumerate(items):
            block_tokens = _est_tokens(_block_text(item))
            if total_tokens + block_tokens > max_context_tokens:
                if position == 0:
                    # best chunk alone over budget: include it anyway, flag it
                    first_chunk_over_budget = True
                    log.warning(
                        "context_first_chunk_exceeds_budget",
                        extra={
                            "block_tokens": block_tokens,
                            "max_context_tokens": max_context_tokens,
                        },
                    )
                else:
                    break
            kept.append(item)
            total_tokens += block_tokens

        context_text = _assemble(kept)

        result = RetrievalResult(
            query=query,
            items=kept,
            context_text=context_text,
            total_est_tokens=total_tokens,
            n_candidates_after_threshold=n_candidates,
            reranked=reranked,
            first_chunk_over_budget=first_chunk_over_budget,
        )
        log.info(
            "retrieval_done",
            extra={
                "index_config": index.meta.config_name,
                "n_chunks": index.meta.n_chunks,
                "after_threshold": n_candidates,
                "kept": len(kept),
                "total_est_tokens": total_tokens,
                "embed_ms": round(embed_ms, 1),
                "search_ms": round(search_ms, 1),
                "rerank_ms": round(rerank_ms, 1),
                "reranked": reranked,
            },
        )
        return result


def _assemble(items: list[EvidenceItem]) -> str:
    blocks = [f"[{i + 1}] {_block_text(item)}" for i, item in enumerate(items)]
    return "\n\n".join(blocks)
