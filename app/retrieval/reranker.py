"""Optional cross-encoder reranker (Phase 1 config C only).

A cross-encoder reads (query, chunk) pairs JOINTLY and outputs a relevance
score. It is more accurate than bi-encoder cosine but costs one forward pass
per candidate, so it only ever sees the top-N candidates already selected by
the cheap index search — never the whole library.

Honesty notes (measured + recorded, do not silently change):
- MEASURED on 2026-11-10 with cross-encoder/ms-marco-MiniLM-L-6-v2 and the
  installed library version: the checkpoint's stored activation is
  ``Identity()`` (not the library's num_labels=1 sigmoid default), so
  ``predict`` returns RAW LOGITS — unbounded, e.g. -6.18/-11.37 for an
  on-topic vs off-topic pair. We use them ONLY to re-order the bi-encoder
  survivors. ``similarity_threshold`` applies ONLY to bi-encoder cosine;
  no threshold is ever applied to cross-encoder logits. Any report must
  label which scorer produced a shown number.
- The real model (~80 MB, offline after first download) is NEVER imported by
  unit tests; tests inject a stub predictor via ``_predict_for_tests`` or use
  a fake reranker.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from app.core.logging_setup import get_logger
from app.retrieval.chunker import Chunk

log = get_logger("relay.retrieval.reranker")

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # lazy: big download + torch import on first use
        self._lock = threading.Lock()

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder  # deferred import

                started = time.perf_counter()
                self._model = CrossEncoder(self.model_name)
                load_ms = (time.perf_counter() - started) * 1000
                log.info(
                    "reranker_loaded",
                    extra={"model": self.model_name, "load_ms": round(load_ms, 1)},
                )
            return self._model

    def _predict_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Raw scores for (query, text) pairs. Overridable in tests."""
        model = self._ensure_model()
        with self._lock:
            scores = model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(
        self,
        query: str,
        scored_chunks: list[tuple[float, Chunk]],
        top_k: int,
    ) -> list[tuple[float, Chunk]]:
        """Re-sort bi-encoder survivors by cross-encoder score, best first.

        The input scores (bi-encoder cosine) are discarded from the output —
        keeping them alongside cross-encoder logits would invite threshold
        confusion. Returns at most ``top_k`` pairs scored by the
        cross-encoder (raw logits, see module docstring).
        """
        if not scored_chunks or top_k <= 0:
            return []
        if not query or not query.strip():
            raise ValueError("empty query cannot be reranked")
        started = time.perf_counter()
        scores = self._predict_pairs([(query, chunk.text) for _, chunk in scored_chunks])
        if scores.shape[0] != len(scored_chunks):
            raise RuntimeError("reranker returned wrong number of scores")
        order = np.argsort(-scores)
        out: list[tuple[float, Chunk]] = [
            (float(scores[position]), scored_chunks[int(position)][1]) for position in order
        ]
        log.info(
            "rerank_done",
            extra={
                "n_candidates": len(scored_chunks),
                "kept": min(top_k, len(out)),
                "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return out[:top_k]
