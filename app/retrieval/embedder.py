"""Local embedding model wrapper (all-MiniLM-L6-v2 by default).

Free, offline after first download, documented. The model name is a settings
knob, not a hardcode of library behavior. Encoding is CPU-bound, so callers
inside the request path must run it in a worker thread (asyncio.to_thread) —
the retriever enforces that; CLI scripts may call the sync methods directly.

Note: the installed sentence-transformers encode() has NO ``is_query``
parameter; unknown kwargs would be forwarded unpredictably. MiniLM is a
symmetric bi-encoder (queries and documents embed identically), so we pass
none. If a future asymmetric model is configured, its query/document asymmetry
comes via ``prompt_name``, and this wrapper must be updated then.
"""
from __future__ import annotations

import threading

import numpy as np

from app.core.logging_setup import get_logger

log = get_logger("relay.retrieval.embedder")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Sentence-transformers wrapper; encode() is serialized via a lock because
    torch modules are not guaranteed thread-safe for concurrent calls."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # lazy: big download + torch import on first use
        self._lock = threading.Lock()

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # deferred import

                started_at = __import__("time").perf_counter()
                self._model = SentenceTransformer(self.model_name)
                load_ms = (__import__("time").perf_counter() - started_at) * 1000
                log.info(
                    "embedder_loaded",
                    extra={"model": self.model_name, "load_ms": round(load_ms, 1)},
                )
            return self._model

    @property
    def dimension(self) -> int:
        """Hidden size of the embedding model (384 for MiniLM defaults)."""
        model = self._ensure_model()
        # newer sentence-transformers renamed the getter; use the new name
        # when present to avoid the FutureWarning
        get_dimension = getattr(model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = model.get_sentence_embedding_dimension
        width = get_dimension()
        if width is None:
            raise RuntimeError("embedding model did not report a dimension")
        return int(width)

    def encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalized float32 embeddings, shape [n, dim]."""
        model = self._ensure_model()
        with self._lock:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("empty query cannot be embedded")
        return self.encode([query.strip()])[0]
