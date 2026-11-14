"""Flat NumPy cosine-similarity index (decision D2).

Trade-off (documented): exhaustive dot-product over ~10^2-10^3 chunks is
microseconds on modern hardware and needs zero native deps — but this design
does not scale to 10^5+ chunks. Revisit trigger is recorded in the decision
log. Cosine here equals dot because all vectors (and the query) are
L2-normalized at embedding time. Persisted as .npy + JSONL so the exact index
used in every experiment can be reproduced and inspected.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydantic

from app.retrieval.chunker import Chunk


class IndexMeta(pydantic.BaseModel):
    config_name: str
    chunk_tokens: int
    overlap_tokens: int
    model_name: str
    n_chunks: int
    dim: int
    built_at: str


class RetrievalIndex:
    """Immutable chunk store + normalized matrix; search returns
    (score, chunk) pairs sorted by descending score."""

    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, meta: IndexMeta) -> None:
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2-D [n_chunks, dim]")
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"matrix rows {matrix.shape[0]} != chunks {len(chunks)}"
            )
        self.chunks = chunks
        self.matrix = matrix
        self.meta = meta

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder,  # Embedder-compatible object (duck-typed for tests)
        config_name: str,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> "RetrievalIndex":
        if not chunks:
            raise ValueError("cannot build an empty index")
        matrix = embedder.encode([c.text for c in chunks])
        norm = np.linalg.norm(matrix, axis=1)
        if np.any(np.isclose(norm, 0.0)):
            raise ValueError("embedding produced a zero vector")
        matrix = (matrix / norm[:, None]).astype(np.float32)  # defensive re-normalization
        meta = IndexMeta(
            config_name=config_name,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            model_name=getattr(embedder, "model_name", "unknown"),
            n_chunks=len(chunks),
            dim=int(matrix.shape[1]),
            built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        return cls(chunks, matrix, meta)

    def search(
        self, query_vector: np.ndarray, top_k: int, min_score: float
    ) -> list[tuple[float, Chunk]]:
        """top_k results above min_score, best first; threshold filters."""
        if query_vector.ndim != 1 or query_vector.shape[0] != self.matrix.shape[1]:
            raise ValueError("query vector shape mismatch")
        scores = self.matrix @ query_vector  # cosine, both normalized
        order = np.argsort(-scores)
        results: list[tuple[float, Chunk]] = []
        for position in order:
            score = float(scores[position])
            if score < min_score:
                break
            results.append((score, self.chunks[int(position)]))
            if len(results) >= top_k:
                break
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "matrix.npy", self.matrix)
        (directory / "chunks.jsonl").write_text(
            "\n".join(c.model_dump_json() for c in self.chunks), encoding="utf-8"
        )
        (directory / "meta.json").write_text(self.meta.model_dump_json(), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "RetrievalIndex":
        matrix = np.load(directory / "matrix.npy")
        chunks = [
            Chunk.model_validate(json.loads(line))
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        meta = IndexMeta.model_validate_json(
            (directory / "meta.json").read_text(encoding="utf-8")
        )
        return cls(chunks, matrix, meta)
