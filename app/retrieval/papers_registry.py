"""The exact paper set mandated by the brief, pinned to exact arXiv versions.

Facts verified against arxiv.org abs pages on 2026-11-10 (see
docs/decision_log.md verified-fact log):
- 1706.03762v7 exists (latest version, 2 Aug 2023)
- 2410.00037v1 exists (17 Sep 2024); v2 is arXiv's current version, but the
  project brief pins v1 — recorded decision, not an oversight
- 2507.07955v2 exists (15 Jul 2025, latest)

Citations produced anywhere in this project must use arxiv_id + version
exactly as pinned here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperSpec:
    key: str
    title: str
    arxiv_id: str  # e.g. "1706.03762"
    version: str  # e.g. "v7"

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}{self.version}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.versioned_id}"

    @property
    def pdf_filename(self) -> str:
        return f"{self.key}_{self.arxiv_id}{self.version}.pdf"


PAPERS: tuple[PaperSpec, ...] = (
    PaperSpec("attention", "Attention Is All You Need", "1706.03762", "v7"),
    PaperSpec(
        "moshi",
        "Moshi: a speech-text foundation model for real-time dialogue",
        "2410.00037",
        "v1",
    ),
    PaperSpec(
        "hnet",
        "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "2507.07955",
        "v2",
    ),
)

PAPERS_BY_KEY = {p.key: p for p in PAPERS}
VERSIONED_IDS = {p.versioned_id for p in PAPERS}


def is_known_versioned_id(value: str) -> bool:
    """Used by citation verification (WP7): the cited paper must exist in
    THIS library with the exact version."""
    return value in VERSIONED_IDS
