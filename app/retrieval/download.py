"""Download the exact arXiv PDFs with integrity checks.

Integrity contract per file:
- served bytes start with the PDF magic header (``%PDF``)
- size >= 50_000 bytes (guard against error pages)
- SHA-256 recorded in the manifest so a later re-run can detect changes
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pydantic

from app.core.logging_setup import get_logger
from app.retrieval.papers_registry import PaperSpec

log = get_logger("relay.retrieval.download")

_MIN_PDF_BYTES = 50_000
_MANIFEST_NAME = "download_manifest.json"


class DownloadResult(pydantic.BaseModel):
    key: str
    versioned_id: str
    url: str
    file: str  # relative to papers/original_pdfs
    sha256: str
    size_bytes: int
    downloaded_at: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


async def download_paper(spec: PaperSpec, papers_dir: Path) -> DownloadResult:
    """Download one paper unless a valid file already exists; verify integrity."""
    original_dir = papers_dir / "original_pdfs"
    original_dir.mkdir(parents=True, exist_ok=True)
    dest = original_dir / spec.pdf_filename

    if dest.is_file() and dest.stat().st_size >= _MIN_PDF_BYTES:
        log.info("download_skip_existing", extra={"paper": spec.key, "file": dest.name})
        return DownloadResult(
            key=spec.key,
            versioned_id=spec.versioned_id,
            url=spec.pdf_url,
            file=dest.name,
            sha256=sha256_of(dest),
            size_bytes=dest.stat().st_size,
            downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        # Short-lived client per paper; papers are tiny (~0.5-3 MB total).
        response = await client.get(spec.pdf_url, headers={"User-Agent": "the-relay-ingest/0.1"})
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            raise RuntimeError(
                f"{spec.versioned_id}: expected application/pdf, got '{content_type}'"
            )
        if not response.content.startswith(b"%PDF"):
            raise RuntimeError(f"{spec.versioned_id}: response does not start with %PDF magic")
        if len(response.content) < _MIN_PDF_BYTES:
            raise RuntimeError(
                f"{spec.versioned_id}: suspiciously small download ({len(response.content)} B)"
            )
        dest.write_bytes(response.content)
        size = len(response.content)
        log.info(
            "download_ok",
            extra={
                "paper": spec.key,
                "versioned_id": spec.versioned_id,
                "bytes": size,
                "sha256": hashlib.sha256(response.content).hexdigest()[:12],
            },
        )

    return DownloadResult(
        key=spec.key,
        versioned_id=spec.versioned_id,
        url=spec.pdf_url,
        file=dest.name,
        sha256=sha256_of(dest),
        size_bytes=dest.stat().st_size,
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


async def download_all(papers_dir: Path) -> list[DownloadResult]:
    """Sequential downloads (3 small files — parallelism buys nothing here)."""
    from app.retrieval.papers_registry import PAPERS

    results = [await download_paper(spec, papers_dir) for spec in PAPERS]
    papers_dir / "original_pdfs"  # touch for readability
    manifest_path = papers_dir / "original_pdfs" / _MANIFEST_NAME
    manifest_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8"
    )
    log.info("download_manifest_written", extra={"file": str(manifest_path)})
    return results
