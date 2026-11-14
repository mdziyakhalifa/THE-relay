"""Download the three exact arXiv PDFs with integrity checks.

Run:  .venv\\Scripts\\python.exe -m scripts.download_papers
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from app.core.logging_setup import configure_logging, force_utf8_stdio
from app.core.settings import get_settings
from app.retrieval.download import download_all


def main() -> None:
    force_utf8_stdio()
    configure_logging("INFO")
    settings = get_settings()
    papers_dir = Path(settings.papers_dir)

    started = time.perf_counter()
    results = asyncio.run(download_all(papers_dir))
    duration_ms = (time.perf_counter() - started) * 1000

    print("\nDownloaded / verified:")
    for r in results:
        print(f"  {r.key:<10} {r.versioned_id:<15} {r.size_bytes:>9,} B  sha256:{r.sha256[:12]}")
    print(f"took {duration_ms:.0f} ms (measured)")


if __name__ == "__main__":
    main()
