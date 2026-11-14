"""Verify that printed footer page numbers match PDF page indices.

Run:  .venv\\Scripts\\python.exe -m scripts.verify_pagenums

Measured result on 2026-11-10 (evidence for citation integrity, see report):
  attention 1706.03762v7: 14/15 footers == PDF index, 1 page prints no number, 0 mismatches
  moshi     2410.00037v1: 66/66 match
  hnet      2507.07955v2: 39/39 match
So this project's "page" citations mean PDF page index AND printed page number.
"""
from __future__ import annotations

import re
import sys

sys.path.insert(0, ".")

import pymupdf

from app.core.logging_setup import force_utf8_stdio
from app.core.settings import get_settings
from app.retrieval.papers_registry import PAPERS

_DIGITS = re.compile(r"^\s*(\d{1,3})\s*$")


def main() -> None:
    papers_dir = get_settings().papers_dir
    for spec in PAPERS:
        pdf_path = papers_dir / "original_pdfs" / spec.pdf_filename
        doc = pymupdf.open(str(pdf_path))
        matches: list[tuple[int, int]] = []
        mismatches: list[tuple[int, int]] = []
        no_footer: list[int] = []
        for index, page in enumerate(doc, start=1):
            height = page.rect.height
            footer = page.get_text("text", clip=pymupdf.Rect(0, height - 60, page.rect.width, height))
            footer_lines = [line.strip() for line in footer.splitlines() if line.strip()]
            if not footer_lines:
                no_footer.append(index)
                continue
            nums = [int(m.group(1)) for m in (_DIGITS.match(line) for line in footer_lines) if m]
            if not nums:
                no_footer.append(index)
                continue
            printed = nums[0]
            if printed == index:
                matches.append((index, printed))
            else:
                mismatches.append((index, printed))
        print(
            f"{spec.key:<10} {spec.versioned_id:<15} pages={len(doc)} "
            f"footer==pdf-index: {len(matches):>3}  mismatched: {len(mismatches):>3} "
            f"no-printed-number: {len(no_footer):>3}"
        )
        if mismatches[:5]:
            print("   sample mismatches (pdf_index, printed):", mismatches[:5])
        doc.close()


if __name__ == "__main__":
    main()
