"""Citation validator (WP7) — provenance honesty for delivered answers.

Contract (D16): a valid citation tag is a bracketed span containing
`arXiv:<versioned_id>` and `p.<page>` — the same provenance the retrieval
evidence blocks carry (`[Title — arXiv:<id>, p.<n>, §<section>]`), so the
model copies real tags rather than inventing forms.

Validation rules for a NON-refusal answer:
- ZERO citations found  -> INVALID (uncited: claims without provenance).
- Any cited (id, page) pair NOT present in the delivered evidence -> INVALID
  (page-pinned chunks (D4) make this check exact — a wrong page can never
  be explained away).
- Only citations pointing at pages actually delivered -> VALID.

Refusals are not validated (no citations expected); the caller flags them
as refusals and WP11 grades refused_correctly / refused_wrongly.
"""
from __future__ import annotations

import re

import pydantic

CITATION_RE = re.compile(
    r"\[[^\]]*?arXiv:(?P<id>\d{4}\.\d{4,5}v\d+)[^\]]*?p\.\s*(?P<page>\d+)[^\]]*\]",
    re.IGNORECASE,
)


class CitedRef(pydantic.BaseModel):
    arxiv_id: str
    page: int


class CitationReport(pydantic.BaseModel):
    valid: bool
    uncited: bool
    cited: list[CitedRef]
    unknown: list[CitedRef]  # cited but not in the delivered evidence

    @property
    def has_problems(self) -> bool:
        return not self.valid


def validate_citations(answer: str, delivered_items: list) -> CitationReport:
    """Check an answer's citations against the evidence actually delivered.

    `delivered_items` are EvidenceItem-like objects with versioned_id/page.
    """
    known: set[tuple[str, int]] = {
        (item.versioned_id, int(item.page)) for item in delivered_items
    }
    cited: list[CitedRef] = []
    unknown: list[CitedRef] = []
    for match in CITATION_RE.finditer(answer or ""):
        ref = CitedRef(arxiv_id=match.group("id"), page=int(match.group("page")))
        cited.append(ref)
        if (ref.arxiv_id, ref.page) not in known:
            unknown.append(ref)
    uncited = not cited
    valid = (not uncited) and (not unknown)
    return CitationReport(valid=valid, uncited=uncited, cited=cited, unknown=unknown)
