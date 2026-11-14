"""Prompt builders + the understanding-output contract (WP7).

D5: question understanding and retrieval-sufficiency classification are ONE
structured-JSON call (quota discipline on the free tier). The parser here is
DEFENSIVE by design: model output that is not the exact JSON shape is never
trusted — it maps to `None` and the runner decides honestly what that means
(recorded decision, D16): with evidence present, an unparseable
understanding does NOT fabricate a refusal; the citation validator backstops
the delivered answer instead.

The answer prompt mandates the citation tag form the validator understands:
a bracketed tag containing `arXiv:<versioned_id>` and `p.<page>` — exactly
the provenance the evidence blocks carry, so the model can copy them.
"""
from __future__ import annotations

import json

UNDERSTANDING_SYSTEM = (
    "You are the understanding stage of a retrieval pipeline that answers "
    "questions ONLY from three pinned arXiv papers (1706.03762v7, "
    "2410.00037v1, 2507.07955v2). Judge whether the EVIDENCE below contains "
    "enough to answer the QUESTION faithfully. Reply with STRICT JSON only, "
    "no prose, no code fences: "
    '{"sufficient": <true|false>, "refined_search_terms": <string or null>, '
    '"reason": <short string>}. '
    "sufficient=false requires refined_search_terms for one retrieval retry."
)

ANSWER_SYSTEM = (
    "You answer questions using ONLY the EVIDENCE below, taken from three "
    "pinned arXiv papers. Every factual claim must carry a citation tag "
    "copied from the evidence, in the form [arXiv:<id> p.<page>]. Do not "
    "use outside knowledge. If the evidence is insufficient, state that you "
    "cannot answer the question from the pinned papers and cite nothing."
)


def understanding_user_prompt(question: str, evidence_context: str) -> str:
    return (
        f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_context}\n\n"
        "Reply with the JSON object only."
    )


def answer_user_prompt(question: str, evidence_context: str) -> str:
    return f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_context}"


def parse_understanding(content: str) -> dict | None:
    """Parse the understanding call's output; None when not the exact shape.

    Accepted shape (all validated): {"sufficient": bool,
    "refined_search_terms": str|null, "reason": str}. Extra keys ignored;
    code fences stripped; the first {...} block wins.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("sufficient"), bool):
        return None
    refined = data.get("refined_search_terms")
    if refined is not None and not isinstance(refined, str):
        return None
    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = ""
    return {
        "sufficient": data["sufficient"],
        "refined_search_terms": refined or None,
        "reason": reason,
    }
