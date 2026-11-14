"""The stand-in's DOCUMENTED tokenizer — the token-counting contract.

This is the single source of truth for how the stand-in service computes
the `usage` block it REPORTS to clients. The request recorder (D13)
aggregates only provider-REPORTED usage; for the stand-in that reported
usage is produced HERE, deterministically, by this exact specification.

Contract (fixed as of 2026-11-11, WP6 — changes require a decision-log
entry because results.csv token columns depend on it):

1. A "token" is one match of the regex  r"\\w+|[^\\w\\s]"  over the text:
   every maximal run of word characters (letters, digits, underscore —
   Unicode-aware, so "café" is one token) is one token, and every single
   non-word, non-whitespace character (punctuation) is one token.
   Whitespace is never a token and never contributes.
2. Empty or whitespace-only text counts as 0 tokens.
3. Chat framing overhead: each message contributes
   count_tokens(role) + count_tokens(content) + PER_MESSAGE_OVERHEAD
   tokens, where PER_MESSAGE_OVERHEAD = 4 stands for the role/content
   delimiters and per-message framing an OpenAI-style chat template adds.
   This overhead is an APPROXIMATION by design and is documented as such —
   the point is determinism and reproducibility, not OpenAI parity.
4. `prompt_tokens`  = sum over messages of the rule in (3).
5. `completion_tokens` = count_tokens(generated completion text) exactly.
6. `total_tokens` = prompt_tokens + completion_tokens.

Determinism guarantee: the same request always produces the same usage
numbers — the service holds no hidden state that affects counting.
"""
from __future__ import annotations

import re
from typing import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]")

PER_MESSAGE_OVERHEAD_TOKENS = 4
DEFAULT_COMPLETION_TOKENS = 64


def encode(text: str) -> list[str]:
    """The documented token list for one text (rule 1/2)."""
    return TOKEN_RE.findall(text or "")


def count_tokens(text: str) -> int:
    """The documented token count for one text (rule 1/2)."""
    return len(encode(text))


def prompt_tokens(messages: Iterable[dict]) -> int:
    """prompt_tokens over chat messages (rule 3/4).

    Messages are `{role, content}` dicts (validated by the service before
    this is called).
    """
    total = 0
    for message in messages:
        total += (
            count_tokens(str(message.get("role", "")))
            + count_tokens(str(message.get("content", "")))
            + PER_MESSAGE_OVERHEAD_TOKENS
        )
    return total
