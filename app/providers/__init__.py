"""Provider transport layer (WP5): Groq via HTTPX (D1) + local stand-in.

Design boundaries:
- These clients are PURE TRANSPORT: build request, parse response, raise
  typed errors, expose captured rate-limit headers. Reliability (retry,
  breaker, limiter, budgets) wraps them at the orchestrator level (WP7) —
  a transport must not retry on its own.
- The API key is read from the environment at construction time and lives
  ONLY in the HTTP Authorization header (never logged, never in settings).
- Header semantics per Groq documentation (fetched 2026-11-10):
  x-ratelimit-limit-requests = RPD, x-ratelimit-limit-tokens = TPM,
  remaining-*/reset-* always sent, retry-after only with 429. The
  minute-vs-daily 429 classification below is DOCUMENTED-HEADER-based and
  must be confirmed against MEASURED signatures at WP9 (D8 revisit rule).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx
import pydantic

from app.core.logging_setup import get_logger
from app.reliability.errors import (
    AuthenticationError,
    MalformedRequestError,
    ProviderServerError,
    RateLimitDailyError,
    RateLimitMinuteError,
    TransientNetworkError,
)
from app.reliability.limiter import ProviderLimitsSnapshot

log = get_logger("relay.providers")

GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

_HEADER_LIMIT_REQUESTS = "x-ratelimit-limit-requests"  # documented: RPD
_HEADER_REMAINING_REQUESTS = "x-ratelimit-remaining-requests"
_HEADER_LIMIT_TOKENS = "x-ratelimit-limit-tokens"  # documented: TPM
_HEADER_REMAINING_TOKENS = "x-ratelimit-remaining-tokens"
_HEADER_RETRY_AFTER = "retry-after"

_MAX_BODY_SNIPPET = 500


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str


class ChatUsage(pydantic.BaseModel):
    """Provider-REPORTED usage only; absent fields stay absent."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(pydantic.BaseModel):
    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: ChatUsage | None = None
    limits: ProviderLimitsSnapshot  # always captured (headers documented present)
    transport_ms: float  # MEASURED wall time of this HTTP call


class ChatChunk(pydantic.BaseModel):
    content_delta: str
    finish_reason: str | None = None
    usage: ChatUsage | None = None  # reported on the final chunk when available


class LLMProvider(ABC):
    """Common interface both Groq and the stand-in implement."""

    name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> ChatResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]: ...


def limits_from_headers(headers: httpx.Headers) -> ProviderLimitsSnapshot:
    """Parse the documented x-ratelimit-* set; unknown stays None."""

    def _int(name: str) -> int | None:
        value = headers.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            log.warning("unparseable_rate_limit_header", extra={"header": name})
            return None

    return ProviderLimitsSnapshot(
        rpd_limit=_int(_HEADER_LIMIT_REQUESTS),
        rpd_remaining=_int(_HEADER_REMAINING_REQUESTS),
        tpm_limit=_int(_HEADER_LIMIT_TOKENS),
        tpm_remaining=_int(_HEADER_REMAINING_TOKENS),
    )


def retry_after_from_headers(headers: httpx.Headers) -> float | None:
    value = headers.get(_HEADER_RETRY_AFTER)
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def raise_for_status(status_code: int, headers: httpx.Headers, body: str) -> None:
    """Map an HTTP status to the typed taxonomy (single place, WP5).

    429 classification (D8): the documented remaining-REQUESTS counter maps
    to the DAILY RPD budget; zero there means stop, everything else is a
    per-minute/per-window condition honored via retry-after. CONFIRM at WP9.
    """
    snippet = body[:_MAX_BODY_SNIPPET]
    if status_code in (401, 403):
        raise AuthenticationError(
            f"provider rejected credentials ({status_code})", status_code=status_code, body=snippet
        )
    if status_code == 429:
        remaining_requests = None
        value = headers.get(_HEADER_REMAINING_REQUESTS)
        if value is not None and value.lstrip("-").isdigit():
            remaining_requests = int(value)
        retry_after = retry_after_from_headers(headers)
        if remaining_requests == 0:
            raise RateLimitDailyError(
                "daily request quota exhausted (remaining-requests == 0)",
                retry_after_s=retry_after,
                body=snippet,
            )
        raise RateLimitMinuteError(
            "rate limit hit (per-minute/window)",
            retry_after_s=retry_after,
            body=snippet,
        )
    if 500 <= status_code <= 599:
        raise ProviderServerError(status_code, f"provider server error {status_code}", body=snippet)
    if status_code >= 400:
        raise MalformedRequestError(
            f"request rejected with {status_code}", status_code=status_code, body=snippet
        )


def wrap_transport_error(error: BaseException) -> Exception:
    """Network-level httpx failures -> typed transient errors."""
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return TransientNetworkError(
            type(error).__name__, detail=str(error)[:_MAX_BODY_SNIPPET]
        )
    return error  # not ours to translate
