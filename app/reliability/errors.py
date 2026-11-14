"""Typed, centralized error taxonomy for provider access.

Every other reliability component and both providers (WP5) raise/consume THESE
classes only. This is where "what is retryable" is decided exactly once:

- RETRYABLE (transient, worth another attempt): transient network faults,
  provider 5xx, per-minute saturation (429 limited to RPM/TPM window).
- NON-RETRYABLE (retries cannot fix, retrying is waste or harm):
  daily/idle-window quota exhaustion (D8: NEVER retried), auth failures,
  malformed requests, capacity rejections from OUR OWN queue (backpressure is
  a routing decision, not something to hammer), and per-request budget
  guards.

Local-only errors (capacity/budget) do not describe the provider; they are
NOT inputs to the circuit breaker either (see breaker).
"""
from __future__ import annotations


class RelayError(Exception):
    """Root of the project's own error taxonomy (distinguishable from stdlib)."""

    attrs: dict[str, object] = {}

    def __init__(self, message: str, **attrs: object) -> None:
        super().__init__(message)
        self.attrs = attrs

    def __repr__(self) -> str:  # logs keep the structured view useful
        return f"{type(self).__name__}({self!s},{self.attrs})"


# ---------------------------------------------------------------- retryable


class RetryableProviderError(RelayError):
    """Base for provider faults worth another attempt after backoff."""

    retry_after_s: float | None
    """From a provider `retry-after` header when one was sent (else None)."""

    def __init__(self, message: str, retry_after_s: float | None = None, **attrs) -> None:
        self.retry_after_s = retry_after_s
        if "retry_after_s" not in attrs:
            attrs["retry_after_s"] = retry_after_s
        super().__init__(message, **attrs)


class TransientNetworkError(RetryableProviderError):
    """Timeouts, connect/read failures — the network path itself."""


class ProviderServerError(RetryableProviderError):
    """HTTP 5xx — provider-side transient failure."""

    status_code: int

    def __init__(self, status_code: int, message: str, **attrs) -> None:
        self.status_code = status_code
        super().__init__(message, status_code=status_code, **attrs)


class RateLimitMinuteError(RetryableProviderError):
    """HTTP 429 limited to the per-minute window (RPM or TPM exceeded).

    Retried with backoff honoring `retry-after` (D8); per-minute, NOT daily.
    Deciding minute-vs-daily is the provider adapter's job at parse time
    (signatures to be MEASURED at WP9 — nothing is inferred here).
    """


# ------------------------------------------------------------ non-retryable


class NonRetryableProviderError(RelayError):
    """Base for permanent-for-now faults; the retry policy raises through."""

    status_code: int | None
    retry_after_s: float | None

    def __init__(
        self,
        message: str,
        retry_after_s: float | None = None,
        status_code: int | None = None,
        **attrs: object,
    ) -> None:
        self.retry_after_s = retry_after_s
        self.status_code = status_code
        attrs.setdefault("retry_after_s", retry_after_s)
        attrs.setdefault("status_code", status_code)
        super().__init__(message, **attrs)


class RateLimitDailyError(NonRetryableProviderError):
    """HTTP 429 exhausting an RPD/TPD (or other sustained) quota — D8 says the
    answer is STOP, not backoff. Reset expected to be reviewed from the
    next response's x-ratelimit-reset-* headers (re-read every call)."""


class AuthenticationError(NonRetryableProviderError):
    """HTTP 401/403 — bad/absent key; retries cannot fix configuration."""


class MalformedRequestError(NonRetryableProviderError):
    """HTTP 4xx that is OUR bug (400/404/422) — do not burn the budget on it."""

# ---------------------------------------------------------- capacity/budget


class CapacityRejected(RelayError):
    """Request refused by OUR bounded queue (backpressure). Not a provider
    fault; the orchestrator maps it to an honest rejected outcome row."""


class BreakerOpen(RelayError):
    """Call refused by OUR circuit breaker (cooldown or half-open cap).
    Not a provider fault; retried locally it would only hammer a sick
    provider. `ready_at_s` (monotonic) is included when known."""


class QuotaBlock(RelayError):
    """Call refused by OUR local quota projection (observed headers say the
    org's remaining requests/tokens cannot cover this call, incl. daily
    exhaustion per D8 — no request is sent). Not a provider fault."""


class BudgetExceeded(RelayError):
    """Base for per-request guardrails (deadline, model calls, retries)."""


class DeadlineExceeded(BudgetExceeded):
    """Wall-clock request deadline hit."""


class ModelCallsExhausted(BudgetExceeded):
    """max_model_calls reached for one request."""


class RetryBudgetExhausted(BudgetExceeded):
    """Per-request retry allowance spent; further retries are not offered."""


class RetrievalRoundsExhausted(BudgetExceeded):
    """max_retrieval_rounds reached."""


class UserCancelled(RelayError):
    """Request abandoned/cancelled by the user while detectable. Not a
    provider fault and never a breaker input; recorded as an honest
    outcome row (status `error`, typed identity `user_cancelled`)."""


class ConfigurationError(RelayError):
    """Our own setup is wrong (missing API key, bad endpoint). Retries
    cannot fix configuration; surfaced before any traffic is sent."""


def is_provider_fault(err: BaseException) -> bool:
    """Circuit-breaker input policy: provider-health signals only.

    Counted (D20, user-approved 2026-11-12): transient network faults and
    5xx — signals that the PROVIDER ITSELF is sick. NOT counted: quota
    429s (RateLimitMinuteError + RateLimitDailyError — quota pressure is
    an expected business condition owned by the limiter/retry layer, not
    provider sickness; a 429 storm must produce honest rate_limited rows,
    not a health-trip), auth/malformed requests, our own capacity/budget
    guards.

    Revision history: WP4 originally counted per-minute 429s ("sustained
    saturation is a breaker concern"); WP10 load measurement showed a
    minute-429 storm ending in 19 fast breaker_open refusals, the user
    chose the exclusion semantics, and the minute429 scenario was
    re-measured clean (D19 measured the old behavior, D20 replaced it).
    """
    return isinstance(err, TransientNetworkError | ProviderServerError)
