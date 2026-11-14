"""Local quota limiter driven by OBSERVED provider headers (decisions D8).

Honesty-first design:
- The limiter NEVER assumes free-tier limits. Until headers arrive it is in
  observe-only mode (everything admitted, one warning logged). Any limit
  number here comes from a response header, which Groq documents as always
  present (x-ratelimit-*).
- Projection = last-observed remaining MINUS our usage booked since that
  observation. Header observations REPLACE the projection (they are
  authoritative); documented behavior is headers on every response, so the
  projection is refreshed constantly.
- D21 (WP10, measured + user-approved): (a) the RUNNER books a call's
  usage BEFORE observing that same response's headers — an observation
  already includes the call that produced it, so the old order
  (observe -> book) double-counted it (measured: projected
  2924 - 2737 = 187 -> 8 spurious local quota blocks over 4.5 min).
  (b) Minute-window bookings are timestamped and AGE OUT of the minute
  projection after minute_window_s (default 60 s): the provider's own
  window resets without client-visible traffic, and a local block must
  not ratchet shut forever. Daily (RPD) bookings NEVER age.
- Daily exhaustion (RPD/TPD): projected remaining <= 0 means NO requests are
  sent, period (D8). The unblock signal is a NEW header observation only —
  this limiter does not guess reset times (x-ratelimit-reset-* parsing and
  time-aware unblocking can be added when measured at WP9).
- Between observations a burst can overshoot; the provider's own 429 (and
  the retry executor's per-minute handling) is the backstop. Recorded as a
  known limitation in the decision log.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from app.core.logging_setup import get_logger
from app.reliability.errors import QuotaBlock

log = get_logger("relay.reliability.limiter")


@dataclass
class ProviderLimitsSnapshot:
    """Parsed from x-ratelimit-* headers by the provider adapter (WP5).
    Every field optional: unknown stays unknown — nothing is invented."""

    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    # minute-window remaining are naturally refreshed by every response;
    # daily remaining is authoritative as observed.


@dataclass
class _Booking:
    """One locally-booked usage increment, timestamped for window aging (D21)."""

    at: float
    requests: int = 0
    tokens: int = 0


@dataclass
class _UsageSinceObservation:
    entries: list[_Booking] = field(default_factory=list)

    def add(self, at: float, *, requests: int = 0, tokens: int = 0) -> None:
        self.entries.append(_Booking(at=at, requests=requests, tokens=tokens))

    def totals(self, now: float, window_s: float) -> tuple[int, int, int, int]:
        """(all_requests, all_tokens, in_window_requests, in_window_tokens).

        Daily accounting uses the ALL totals (a daily quota does not decay);
        minute accounting uses the WINDOW totals only (D21: booked usage
        older than the provider's minute window no longer counts against
        the minute projection — the provider's own window has already
        reset, which the WP10 real probes measured server-side)."""
        all_requests = sum(b.requests for b in self.entries)
        all_tokens = sum(b.tokens for b in self.entries)
        in_window_requests = sum(
            b.requests for b in self.entries if now - b.at < window_s
        )
        in_window_tokens = sum(
            b.tokens for b in self.entries if now - b.at < window_s
        )
        return all_requests, all_tokens, in_window_requests, in_window_tokens


@dataclass
class QuotaDecision:
    allowed: bool
    reason: str = ""  # which quota projected insufficient, when refused
    observe_only: bool = False


@dataclass
class QuotaState:
    snapshot: ProviderLimitsSnapshot = field(default_factory=ProviderLimitsSnapshot)
    since: _UsageSinceObservation = field(default_factory=_UsageSinceObservation)

    def _projected(self, remaining: int | None, used: int) -> int | None:
        if remaining is None:
            return None
        return remaining - used


class HeaderQuotaLimiter:
    def __init__(
        self,
        name: str = "groq",
        clock: Callable[[], float] = time.monotonic,
        minute_window_s: float = 60.0,
    ) -> None:
        self.name = name
        self.clock = clock
        self.minute_window_s = minute_window_s
        self._state = QuotaState()
        self._warned_unobserved = False

    # ------------------------------------------------------------- observe

    def observe(self, snapshot: ProviderLimitsSnapshot) -> None:
        """Replace the projection with authoritative header values."""
        self._state.snapshot = snapshot
        self._state.since = _UsageSinceObservation()
        log.debug(
            "quota_observed",
            extra={
                "limiter": self.name,
                "rpd_remaining": snapshot.rpd_remaining,
                "tpm_remaining": snapshot.tpm_remaining,
                "rpm_remaining": snapshot.rpm_remaining,
            },
        )

    def report_usage(self, *, requests: int = 1, tokens: int = 0) -> None:
        """Book local usage between header observations (timestamped, D21)."""
        self._state.since.add(self.clock(), requests=requests, tokens=tokens)

    # --------------------------------------------------------------- admit

    def admit(self, est_tokens: int = 1) -> QuotaDecision:
        """Decide whether one provider call is allowed under the projection.
        Raises QuotaBlock on refusal (call-site maps it to an honest outcome)."""
        snapshot = self._state.snapshot
        if snapshot.rpd_limit is None and snapshot.tpm_limit is None and snapshot.rpm_limit is None:
            if not self._warned_unobserved:
                self._warned_unobserved = True
                log.warning(
                    "limiter_observe_only",
                    extra={"limiter": self.name},
                )
            return QuotaDecision(allowed=True, observe_only=True)

        est_tokens = max(1, int(est_tokens))
        now = self.clock()
        all_requests, _all_tokens, win_requests, win_tokens = self._state.since.totals(
            now, self.minute_window_s
        )

        # daily first (D8: daily exhaustion stops everything; bookings for
        # the daily window NEVER age — the day does not reset itself)
        if snapshot.rpd_remaining is not None:
            projected = snapshot.rpd_remaining - all_requests
            if projected <= 0:
                log.warning(
                    "quota_block_daily",
                    extra={"limiter": self.name, "projected": projected},
                )
                raise QuotaBlock(
                    "daily request quota projected exhausted (RPD)",
                    quota="rpd",
                    projected=projected,
                )

        if snapshot.rpm_remaining is not None:
            projected = snapshot.rpm_remaining - win_requests
            if projected <= 0:
                log.warning(
                    "quota_block_minute",
                    extra={"limiter": self.name, "projected": projected},
                )
                raise QuotaBlock(
                    "per-minute request quota projected exhausted (RPM)",
                    quota="rpm",
                    projected=projected,
                )

        if snapshot.tpm_remaining is not None:
            projected = snapshot.tpm_remaining - win_tokens
            if projected < est_tokens:
                log.warning(
                    "quota_block_tokens",
                    extra={"limiter": self.name, "projected": projected, "need": est_tokens},
                )
                raise QuotaBlock(
                    "token quota projected exhausted (TPM)",
                    quota="tpm",
                    projected=projected,
                )

        return QuotaDecision(allowed=True)

    # -------------------------------------------------------- introspection

    def summary(self) -> dict:
        snapshot = self._state.snapshot
        now = self.clock()
        all_requests, all_tokens, win_requests, win_tokens = self._state.since.totals(
            now, self.minute_window_s
        )
        return {
            "observed": any(
                v is not None
                for v in (
                    snapshot.rpd_limit,
                    snapshot.tpm_limit,
                    snapshot.rpm_limit,
                )
            ),
            "rpd_remaining": snapshot.rpd_remaining,
            "tpm_remaining": snapshot.tpm_remaining,
            "rpm_remaining": snapshot.rpm_remaining,
            "usage_since_observation": {
                "requests": all_requests,
                "tokens": all_tokens,
                "requests_in_window": win_requests,
                "tokens_in_window": win_tokens,
                "minute_window_s": self.minute_window_s,
            },
        }
