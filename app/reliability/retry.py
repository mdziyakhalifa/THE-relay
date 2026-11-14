"""Retry policy executor (decision D8 + Phase-1 retry policy).

One place decides "retry or raise": RETRYABLE provider errors get exponential
backoff with bounded jitter; NON-retryable errors (daily quota per D8, auth,
malformed requests) pass through untouched. All sleeps go through an injected
async sleeper so tests are exact and nothing here ever blocks for real.

The per-request retry allowance is a NUMBER given by the caller (WP7's budget
owns it); when retries would exceed it, `RetryBudgetExhausted` is raised
instead of retrying silently past the guarantee.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from app.core.logging_setup import get_logger
from app.core.settings import RetryPolicy
from app.reliability.errors import (
    NonRetryableProviderError,
    RateLimitDailyError,
    RetryBudgetExhausted,
    RetryableProviderError,
)

log = get_logger("relay.reliability.retry")


class RetryExecutor:
    """Runs ``func()`` until success, a non-retryable error, or attempt/budget
    exhaustion. `attempts_allotted` defaults to the policy's per-call bound."""

    def __init__(
        self,
        policy: RetryPolicy,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        rng: random.Random | None = None,
        on_retry: Callable[[int, float, BaseException], None] | None = None,
    ) -> None:
        self.policy = policy  # retained so callers can derive per-run bounds
        # default sleeper: real asyncio.sleep — tests ALWAYS inject their own
        self._sleeper = sleeper or (lambda seconds: asyncio.sleep(seconds))
        self._rng = rng or random.Random()
        self._on_retry = on_retry

    def _delay_for(self, attempt_failure: int, error: RetryableProviderError) -> float:
        """Backoff after the Nth failure (0-based): base * factor^N with
        bounded jitter (+-jitter_fraction). Honoring `retry-after` when the
        provider sent one: take the LARGER of the two — never shorter than
        the provider requested, so a provider's 0-second answer cannot
        collapse the backoff either."""
        delay = self.policy.backoff_base_s * (self.policy.backoff_factor**attempt_failure)
        jittered = delay * self._rng.uniform(
            1.0 - self.policy.jitter_fraction, 1.0 + self.policy.jitter_fraction
        )
        if error.retry_after_s is not None and error.retry_after_s > jittered:
            jittered = error.retry_after_s
        return jittered

    async def run(
        self,
        func: Callable[[], Awaitable],
        *,
        attempts_allotted: int | None = None,
        request_retry_allowance: int | None = None,
        label: str = "provider call",
    ):
        attempts_left = (
            attempts_allotted if attempts_allotted is not None else self.policy.max_attempts
        )
        retries_allowed = request_retry_allowance  # None => no request-level cap
        attempt_number = 0
        while True:
            try:
                attempt_number += 1
                return await func()
            except RateLimitDailyError:
                raise  # D8: daily quota exhaustion is never retried
            except RetryableProviderError as error:
                if attempt_number >= attempts_left:
                    raise
                if retries_allowed is not None:
                    if retries_allowed <= 0:
                        raise RetryBudgetExhausted(
                            f"{label}: retry allowance spent",
                            after_attempt=attempt_number,
                        ) from error
                    retries_allowed -= 1
                delay = self._delay_for(attempt_number - 1, error)
                if self._on_retry is not None:
                    self._on_retry(attempt_number, delay, error)
                log.debug(
                    "retry_scheduled",
                    extra={
                        "label": label,
                        "attempt": attempt_number,
                        "delay_s": round(delay, 3),
                        "error": type(error).__name__,
                    },
                )
                await self._sleeper(delay)
            except NonRetryableProviderError:
                raise
            # anything else (bugs, budget guards) passes through untouched
