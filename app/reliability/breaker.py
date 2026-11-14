"""Circuit breaker around provider calls (Phase-1 breaker policy).

Semantics (recorded decisions):
- OPENING counts PROVIDER-HEALTH failures only (``is_provider_fault`` inside
  a `failure_window_s` sliding window). Quota 429s (minute AND daily — D20),
  auth, malformed requests, and our own capacity/budget guards are NOT
  breaker inputs — they are business conditions, not sickness.
- Success semantics are CONSECUTIVE failures: one clean success in CLOSED
  state clears the window (simple, conservative, testable).
- HALF_OPEN permits at most ``half_open_max_calls`` probes; everything else
  is refused with `BreakerOpen` until the probe resolves.
- Open-cooldown clock is injectable (monotonic seconds); tests script it.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum

from app.core.logging_setup import get_logger
from app.core.settings import BreakerPolicy
from app.reliability.errors import BreakerOpen, is_provider_fault

log = get_logger("relay.reliability.breaker")


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


MONO_CLOCK = time.monotonic


class CircuitBreaker:
    def __init__(
        self,
        policy: BreakerPolicy,
        clock: Callable[[], float] = MONO_CLOCK,
        name: str = "provider",
    ) -> None:
        self.policy = policy
        self.clock = clock
        self.name = name
        self.state = State.CLOSED
        self._failure_times: list[float] = []  # consecutive-failure window
        self._open_until = 0.0
        self._half_open_permits = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ inspection

    @property
    def failures_in_window(self) -> int:
        now = self.clock()
        self._failure_times = [
            t for t in self._failure_times if now - t < self.policy.failure_window_s
        ]
        return len(self._failure_times)

    @property
    def seconds_until_ready(self) -> float:
        if self.state is State.OPEN:
            return max(0.0, self._open_until - self.clock())
        return 0.0

    # ------------------------------------------------------------- execution

    async def call(self, func: Callable[..., Awaitable], *args, **kwargs):
        async with self._lock:
            now = self.clock()
            if self.state is State.OPEN:
                if now < self._open_until:
                    raise BreakerOpen(
                        f"{self.name}: breaker open, probe at +"
                        f"{self._open_until - now:.2f}s",
                        ready_at=self._open_until,
                    )
                self._enter_half_open(now)
            if self.state is State.HALF_OPEN:
                if self._half_open_permits <= 0:
                    raise BreakerOpen(
                        f"{self.name}: half-open call budget spent",
                        ready_at=self._open_until,
                    )
                self._half_open_permits -= 1

        try:
            result = await func(*args, **kwargs)
        except BaseException as error:
            if is_provider_fault(error):
                async with self._lock:
                    self._record_failure(self.clock())
            raise
        async with self._lock:
            self._record_success()
        return result

    # ---------------------------------------------------------- transitions

    def _enter_half_open(self, now: float) -> None:
        self.state = State.HALF_OPEN
        self._half_open_permits = self.policy.half_open_max_calls
        log.info(
            "breaker_half_open", extra={"breaker": self.name, "state": self.state}
        )

    def _record_failure(self, now: float) -> None:
        if self.state is State.OPEN:
            return  # already open; nothing new to learn
        if self.state is State.HALF_OPEN:
            self._open_until = now + self.policy.open_seconds
            self.state = State.OPEN
            log.warning(
                "breaker_open",
                extra={"breaker": self.name, "reason": "half_open_probe_failed"},
            )
            return
        self._failure_times = [
            t for t in self._failure_times if now - t < self.policy.failure_window_s
        ]
        self._failure_times.append(now)
        if len(self._failure_times) >= self.policy.failure_threshold:
            self._open_until = now + self.policy.open_seconds
            self.state = State.OPEN
            log.warning(
                "breaker_open",
                extra={
                    "breaker": self.name,
                    "failures": len(self._failure_times),
                    "window_s": self.policy.failure_window_s,
                },
            )

    def _record_success(self) -> None:
        if self.state is State.HALF_OPEN:
            self.state = State.CLOSED
            # 'name' is a reserved LogRecord attribute: extra={"name": ...}
            # crashes makeRecord (measured under WP10 load as
            # unexpected_KeyError on every half-open -> closed transition)
            log.info("breaker_closed", extra={"breaker": self.name})
        if self.state is State.CLOSED:
            self._failure_times = []  # consecutive-failure semantics
