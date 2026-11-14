"""Unit tests for the circuit breaker (scripted clock; no real time)."""
import asyncio

import pytest

from app.core.settings import BreakerPolicy
from app.reliability.breaker import CircuitBreaker, State
from app.reliability.errors import (
    AuthenticationError,
    BreakerOpen,
    ProviderServerError,
    RateLimitDailyError,
    RateLimitMinuteError,
    TransientNetworkError,
)


class ScriptedClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _breaker(clock: ScriptedClock, **kwargs) -> CircuitBreaker:
    defaults = dict(failure_threshold=3, failure_window_s=30.0, open_seconds=20.0, half_open_max_calls=1)
    defaults.update(kwargs)
    return CircuitBreaker(BreakerPolicy(**defaults), clock=clock, name="test")


async def _raise_or_ok(error: BaseException | None) -> str:
    if error is not None:
        raise error
    return "ok"


async def test_opens_after_threshold_then_refuses():
    clock, breaker = ScriptedClock(), None
    breaker = _breaker(clock)
    err = ProviderServerError(500, "boom")

    for _ in range(3):
        with pytest.raises(ProviderServerError):
            await breaker.call(_raise_or_ok, err)
    assert breaker.state is State.OPEN

    with pytest.raises(BreakerOpen):
        await breaker.call(_raise_or_ok, None)
    assert breaker.seconds_until_ready > 0


async def test_failure_window_pruned_by_clock():
    clock = ScriptedClock()
    breaker = _breaker(clock, failure_threshold=3, failure_window_s=10.0)
    for _ in range(2):
        with pytest.raises(ProviderServerError):
            await breaker.call(_raise_or_ok, ProviderServerError(500, "e"))
    clock.advance(11.0)  # old failures leave the window
    with pytest.raises(ProviderServerError):
        await breaker.call(_raise_or_ok, ProviderServerError(500, "fresh"))
    assert breaker.state is State.CLOSED, "one fresh failure cannot trip a 3-threshold breaker"


async def test_half_open_success_closes_without_logrecord_name_crash():
    """WP10 finding: `extra={"name": ...}` is a reserved LogRecord attribute;
    makeRecord raises KeyError whenever INFO logging is ENABLED — pytest's
    WARNING default hid it, the WP10 load run measured it as
    unexpected_KeyError on the FIRST successful half-open probe."""
    import logging

    logger = logging.getLogger("relay.reliability.breaker")
    clock = ScriptedClock()
    breaker = _breaker(clock)
    original_level = logger.level
    logger.setLevel(logging.INFO)  # force makeRecord along the success path
    try:
        for _ in range(3):
            with pytest.raises(ProviderServerError):
                await breaker.call(_raise_or_ok, ProviderServerError(500, "e"))
        clock.advance(25.0)  # cooldown over -> next call is the half-open probe
        assert await breaker.call(_raise_or_ok, None) == "ok"
        assert breaker.state is State.CLOSED
    finally:
        logger.setLevel(original_level)


async def test_cooldown_elapsed_allows_exactly_one_concurrent_probe():
    clock = ScriptedClock()
    breaker = _breaker(clock)
    for _ in range(3):
        with pytest.raises(ProviderServerError):
            await breaker.call(_raise_or_ok, ProviderServerError(503, "e"))
    clock.advance(25.0)  # cooldown over

    released = asyncio.Event()

    async def probe() -> str:
        await released.wait()
        return "ok"

    task = asyncio.create_task(breaker.call(probe))
    await asyncio.sleep(0)  # let the task take the single half-open spot
    with pytest.raises(BreakerOpen):
        await breaker.call(_raise_or_ok, None)  # the spot is claimed
    released.set()
    assert await task == "ok"
    assert breaker.state is State.CLOSED, "a clean half-open probe closes the breaker"


async def test_half_open_failure_reopens_with_fresh_cooldown():
    clock = ScriptedClock()
    breaker = _breaker(clock)
    for _ in range(3):
        with pytest.raises(ProviderServerError):
            await breaker.call(_raise_or_ok, ProviderServerError(500, "e"))
    clock.advance(30.0)
    open_until_first = breaker._open_until

    with pytest.raises(ProviderServerError):
        await breaker.call(_raise_or_ok, ProviderServerError(500, "probe failed"))
    assert breaker.state is State.OPEN
    assert breaker._open_until >= clock.now, "new cooldown measured from the failure"
    assert breaker._open_until != open_until_first


async def test_success_clears_consecutive_failures():
    clock = ScriptedClock()
    breaker = _breaker(clock, failure_threshold=3)
    for _ in range(2):
        with pytest.raises(ProviderServerError):
            await breaker.call(_raise_or_ok, ProviderServerError(500, "e"))
    assert await breaker.call(_raise_or_ok, None) == "ok"
    assert breaker.failures_in_window == 0
    with pytest.raises(ProviderServerError):
        await breaker.call(_raise_or_ok, ProviderServerError(500, "one is not enough"))
    assert breaker.state is State.CLOSED


async def test_non_provider_faults_are_not_counted():
    clock = ScriptedClock()
    breaker = _breaker(clock, failure_threshold=3)
    for _ in range(3):
        with pytest.raises(AuthenticationError):
            await breaker.call(_raise_or_ok, AuthenticationError("bad key"))
        with pytest.raises(RateLimitDailyError):
            await breaker.call(_raise_or_ok, RateLimitDailyError("day exhausted"))
    assert breaker.state is State.CLOSED, "business conditions never trip the breaker"


async def test_minute_rate_limit_is_excluded_from_breaker_counting():
    """D20 (user-approved 2026-11-12): quota 429s are a limiter/retry concern,
    not provider health. A sustained minute-429 storm must produce honest
    rate_limited rows — never a health-trip (D19 measured the OLD behavior:
    19 breaker_open refusals; replaced by this semantics)."""
    clock = ScriptedClock()
    breaker = _breaker(clock, failure_threshold=2)
    for _ in range(4):  # more than the threshold would ever need
        with pytest.raises(RateLimitMinuteError):
            await breaker.call(_raise_or_ok, RateLimitMinuteError("saturation"))
    assert breaker.state is State.CLOSED
    assert breaker.failures_in_window == 0
    # ...and a real fault right after still counts normally
    with pytest.raises(ProviderServerError):
        await breaker.call(_raise_or_ok, ProviderServerError(500, "e"))
    assert breaker.failures_in_window == 1
