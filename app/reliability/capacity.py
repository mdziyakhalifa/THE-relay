"""Admission control: in-flight cap + bounded waiting queue (Phase 1).

Semantics:
- Up to ``in_flight_limit`` requests are RUNNING simultaneously.
- When all slots are taken, additional requests WAIT; at most
  ``queue_limit`` may wait. Beyond that the request is REJECTED with
  `CapacityRejected` — backpressure is an honest, immediately-measurable
  outcome, not a silent pile-up (results.csv records it as rejected).
- Single-loop asyncio; counters are exact because the decision section
  between increment and acquire never awaits.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.logging_setup import get_logger
from app.core.settings import AppCapacity
from app.reliability.errors import CapacityRejected

log = get_logger("relay.reliability.capacity")


class RequestGate:
    def __init__(self, capacity: AppCapacity, name: str = "requests") -> None:
        if capacity.in_flight_limit < 1:
            raise ValueError("in_flight_limit must be >= 1")
        if capacity.queue_limit < 0:
            raise ValueError("queue_limit must be >= 0")
        self.in_flight_limit = capacity.in_flight_limit
        self.queue_limit = capacity.queue_limit
        self.name = name
        self._sem = asyncio.Semaphore(capacity.in_flight_limit)
        self._waiting = 0
        self._in_flight = 0

    # ---------------------------------------------------------- inspection

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return self._waiting

    # ------------------------------------------------------------- control

    async def admit(self) -> bool:
        """Take a slot (immediately or via the queue) or raise CapacityRejected.
        Returns True when this admission had to WAIT (queued before running)."""
        would_wait = self._in_flight >= self.in_flight_limit
        if would_wait:
            if self._waiting >= self.queue_limit:
                log.warning(
                    "capacity_rejected",
                    extra={"gate": self.name, "waiting": self._waiting},
                )
                raise CapacityRejected(
                    f"{self.name}: queue limit {self.queue_limit} reached",
                    waiting=self._waiting,
                    in_flight=self._in_flight,
                )
            self._waiting += 1
        try:
            await self._sem.acquire()
        except asyncio.CancelledError:
            # cancellation while queued: undo the waiting counter or the
            # queue-wait evidence (D13) leaks forever after
            if would_wait:
                self._waiting -= 1
            raise
        self._in_flight += 1
        if would_wait:
            self._waiting -= 1
        log.info(
            "capacity_admitted",
            extra={
                "gate": self.name,
                "in_flight": self._in_flight,
                "waited": would_wait,
            },
        )
        return would_wait

    def release(self) -> None:
        if self._in_flight <= 0:
            raise RuntimeError("release() without an admitted request")
        self._in_flight -= 1
        self._sem.release()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[bool]:
        admitted_after_wait = await self.admit()
        try:
            yield admitted_after_wait
        finally:
            self.release()
