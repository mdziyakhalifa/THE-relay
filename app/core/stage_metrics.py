"""Per-stage timing for the orchestration pipeline.

Rules of use (measurement discipline):
- Durations are wall-clock milliseconds recorded per stage invocation. A stage
  name used multiple times (retrieval loops, regeneration) ACCUMULATES a list,
  so retry/repeat counts stay visible instead of being averaged away.
- The clock is injectable: unit tests pass a scripted fake clock, so timing
  assertions are exact, not sleep-based flakiness.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

Clock = Callable[[], float]


def perf_clock() -> float:
    return time.perf_counter()


@dataclass
class StageMetrics:
    """Duration tracker for named pipeline stages within one request."""

    clock: Clock = field(default=perf_clock)
    _open: dict[str, float] = field(default_factory=dict)
    durations_ms: dict[str, list[float]] = field(default_factory=dict)

    def begin(self, stage: str) -> None:
        if stage in self._open:
            raise RuntimeError(f"stage {stage!r} is already open")
        self._open[stage] = self.clock()

    def end(self, stage: str) -> float:
        if stage not in self._open:
            raise KeyError(f"stage {stage!r} was never begun")
        started_at = self._open.pop(stage)
        duration_ms = (self.clock() - started_at) * 1000.0
        self.durations_ms.setdefault(stage, []).append(round(duration_ms, 3))
        return self.durations_ms[stage][-1]

    @asynccontextmanager
    async def timed(self, stage: str) -> AsyncIterator[str]:
        """``async with metrics.timed("retrieval"):`` — records even on error."""
        self.begin(stage)
        try:
            yield stage
        finally:
            if stage in self._open:  # error path still closes cleanly
                self.end(stage)

    def count(self, stage: str) -> int:
        return len(self.durations_ms.get(stage, []))

    def snapshot(self) -> dict[str, float]:
        """Stage -> summed duration in ms (rounded), sorted by stage name."""
        return {
            stage: round(sum(durations), 3)
            for stage, durations in sorted(self.durations_ms.items())
        }

    @property
    def total_ms(self) -> float:
        return round(sum(sum(v) for v in self.durations_ms.values()), 3)
