"""Per-request guardrails (Phase-1 pipeline budgets).

A `RequestBudget` is created per request by the orchestrator (WP7) and
passes its claims to pipeline stages. Raising guards as typed errors keeps
the honest-outcome mapping in one place: exceeded budget is a DEFINED
result (recorded row), not a crash.

All wall-clock checks go through an injectable clock; tests script it.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from app.core.settings import PipelineBudgets, RetryPolicy
from app.reliability.errors import (
    DeadlineExceeded,
    ModelCallsExhausted,
    RetrievalRoundsExhausted,
    RetryBudgetExhausted,
)

PERF_CLOCK = time.perf_counter


class RequestBudget:
    """One request's wall-clock deadline + call/round/retry allowances."""

    def __init__(
        self,
        pipeline: PipelineBudgets,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], float] = PERF_CLOCK,
        start_at: float | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.retry_policy = retry_policy or RetryPolicy()
        self.clock = clock
        self.started_at = self.clock() if start_at is None else start_at
        self.deadline_at = self.started_at + pipeline.request_deadline_ms / 1000.0
        self.model_calls = 0
        self.retries_used = 0
        self.retrieval_rounds = 0

    # ------------------------------------------------------------- claims

    def check_deadline(self) -> None:
        """Raise DeadlineExceeded if the wall-clock budget ran out."""
        now = self.clock()
        if now >= self.deadline_at:
            raise DeadlineExceeded(
                "request deadline reached",
                elapsed_ms=round((now - self.started_at) * 1000, 1),
            )

    def claim_model_call(self) -> None:
        self.check_deadline()
        self.model_calls += 1
        if self.model_calls > self.pipeline.max_model_calls:
            raise ModelCallsExhausted(
                f"max_model_calls={self.pipeline.max_model_calls} reached",
                calls=self.model_calls,
            )

    def claim_retrieval_round(self) -> None:
        self.retrieval_rounds += 1
        if self.retrieval_rounds > self.pipeline.max_retrieval_rounds:
            raise RetrievalRoundsExhausted(
                f"max_retrieval_rounds={self.pipeline.max_retrieval_rounds} reached",
                rounds=self.retrieval_rounds,
            )

    def retry_allowance_left(self) -> int:
        return max(
            0, self.retry_policy.retries_per_request_budget - self.retries_used
        )

    def claim_retry(self) -> None:
        """Consume one retry from the per-request allowance; raising
        `RetryBudgetExhausted` when none is left (the retry executor treats
        a zero allowance as a guard, not as silent skipping)."""
        if self.retry_allowance_left() <= 0:
            raise RetryBudgetExhausted(
                f"retry allowance {self.retry_policy.retries_per_request_budget} spent",
                retries_used=self.retries_used,
            )
        self.retries_used += 1

    # -------------------------------------------------------- introspection

    def remaining_ms(self) -> float:
        return max(0.0, (self.deadline_at - self.clock()) * 1000)

    def summary(self) -> dict:
        """Structured view for logs / the results row; nothing fabricated."""
        return {
            "started_at": self.started_at,
            "remaining_ms": round(self.remaining_ms(), 1),
            "model_calls": self.model_calls,
            "retries_used": self.retries_used,
            "retrieval_rounds": self.retrieval_rounds,
        }
