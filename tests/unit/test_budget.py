"""Unit tests for per-request budget guardrails (scripted clock)."""
import pytest

from app.core.settings import PipelineBudgets, RetryPolicy
from app.reliability.budget import RequestBudget
from app.reliability.errors import (
    DeadlineExceeded,
    ModelCallsExhausted,
    RetrievalRoundsExhausted,
    RetryBudgetExhausted,
)


class ScriptedClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _budget(clock: ScriptedClock, **overrides) -> RequestBudget:
    pipeline_fields = [
        key
        for key in (
            "max_model_calls",
            "max_retrieval_rounds",
            "request_deadline_ms",
        )
    ]
    retry_fields = ["retries_per_request_budget"]
    pipeline_kwargs = {k: overrides.pop(k) for k in pipeline_fields if k in overrides}
    retry_kwargs = {k: overrides.pop(k) for k in retry_fields if k in overrides}
    pipeline = PipelineBudgets(**pipeline_kwargs)
    retry_policy = RetryPolicy(**retry_kwargs)
    return RequestBudget(pipeline, retry_policy=retry_policy, clock=clock)


def test_model_call_claims_count_and_guard():
    clock = ScriptedClock()
    budget = _budget(clock, max_model_calls=2)
    budget.claim_model_call()
    budget.claim_model_call()
    with pytest.raises(ModelCallsExhausted):
        budget.claim_model_call()
    assert budget.summary()["model_calls"] == 3  # attempt happened, was refused


async def test_deadline_exceeded_at_exact_wall():
    clock = ScriptedClock(start=0.0)
    budget = _budget(clock, request_deadline_ms=1000.0)
    budget.check_deadline()  # fine at start
    clock.advance(0.999)
    budget.check_deadline()  # still inside (strictly before deadline)
    clock.advance(0.001)  # == deadline boundary
    with pytest.raises(DeadlineExceeded):
        budget.check_deadline()


def test_claim_model_call_checks_deadline_first():
    clock = ScriptedClock(start=0.0)
    budget = _budget(clock, request_deadline_ms=500.0)
    clock.advance(0.6)
    with pytest.raises(DeadlineExceeded):
        budget.claim_model_call()


def test_retrieval_rounds_guard():
    budget = _budget(ScriptedClock(), max_retrieval_rounds=2)
    budget.claim_retrieval_round()
    budget.claim_retrieval_round()
    with pytest.raises(RetrievalRoundsExhausted):
        budget.claim_retrieval_round()


def test_retry_allowance_claim_and_introspection():
    budget = _budget(ScriptedClock(), retries_per_request_budget=3)
    budget.claim_retry()
    budget.claim_retry()
    assert budget.retry_allowance_left() == 1
    budget.claim_retry()
    with pytest.raises(RetryBudgetExhausted):
        budget.claim_retry()
