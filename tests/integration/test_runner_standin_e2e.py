"""End-to-end runner test against the REAL stand-in service (ASGI).

Proves the honest-token-accounting chain end to end: the service computes
usage with its DOCUMENTED tokenizer, the shared transport reads it, the
runner aggregates provider-REPORTED usage across stages, and the
results.csv row carries exactly those numbers — nothing estimated, nothing
invented. Also exercises a scripted rate-limit round trip through the
runner's typed-outcome mapping.
"""
import csv
import json

import httpx

from app.orchestrator.events import EventSink
from app.orchestrator.prompts import (
    ANSWER_SYSTEM,
    UNDERSTANDING_SYSTEM,
    answer_user_prompt,
    understanding_user_prompt,
)
from app.orchestrator.runner import RelayRunner
from app.providers.standin import StandInProvider
from app.reliability.capacity import RequestGate
from app.reliability.breaker import CircuitBreaker
from app.reliability.limiter import HeaderQuotaLimiter
from app.reliability.recorder import RequestRecorder
from app.core.settings import AppCapacity, BreakerPolicy, PipelineBudgets, RetryPolicy
from app.retrieval.retriever import EvidenceItem
from app.standin.service import create_app
from app.standin.tokenizer import PER_MESSAGE_OVERHEAD_TOKENS, count_tokens

TS = "2026-11-11T12:00:00.000Z"
ITEM = EvidenceItem(
    chunk_id="att-p4-1",
    paper_key="attention",
    title="Attention Is All You Need",
    versioned_id="1706.03762v7",
    page=4,
    section="3.2",
    text="Scaled dot-product attention computes the dot products of the query with all keys.",
    est_tokens=20,
    score=0.61,
)
CITED_ANSWER = "Scaled dot-product attention multiplies queries by keys [arXiv:1706.03762v7 p.4]."
QUESTION = "How does scaled dot-product attention work?"


def _messages(stage: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": stage},
        {"role": "user", "content": context},
    ]


def _runner(tmp_path, provider):
    return RelayRunner(
        retrievers={"B": _StubRetriever()},
        provider=provider,
        model="standin-sim",
        gate=RequestGate(AppCapacity()),
        breaker=CircuitBreaker(BreakerPolicy()),
        limiter=HeaderQuotaLimiter(),
        pipeline=PipelineBudgets(),
        retry_policy=RetryPolicy(),
        recorder=RequestRecorder(tmp_path / "results.csv"),
        sink=EventSink(tmp_path / "events"),
    )


class _StubRetriever:
    def __init__(self):
        self.queries = []

    async def retrieve(self, query, **kwargs):
        self.queries.append(query)
        block = f"[{ITEM.title} — arXiv:{ITEM.versioned_id}, p.{ITEM.page}, §{ITEM.section}]\n{ITEM.text}"
        from app.retrieval.retriever import RetrievalResult

        return RetrievalResult(
            query=query,
            items=[ITEM],
            context_text=block,
            total_est_tokens=ITEM.est_tokens,
            n_candidates_after_threshold=5,
            reranked=False,
            first_chunk_over_budget=False,
        )


async def test_runner_end_to_end_tokens_match_documented_tokenizer(tmp_path):
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        provider = StandInProvider(base_url="http://test", http=client)
        await client.post(
            "/_admin/shaping", json={"completion_text": CITED_ANSWER}
        )
        runner = _runner(tmp_path, provider)

        outcome = await runner.run(
            request_id="req-e2e",
            scenario="e2e",
            question=QUESTION,
            retrieval_setting="B",
            start_time=TS,
        )
        state = (await client.get("/_admin/state")).json()

    assert outcome.status == "ok"
    assert outcome.outcome_kind == "completed"  # unparseable understanding (D16) -> proceed
    assert outcome.citations_valid is True
    assert outcome.first_token_ms is not None

    # expected usage: two model calls (understanding + answer), both shaped
    # to CITED_ANSWER; prompts are the runner's exact prompt builders.
    u_messages = _messages(
        UNDERSTANDING_SYSTEM,
        understanding_user_prompt(QUESTION, _context()),
    )
    a_messages = _messages(
        ANSWER_SYSTEM,
        answer_user_prompt(QUESTION, _context()),
    )

    def prompt_tokens(msgs):
        return sum(
            count_tokens(m["role"]) + count_tokens(m["content"]) + PER_MESSAGE_OVERHEAD_TOKENS
            for m in msgs
        )

    expected_in = prompt_tokens(u_messages) + prompt_tokens(a_messages)
    expected_out = 2 * count_tokens(CITED_ANSWER)

    rows = list(csv.reader((tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()))[1:]
    assert len(rows) == 1
    row = rows[0]
    assert row[6] == "ok"
    assert row[7] == "1"  # understanding + answer = 2 attempts -> 1 retry
    assert row[8] == str(expected_in), "tokens_in must equal REPORTED usage summed"
    assert row[9] == str(expected_out), "tokens_out must equal REPORTED usage summed"
    assert row[10] == "B"
    assert row[3] == TS
    assert float(row[4]) > 0
    # service counters agree with the pipeline shape
    assert state["counters"]["requests_total"] == 2


def _context() -> str:
    return (
        f"[{ITEM.title} — arXiv:{ITEM.versioned_id}, p.{ITEM.page}, §{ITEM.section}]\n"
        f"{ITEM.text}"
    )


async def test_runner_maps_scripted_daily_rate_limit(tmp_path):
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        provider = StandInProvider(base_url="http://test", http=client)
        await client.post("/_admin/mode", json={"mode": "rate_limited_daily"})
        runner = _runner(tmp_path, provider)

        outcome = await runner.run(
            request_id="req-rl",
            scenario="e2e",
            question=QUESTION,
            retrieval_setting="B",
            start_time=TS,
        )
        events = [
            json.loads(line)
            for line in (tmp_path / "events" / "req-rl.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    assert outcome.status == "rate_limited"
    assert outcome.outcome_kind == "provider_rate_limit_daily"
    assert outcome.answer == ""  # no user-visible answer for a failed request
    assert outcome.first_token_ms is None
    rows = list(csv.reader((tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()))[1:]
    assert rows[0][6] == "rate_limited"
    assert rows[0][7] == "0"  # daily 429 is NEVER retried (D8)
    assert rows[0][4] == ""
    assert any(
        e["kind"] == "outcome_typed" and e["outcome_kind"] == "provider_rate_limit_daily"
        for e in events
    )
