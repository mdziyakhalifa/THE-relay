"""Unit tests for the API layer (WP8) — in-process ASGI, no sockets.

Covers: the outcome->HTTP mapping (200/429/504/502/503), validation that
returns 400/409 WITHOUT offering a request (no results.csv row), the SSE
event grammar of /ask/stream (status/token/reset/done), live regen reset
markers, mid-stream deadline handling, and /health introspection.

Real-server behavior (uvicorn disconnect detection -> user_cancelled row)
is covered by tests/integration/test_api_stream_disconnect.py.
"""
import asyncio
import csv
import json
from pathlib import Path

import httpx

from app.api.service import create_api_app
from app.core.settings import AppCapacity, BreakerPolicy, PipelineBudgets, RetryPolicy
from app.orchestrator.events import EventSink
from app.orchestrator.runner import RelayRunner
from app.providers import ChatChunk, ChatMessage, ChatResponse, ChatUsage, LLMProvider
from app.providers.standin import StandInProvider
from app.reliability.breaker import CircuitBreaker
from app.reliability.capacity import RequestGate
from app.reliability.errors import RateLimitDailyError, RateLimitMinuteError
from app.reliability.limiter import HeaderQuotaLimiter, ProviderLimitsSnapshot
from app.reliability.recorder import RequestRecorder
from app.retrieval.retriever import EvidenceItem, RetrievalResult
from app.standin.service import create_app as create_standin_app
from app.standin.tokenizer import count_tokens

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
VALID_CITED = "Scaled dot-product attention multiplies queries by keys [arXiv:1706.03762v7 p.4]."
UNCITED_ANSWER = "Scaled dot-product attention multiplies queries by keys."
QUESTION = "How does scaled dot-product attention work?"


# ------------------------------------------------------------------ fakes


class StubRetriever:
    def __init__(self, items: list[EvidenceItem]):
        self.items = items

    async def retrieve(self, query, **kwargs) -> RetrievalResult:
        blocks = [
            f"[{i.title} — arXiv:{i.versioned_id}, p.{i.page}, §{i.section}]\n{i.text}"
            for i in self.items
        ]
        return RetrievalResult(
            query=query,
            items=list(self.items),
            context_text="\n\n".join(blocks),
            total_est_tokens=sum(i.est_tokens for i in self.items),
            n_candidates_after_threshold=5,
            reranked=False,
            first_chunk_over_budget=False,
        )


class FakeScriptedProvider(LLMProvider):
    name = "stand-in"

    def __init__(self):
        self.complete_script: list = []
        self.stream_script: list = []

    async def complete(self, messages, *, model, temperature=0.0, max_tokens=None, timeout_s=None):
        item = self.complete_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def stream(self, messages, *, model, temperature=0.0, max_tokens=None):
        return self._gen()

    async def _gen(self):
        item = self.stream_script.pop(0)
        if isinstance(item, Exception):
            raise item
        async for chunk in item():
            yield chunk


def _resp(content: str, prompt: int = 10, comp: int = 5) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="test-model",
        finish_reason="stop",
        usage=ChatUsage(prompt_tokens=prompt, completion_tokens=comp, total_tokens=prompt + comp),
        limits=ProviderLimitsSnapshot(),
        transport_ms=1.0,
    )


def _sufficient(sufficient: bool = True, refined: str | None = None) -> ChatResponse:
    return _resp(
        json.dumps({"sufficient": sufficient, "refined_search_terms": refined, "reason": "x"})
    )


def _deltas(*pieces, usage=(20, 30)):
    """Stream-script item: content deltas, then finish + reported usage."""

    async def gen():
        for piece in pieces:
            yield ChatChunk(content_delta=piece)
        yield ChatChunk(
            content_delta="",
            finish_reason="stop",
            usage=ChatUsage(
                prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage)
            ),
        )

    return gen


def _stalled_deltas(stall_s: float, *pieces, usage=(20, 30)):
    """Stream-script item: deltas, then a stall long enough to blow the
    request deadline BEFORE the finish chunk arrives (partial stream)."""

    async def gen():
        for piece in pieces:
            yield ChatChunk(content_delta=piece)
        await asyncio.sleep(stall_s)
        yield ChatChunk(
            content_delta="",
            finish_reason="stop",
            usage=ChatUsage(
                prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage)
            ),
        )

    return gen


# ----------------------------------------------------------------- harness


def _build(
    tmp_path: Path,
    provider: LLMProvider,
    *,
    retriever: StubRetriever | None = None,
    gate: RequestGate | None = None,
    pipeline: PipelineBudgets | None = None,
):
    gate = gate or RequestGate(AppCapacity())
    breaker = CircuitBreaker(BreakerPolicy())
    limiter = HeaderQuotaLimiter()
    recorder = RequestRecorder(tmp_path / "results.csv")
    runner = RelayRunner(
        retrievers={"B": retriever or StubRetriever([ITEM])},
        provider=provider,
        model="test-model",
        gate=gate,
        breaker=breaker,
        limiter=limiter,
        pipeline=pipeline or PipelineBudgets(),
        retry_policy=RetryPolicy(),
        recorder=recorder,
        sink=EventSink(tmp_path / "events"),
    )
    app = create_api_app(
        runner=runner, gate=gate, breaker=breaker, limiter=limiter, recorder=recorder
    )
    return app, runner, recorder


def _client(app) -> httpx.AsyncClient:
    # base_url is REQUIRED on injected clients (recorded gotcha)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _rows(tmp_path: Path) -> list[list[str]]:
    path = tmp_path / "results.csv"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return list(csv.reader(lines))[1:]


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Named SSE events in order; keepalive comment lines are skipped."""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        kind = None
        data: dict | None = None
        for line in block.splitlines():
            if line.startswith("event: "):
                kind = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        assert kind is not None and data is not None, f"malformed SSE block: {block!r}"
        events.append((kind, data))
    return events


async def _ask(app, *, request_id: str = "api-1", **overrides) -> httpx.Response:
    body = {"question": QUESTION, "request_id": request_id}
    body.update(overrides)
    async with _client(app) as client:
        return await client.post("/ask", json=body)


# ------------------------------------------------------------- /ask basics


async def test_ask_happy_path_maps_ok_to_200_with_row(tmp_path):
    provider = FakeScriptedProvider()
    provider.complete_script = [_sufficient(sufficient=True)]
    provider.stream_script = [_deltas(VALID_CITED)]
    app, _, _ = _build(tmp_path, provider)

    response = await _ask(app)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["outcome_kind"] == "completed"
    assert body["answer"] == VALID_CITED
    assert body["citations"]["valid"] is True
    assert body["source"] == "stand-in"
    assert body["row_recorded"] is True
    assert body["first_token_ms"] > 0
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0][1] == "api"  # default scenario
    assert rows[0][6] == "ok"


async def test_ask_refusal_maps_to_200_refused(tmp_path):
    provider = FakeScriptedProvider()
    provider.complete_script = [_sufficient(sufficient=False, refined=None)]
    app, _, _ = _build(tmp_path, provider)

    response = await _ask(app)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["refused"] is True
    assert "cannot answer" in body["answer"]
    assert body["citations"] is None


async def test_validation_errors_return_400_without_offering(tmp_path):
    provider = FakeScriptedProvider()
    app, _, _ = _build(tmp_path, provider)

    cases = [
        # (json body, expected kind)
        ({"question": "   "}, "empty_question"),
        ({"retrieval_setting": "B"}, "empty_question"),
        ({"question": QUESTION, "retrieval_setting": "Z"}, "unknown_retrieval_setting"),
        ({"question": QUESTION, "request_id": "bad id!"}, "bad_request_id"),
        ({"question": QUESTION, "retrieval_setting": 3}, "unknown_retrieval_setting"),
    ]
    async with _client(app) as client:
        for body, expected_kind in cases:
            response = await client.post("/ask", json=body)
            assert response.status_code == 400, (body, response.text)
            assert response.json()["error"]["kind"] == expected_kind
        response = await client.post(
            "/ask", content=b"not json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["kind"] == "invalid_json"

    assert _rows(tmp_path) == [], "a validation failure must never produce a row"
    assert provider.complete_script == [] and provider.stream_script == []


async def test_unknown_retrieval_setting_lists_valid_options(tmp_path):
    provider = FakeScriptedProvider()
    app, _, _ = _build(tmp_path, provider)
    response = await _ask(app, request_id="api-opt", retrieval_setting="Z")
    assert "B" in response.json()["error"]["message"]


async def test_duplicate_request_id_returns_409_without_second_row(tmp_path):
    provider = FakeScriptedProvider()
    provider.complete_script = [_sufficient(sufficient=True)]
    provider.stream_script = [_deltas(VALID_CITED)]
    app, _, _ = _build(tmp_path, provider)

    first = await _ask(app, request_id="dup-1")
    assert first.status_code == 200
    second = await _ask(app, request_id="dup-1")
    assert second.status_code == 409
    assert second.json()["error"]["kind"] == "duplicate_request_id"
    assert len(_rows(tmp_path)) == 1


async def test_daily_rate_limit_maps_to_429_with_kind(tmp_path):
    standin_app = create_standin_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=standin_app), base_url="http://standin"
    ) as standin_client:
        await standin_client.post("/_admin/mode", json={"mode": "rate_limited_daily"})
        provider = StandInProvider(base_url="http://standin", http=standin_client)
        app, _, _ = _build(tmp_path, provider)

        response = await _ask(app, request_id="rl-1")

        assert response.status_code == 429
        body = response.json()
        assert body["status"] == "rate_limited"
        assert body["outcome_kind"] == "provider_rate_limit_daily"
        rows = _rows(tmp_path)
        assert rows[0][6] == "rate_limited"
        assert rows[0][7] == "0"  # daily 429 is never retried (D8)
        assert rows[0][4] == ""  # no token was ever delivered
        await provider.aclose()


async def test_queue_rejection_maps_to_503_with_not_applicable_row(tmp_path):
    provider = FakeScriptedProvider()
    gate = RequestGate(AppCapacity(in_flight_limit=1, queue_limit=0))
    app, _, _ = _build(tmp_path, provider, gate=gate)

    async with gate.slot():
        response = await _ask(app, request_id="q-1")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["outcome_kind"] == "capacity_rejected"
    rows = _rows(tmp_path)
    assert rows[0][6] == "error"
    assert rows[0][10] == "not_applicable"  # retrieval never ran (D13)


async def test_minute_rate_limit_retries_then_maps_to_429(tmp_path):
    provider = FakeScriptedProvider()
    provider.complete_script = [
        RateLimitMinuteError("429 tpm", retry_after_s=0.001) for _ in range(3)
    ]
    app, runner, _ = _build(tmp_path, provider)
    runner._sleeper = _instant_sleeper  # no real backoff in tests

    response = await _ask(app, request_id="rl-2")

    assert response.status_code == 429
    body = response.json()
    assert body["outcome_kind"] == "provider_rate_limit_minute"
    assert body["retry_after_s"] == 0.001
    rows = _rows(tmp_path)
    assert rows[0][7] == "2"  # 3 attempts -> 2 retries


async def _instant_sleeper(seconds: float) -> None:
    return None


async def test_health_reports_breaker_limiter_gate(tmp_path):
    provider = FakeScriptedProvider()
    app, _, _ = _build(tmp_path, provider)
    async with _client(app) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["breaker"]["state"] == "closed"
    assert body["limiter"]["observed"] is False  # observe-only until headers
    assert body["gate"]["in_flight"] == 0


# ------------------------------------------------------------ /ask/stream


async def test_stream_happy_sequence_ends_with_done(tmp_path):
    provider = FakeScriptedProvider()
    provider.complete_script = [_sufficient(sufficient=True)]
    provider.stream_script = [
        _deltas(
            "Scaled ",
            "dot-product ",
            "attention ",
            "multiplies ",
            "queries ",
            "by keys ",
            "[arXiv:1706.03762v7 p.4].",
        )
    ]
    app, _, _ = _build(tmp_path, provider)

    async with _client(app) as client:
        response = await client.post(
            "/ask/stream", json={"question": QUESTION, "request_id": "s-1"}
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "status"
    assert events[0][1]["stage"] == "admitted"
    assert kinds[-1] == "done"
    stages = [data.get("stage") for kind, data in events if kind == "status"]
    assert stages == ["admitted", "retrieved", "understanding", "answer"]
    tokens = [data["delta"] for kind, data in events if kind == "token"]
    assert "".join(tokens) == VALID_CITED
    assert all(data["request_id"] == "s-1" for _, data in events)

    done = events[-1][1]
    assert done["status"] == "ok"
    assert done["answer"] == VALID_CITED
    assert done["citations"]["valid"] is True
    assert done["tokens_dropped"] == 0
    assert done["kind"] == "done"
    rows = _rows(tmp_path)
    assert rows[0][6] == "ok"
    assert rows[0][0] == "s-1"


async def test_stream_regen_emits_reset_then_valid_tokens(tmp_path):
    provider = FakeScriptedProvider()
    provider.complete_script = [_sufficient(sufficient=True)]
    provider.stream_script = [_deltas(UNCITED_ANSWER), _deltas(VALID_CITED)]
    app, _, _ = _build(tmp_path, provider)

    async with _client(app) as client:
        response = await client.post(
            "/ask/stream", json={"question": QUESTION, "request_id": "s-2"}
        )

    events = _parse_sse(response.text)
    resets = [data for kind, data in events if kind == "reset"]
    assert resets and resets[0]["reason"] == "regen"
    # tokens split around the reset marker: first answer, then the regen
    token_groups: list[str] = []
    current = ""
    for kind, data in events:
        if kind == "token":
            current += data["delta"]
        elif kind == "reset" and current:
            token_groups.append(current)
            current = ""
    if current:
        token_groups.append(current)
    assert token_groups == [UNCITED_ANSWER, VALID_CITED]
    done = events[-1][1]
    assert done["status"] == "ok"
    assert done["answer"] == VALID_CITED  # the DELIVERED answer
    assert done["citations"]["valid"] is True
    rows = _rows(tmp_path)
    assert rows[0][7] == "2"  # understanding + answer + regen = 3 attempts


async def test_stream_mid_stream_deadline_yields_timeout_done(tmp_path):
    provider = FakeScriptedProvider()
    provider.complete_script = [_sufficient(sufficient=True)]
    provider.stream_script = [_stalled_deltas(1.0, "partial ")]
    app, _, _ = _build(
        tmp_path, provider, pipeline=PipelineBudgets(request_deadline_ms=200.0)
    )

    async with _client(app) as client:
        response = await client.post(
            "/ask/stream", json={"question": QUESTION, "request_id": "s-3"}
        )

    events = _parse_sse(response.text)
    resets = [data for kind, data in events if kind == "reset"]
    assert any(r["reason"] == "partial_discarded" for r in resets)
    done = events[-1][1]
    assert done["status"] == "timeout"
    assert done["outcome_kind"] == "deadline_exceeded"
    assert done["answer"] == ""  # nothing user-visible was delivered
    rows = _rows(tmp_path)
    assert rows[0][6] == "timeout"
    assert rows[0][4] == ""  # TTFT stays empty: partial tokens were discarded


async def test_stream_daily_rate_limit_done_event_carries_status(tmp_path):
    standin_app = create_standin_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=standin_app), base_url="http://standin"
    ) as standin_client:
        await standin_client.post("/_admin/mode", json={"mode": "rate_limited_daily"})
        provider = StandInProvider(base_url="http://standin", http=standin_client)
        app, _, _ = _build(tmp_path, provider)
        async with _client(app) as client:
            response = await client.post(
                "/ask/stream", json={"question": QUESTION, "request_id": "s-4"}
            )
        await provider.aclose()

    # SSE convention: transport is 200; the terminal status rides in `done`
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert not [k for k, _ in events if k == "token"]
    done = events[-1][1]
    assert done["status"] == "rate_limited"
    assert done["outcome_kind"] == "provider_rate_limit_daily"
    assert done["answer"] == ""
    rows = _rows(tmp_path)
    assert rows[0][6] == "rate_limited"


async def test_stream_validation_returns_400_json_not_sse(tmp_path):
    provider = FakeScriptedProvider()
    app, _, _ = _build(tmp_path, provider)
    async with _client(app) as client:
        response = await client.post("/ask/stream", json={"question": ""})
    assert response.status_code == 400
    assert response.json()["error"]["kind"] == "empty_question"
    assert _rows(tmp_path) == []


async def test_stream_tokens_match_reported_usage_honestly(tmp_path):
    """The CSV tokens come from provider-REPORTED usage only; the SSE token
    events are the same content the provider streamed — nothing invented."""
    provider = FakeScriptedProvider()
    provider.complete_script = [_sufficient(sufficient=True)]
    provider.stream_script = [_deltas(VALID_CITED)]
    app, _, _ = _build(tmp_path, provider)

    async with _client(app) as client:
        await client.post("/ask/stream", json={"question": QUESTION, "request_id": "s-5"})

    rows = _rows(tmp_path)
    # understanding(10/5) + answer(20/30) reported usage
    assert rows[0][8] == "30" and rows[0][9] == "35"
