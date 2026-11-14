"""Integration: client disconnects mid-stream on a REAL uvicorn server.

Proves the D17 disconnect path end to end over real TCP: the client reads
the first live token event, then closes the connection while the answer is
still streaming; uvicorn's ASGI disconnect listener collapses the response
task group, the SSE generator's cleanup cancels the runner task, and the
runner records its honest `user_cancelled` row (one row per offered
request, D13 — even when the consumer vanished).

The provider talks to a real stand-in uvicorn server (second process
thread) so the whole wire is real; nothing is mocked.
"""
import asyncio
import csv
import json
import threading
import time

import httpx
import pytest
import uvicorn

from app.api.service import create_api_app
from app.core.settings import AppCapacity, BreakerPolicy, PipelineBudgets, RetryPolicy
from app.orchestrator.events import EventSink
from app.orchestrator.runner import RelayRunner
from app.providers.standin import StandInProvider
from app.reliability.breaker import CircuitBreaker
from app.reliability.capacity import RequestGate
from app.reliability.limiter import HeaderQuotaLimiter
from app.reliability.recorder import RequestRecorder
from app.standin.service import create_app as create_standin_app

REQUEST_ID = "disconnect-test"
CITED = "Scaled dot-product attention multiplies queries by keys [arXiv:1706.03762v7 p.4]."
# pacing keeps the stream open for many seconds after the first token, so
# the disconnect always wins the race against stream completion
LONG_ANSWER = " ".join([CITED] * 40)


def _start_server(app) -> tuple[uvicorn.Server, threading.Thread, str]:
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(500):
        if server.started:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("uvicorn server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture
async def live_stack(tmp_path):
    gate = RequestGate(AppCapacity())
    breaker = CircuitBreaker(BreakerPolicy())
    limiter = HeaderQuotaLimiter()
    recorder = RequestRecorder(tmp_path / "results.csv")
    sink = EventSink(tmp_path / "events")

    standin_app = create_standin_app()
    s_server, s_thread, standin_url = _start_server(standin_app)

    # the provider is created in the test loop; httpx binds connections
    # lazily in whichever loop issues the requests (the API server loop)
    provider = StandInProvider(base_url=standin_url)
    runner = RelayRunner(
        retrievers={"B": _StubRetriever()},
        provider=provider,
        model="standin-sim",
        gate=gate,
        breaker=breaker,
        limiter=limiter,
        pipeline=PipelineBudgets(),
        retry_policy=RetryPolicy(),
        recorder=recorder,
        sink=sink,
    )
    api_app = create_api_app(
        runner=runner, gate=gate, breaker=breaker, limiter=limiter, recorder=recorder
    )
    a_server, a_thread, api_url = _start_server(api_app)

    try:
        yield api_url, standin_url, tmp_path
    finally:
        a_server.should_exit = True
        s_server.should_exit = True
        a_thread.join(timeout=10)
        s_thread.join(timeout=10)


class _StubRetriever:
    async def retrieve(self, query, **kwargs):
        from app.retrieval.retriever import EvidenceItem, RetrievalResult

        item = EvidenceItem(
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
        block = (
            f"[{item.title} — arXiv:{item.versioned_id}, p.{item.page}, §{item.section}]\n"
            f"{item.text}"
        )
        return RetrievalResult(
            query=query,
            items=[item],
            context_text=block,
            total_est_tokens=item.est_tokens,
            n_candidates_after_threshold=5,
            reranked=False,
            first_chunk_over_budget=False,
        )


async def test_client_disconnect_records_user_cancelled_row(live_stack, tmp_path):
    api_url, standin_url, results = live_stack

    async with httpx.AsyncClient() as http:
        response = await http.post(
            f"{standin_url}/_admin/shaping",
            json={
                "stream_chunk_delay_ms": 30,
                "completion_text": LONG_ANSWER,
            },
        )
    assert response.status_code == 200

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        async with client.stream(
            "POST",
            f"{api_url}/ask/stream",
            json={"question": "How does attention work?", "request_id": REQUEST_ID},
        ) as response:
            assert response.status_code == 200
            buffer = ""
            saw_token = False
            async for chunk in response.aiter_text():
                buffer += chunk
                if "event: token" in buffer:
                    saw_token = True
                    break  # context exit closes the connection MID-STREAM
            assert saw_token, f"no live token before disconnect; got {buffer[:300]!r}"
    # connection is now closed while the answer was still streaming

    # poll for the honest user_cancelled outcome (within 15 s); the row and
    # the event may land a moment apart, so require BOTH before passing
    events_file = results / "events" / f"{REQUEST_ID}.jsonl"
    csv_file = results / "results.csv"
    outcome_kinds: list[str] = []
    for _ in range(75):
        kinds_now: list[str] = []
        if events_file.is_file():
            for line in events_file.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("kind") == "outcome_typed":
                    kinds_now.append(event.get("outcome_kind"))
        rows_now: list[list[str]] = []
        if csv_file.is_file():
            rows_now = list(csv.reader(csv_file.read_text(encoding="utf-8").splitlines()))[1:]
        if "user_cancelled" in kinds_now and len(rows_now) == 1:
            outcome_kinds = kinds_now
            break
        outcome_kinds = kinds_now
        await asyncio.sleep(0.2)
    else:
        pytest.fail(
            f"user_cancelled never fully recorded; outcome kinds seen: "
            f"{outcome_kinds}, rows: {rows_now}"
        )

    rows = list(csv.reader((results / "results.csv").read_text(encoding="utf-8").splitlines()))[1:]
    assert len(rows) == 1, "exactly ONE row for the offered request (D13)"
    assert rows[0][0] == REQUEST_ID
    assert rows[0][6] == "error"
    assert rows[0][10] == "B"  # retrieval had run before the disconnect
