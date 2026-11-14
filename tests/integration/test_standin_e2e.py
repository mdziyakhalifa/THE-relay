"""End-to-end test: the stand-in service under a REAL uvicorn server.

Boots the actual app (ephemeral port, background thread) and drives it
with the real StandInProvider over real HTTP — proving the full stack
(uvicorn + FastAPI + SSE + shared compat transport) works together, and
that admin scripting reaches the live server.
"""
import asyncio
import threading

import httpx
import pytest
import uvicorn

from app.providers import ChatMessage
from app.providers.standin import StandInProvider
from app.reliability.errors import RateLimitDailyError
from app.standin.service import create_app
from app.standin.tokenizer import count_tokens

MESSAGES = [ChatMessage(role="user", content="hello")]
EXPECTED_PROMPT_TOKENS = count_tokens("user") + count_tokens("hello") + 4


@pytest.fixture
async def live_server():
    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(500):  # up to ~10 s for startup
        if server.started:
            break
        await asyncio.sleep(0.02)
    else:
        raise RuntimeError("stand-in uvicorn server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


async def test_health_and_complete_round_trip_over_real_http(live_server):
    async with httpx.AsyncClient() as http:
        health = await http.get(f"{live_server}/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "stand-in"}

    provider = StandInProvider(base_url=live_server)
    try:
        response = await provider.complete(MESSAGES, model="standin-sim")
        assert response.content
        assert response.model == "standin-sim"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == EXPECTED_PROMPT_TOKENS
        assert response.usage.completion_tokens == 64
        assert response.transport_ms >= 0.0
    finally:
        await provider.aclose()


async def test_stream_round_trip_over_real_http(live_server):
    provider = StandInProvider(base_url=live_server)
    try:
        chunks = [
            chunk async for chunk in provider.stream(
                MESSAGES,
                model="standin-sim",
            )
        ]
    finally:
        await provider.aclose()
    content = "".join(chunk.content_delta for chunk in chunks)
    assert len(content) > 0
    # D14: the shared transport ALWAYS requests include_usage on streams,
    # so the service reports the final usage chunk — and it must match the
    # documented tokenizer exactly.
    usage = chunks[-1].usage
    assert usage is not None
    assert usage.prompt_tokens == EXPECTED_PROMPT_TOKENS
    assert usage.completion_tokens == 64
    assert usage.total_tokens == EXPECTED_PROMPT_TOKENS + 64


async def test_stream_usage_reported_when_requested(live_server):
    # include_usage is a REQUEST-level concern the shared transport always
    # sends on streams; the service honors it with the final usage chunk.
    import json

    import httpx

    body = {
        "model": "standin-sim",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with httpx.AsyncClient() as http:
        response = await http.post(f"{live_server}/chat/completions", json=body)
    assert response.status_code == 200
    events = [
        line[len("data: "):]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    usage_event = json.loads(events[-2])
    assert usage_event["choices"] == []
    assert usage_event["usage"]["prompt_tokens"] == EXPECTED_PROMPT_TOKENS


async def test_scripted_daily_rate_limit_over_real_http(live_server):
    async with httpx.AsyncClient() as http:
        await http.post(
            f"{live_server}/_admin/mode", json={"mode": "rate_limited_daily"}
        )
    provider = StandInProvider(base_url=live_server)
    try:
        with pytest.raises(RateLimitDailyError) as excinfo:
            await provider.complete(MESSAGES, model="standin-sim")
        assert excinfo.value.retry_after_s == 3600.0
    finally:
        await provider.aclose()
