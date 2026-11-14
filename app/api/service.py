"""HTTP API layer (WP8) — FastAPI endpoints over ONE RelayRunner.

Decision D17 (with WP7's D16 runner semantics unchanged underneath):

- `POST /ask` — buffered. Maps the runner's TYPED outcome to HTTP:
    ok (incl. honest refusal)  -> 200
    rate_limited               -> 429 (+ retry_after_s when the provider sent one)
    timeout                    -> 504
    error: capacity_rejected   -> 503 (OUR backpressure, not provider sickness)
    error: everything else     -> 502
  The CSV row semantics are UNCHANGED by this mapping (D13): the row keeps
  status error / outcome_kind capacity_rejected; only the HTTP surface
  distinguishes them.
- `POST /ask/stream` — SSE. The runner forwards live events through its
  per-run `token_sink` into a bounded queue; the endpoint pumps them as
  named SSE events (status / token / reset) and terminates with one
  `done` event carrying the FINAL outcome (full answer text included),
  so a client that dropped tokens or reset its buffer can always
  reconcile. The HTTP transport for SSE is 200 even when the answer
  fails — the terminal status rides in `done` (SSE convention).
- Validation happens BEFORE offering: invalid JSON body, missing/empty
  question, unknown retrieval_setting, malformed request_id -> 400;
  duplicate request_id -> 409. These responses produce NO results.csv row
  — the request was never offered. (The runner keeps its own honest-row
  safety net for direct callers such as the load harness.)
- `start_time` is captured at HTTP entry, so first_token_ms measured by
  the runner (from offer moment) includes queue wait end-to-end and the
  load harness's TTFT boundary matches the internal measurement (D7).
- Client disconnect mid-stream: the response generator is closed
  (Starlette/uvicorn spec 2.3 collapsing task group), its finally cancels
  the runner task, and the runner records its honest `user_cancelled` row.
- `/health` — introspection for the load harness: breaker state, limiter
  projection summary, gate counters.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import AsyncIterator, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.logging_setup import get_logger
from app.orchestrator.runner import RelayOutcome, RelayRunner
from app.reliability.recorder import RequestRecorder, utc_now_iso_ms

log = get_logger("relay.api")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# Live-stream pump tuning (D17): the queue is bounded so a slow client can
# never couple back into the runner; overflow DROPS token events (counted,
# reconciled by the final done event, which always carries the full answer).
_SSE_QUEUE_MAX = 1024
_SSE_POLL_S = 1.0  # heartbeat cadence = disconnect-detection latency bound


class _ApiError(Exception):
    """Validation failure mapped to an HTTP status WITHOUT offering."""

    def __init__(self, status_code: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind
        self.message = message


def create_api_app(
    *,
    runner: RelayRunner,
    gate,
    breaker,
    limiter,
    recorder: RequestRecorder | None = None,
) -> FastAPI:
    app = FastAPI(title="the-relay API", version="0.1.0")

    # CORS for local frontend development; does not affect existing API behavior.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Read-only observability endpoints (frontend integration).
    try:
        from app.api.observability import create_observability_router

        app.include_router(create_observability_router())
    except Exception:
        # Non-fatal: core /ask /ask/stream /health remain available.
        log.exception("observability_router_unavailable")

    # ------------------------------------------------------------- helpers

    async def _parse_and_validate(request: Request) -> tuple[str, str, str, str]:
        """Shared /ask + /ask/stream admission gate. Returns (request_id,
        scenario, question, retrieval_setting) or raises _ApiError."""
        try:
            body = await request.json()
        except Exception:
            raise _ApiError(400, "invalid_json", "request body must be valid JSON")
        if not isinstance(body, dict):
            raise _ApiError(400, "invalid_body", "request body must be a JSON object")
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            raise _ApiError(
                400, "empty_question", "'question' must be a non-empty string"
            )
        retrieval_setting = body.get("retrieval_setting", "B")
        if (
            not isinstance(retrieval_setting, str)
            or retrieval_setting not in runner.retrieval_settings
        ):
            raise _ApiError(
                400,
                "unknown_retrieval_setting",
                f"retrieval_setting must be one of {sorted(runner.retrieval_settings)}",
            )
        scenario = body.get("scenario", "api")
        if not isinstance(scenario, str) or not scenario.strip():
            raise _ApiError(
                400, "bad_scenario", "'scenario' must be a non-empty string"
            )
        request_id = body.get("request_id")
        if request_id is None:
            request_id = f"api-{uuid.uuid4().hex[:12]}"
        elif not isinstance(request_id, str) or not _REQUEST_ID_RE.match(request_id):
            raise _ApiError(
                400,
                "bad_request_id",
                "request_id must match [A-Za-z0-9_.-]{1,64} when supplied",
            )
        if recorder is not None and recorder.has_request(request_id):
            raise _ApiError(
                409,
                "duplicate_request_id",
                f"request_id {request_id!r} already has a results.csv row",
            )
        return request_id, scenario.strip(), question.strip(), retrieval_setting

    def _citations_view(outcome: RelayOutcome) -> dict | None:
        if outcome.citations is None:
            return None
        return {
            "valid": outcome.citations.valid,
            "uncited": outcome.citations.uncited,
            "n_cited": len(outcome.citations.cited),
            "n_unknown": len(outcome.citations.unknown),
        }

    def _body(outcome: RelayOutcome) -> dict:
        return {
            "request_id": outcome.request_id,
            "scenario": outcome.scenario,
            "source": outcome.source,
            "retrieval_setting": outcome.retrieval_setting,
            "status": outcome.status,
            "outcome_kind": outcome.outcome_kind,
            "answer": outcome.answer if outcome.status == "ok" else "",
            "refused": outcome.refused,
            "refusal_reason": outcome.refusal_reason,
            "citations": _citations_view(outcome),
            "first_token_ms": outcome.first_token_ms,
            "total_ms": outcome.total_ms,
            "retries": outcome.retries,
            "tokens_in": outcome.tokens_in,
            "tokens_out": outcome.tokens_out,
            "row_recorded": True,
        }

    def _http_status(outcome: RelayOutcome) -> int:
        if outcome.status == "ok":
            return 200
        if outcome.status == "rate_limited":
            return 429
        if outcome.status == "timeout":
            return 504
        # error: OUR capacity backpressure is 503; provider-health and other
        # faults surface as 502 (bad gateway)
        return 503 if outcome.outcome_kind == "capacity_rejected" else 502

    def _sse(kind: str, payload: dict) -> str:
        return (
            f"event: {kind}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        )

    # ------------------------------------------------------------ handlers

    @app.exception_handler(_ApiError)
    async def _api_error_handler(request: Request, error: _ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {"kind": error.kind, "message": error.message},
            },
        )

    @app.post("/ask")
    async def ask(request: Request) -> JSONResponse:
        request_id, scenario, question, retrieval_setting = (
            await _parse_and_validate(request)
        )
        start_time = utc_now_iso_ms()  # HTTP entry: TTFT covers queue wait (D7)
        try:
            outcome = await runner.run(
                request_id=request_id,
                scenario=scenario,
                question=question,
                retrieval_setting=retrieval_setting,
                start_time=start_time,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # the runner already recorded the row
            log.exception(
                "api_unexpected_error", extra={"request_id": request_id}
            )
            return JSONResponse(
                status_code=500,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "outcome_kind": f"unexpected_{type(error).__name__}",
                    "row_recorded": True,
                },
            )
        response_body = _body(outcome)
        if outcome.status == "rate_limited":
            retry_after = getattr(outcome.outcome, "retry_after_s", None)
            if retry_after is not None:
                response_body["retry_after_s"] = retry_after
        return JSONResponse(status_code=_http_status(outcome), content=response_body)

    @app.post("/ask/stream")
    async def ask_stream(request: Request):
        request_id, scenario, question, retrieval_setting = (
            await _parse_and_validate(request)
        )
        start_time = utc_now_iso_ms()

        queue: asyncio.Queue = asyncio.Queue(maxsize=_SSE_QUEUE_MAX)
        dropped = {"tokens": 0}

        async def sink(event: dict) -> None:
            try:
                queue.put_nowait({"request_id": request_id, **event})
            except asyncio.QueueFull:
                # live-only loss; the final done event reconciles (D17)
                dropped["tokens"] += 1

        run_task = asyncio.create_task(
            runner.run(
                request_id=request_id,
                scenario=scenario,
                question=question,
                retrieval_setting=retrieval_setting,
                start_time=start_time,
                token_sink=sink,
            )
        )

        async def event_stream() -> AsyncIterator[str]:
            try:
                while True:
                    if run_task.done() and queue.empty():
                        break
                    if await request.is_disconnected():
                        # non-blocking poll (pre-cancelled scope); on Windows
                        # proactor uvicorn it also resumes the paused read,
                        # which is what actually drives EOF detection
                        break  # finally cancels the run -> honest row
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=_SSE_POLL_S
                        )
                    except (asyncio.TimeoutError, TimeoutError):
                        # keepalive comment: bounds disconnect-detection
                        # latency and keeps idle links from being reaped
                        yield ": keepalive\n\n"
                        continue
                    yield _sse(str(event.get("kind")), event)
                try:
                    outcome = run_task.result()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # runner re-raises only unexpected programmer bugs (the
                    # row is already recorded); surface it inside the stream
                    yield _sse(
                        "done",
                        {
                            "kind": "done",
                            "request_id": request_id,
                            "status": "error",
                            "outcome_kind": f"unexpected_{type(error).__name__}",
                            "answer": "",
                            "tokens_dropped": dropped["tokens"],
                        },
                    )
                else:
                    done = _body(outcome)
                    done["kind"] = "done"
                    done["tokens_dropped"] = dropped["tokens"]
                    yield _sse("done", done)
            finally:
                if not run_task.done():
                    # client disconnect closes this generator mid-pump:
                    # cancel the run so the runner records its honest
                    # user_cancelled row (D16), then swallow the re-raise
                    run_task.cancel()
                    try:
                        await run_task
                    except asyncio.CancelledError:
                        pass  # the expected honest outcome of this cleanup
                    except Exception:
                        log.exception(
                            "stream_cleanup_error",
                            extra={"request_id": request_id},
                        )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "breaker": {"state": str(breaker.state)},
            "limiter": limiter.summary(),
            "gate": {
                "in_flight": gate.in_flight,
                "waiting": gate.waiting,
                "in_flight_limit": gate.in_flight_limit,
                "queue_limit": gate.queue_limit,
            },
        }

    return app
