"""The stand-in service — our local OpenAI-compatible LLM simulator (WP6).

This is a SIMULATOR, not a proxy: every behavior it exhibits is scripted
and deterministic. Honesty rule (mirrors D13/D14): numbers measured while
talking to this service describe THIS SERVICE and are never presented as
Groq measurements; results.csv rows carry `source=stand-in` for exactly
this reason.

Wire contract (fulfills the one declared in `app/providers/standin.py`):

- `POST /chat/completions` — OpenAI-compatible, non-streaming AND SSE
  streaming (`stream: true`). Accepts `model` (required, echoed back),
  `messages` ({role, content} list), `temperature` (accepted, ignored —
  documented), `max_tokens` (honored as an EXACT completion length target;
  finish_reason is `length` when max_tokens set, `stop` otherwise),
  `stream_options.include_usage` (usage chunk is emitted ONLY when this is
  requested, matching OpenAI's documented behavior).
- `usage` is computed with the DOCUMENTED tokenizer (`app/standin/
  tokenizer.py`) and reported in the non-streaming body and, when
  include_usage was requested, on the stream's final chunk (OpenAI shape:
  `choices: []` + `usage`). The service NEVER omits usage when
  include_usage was sent — a service guarantee, stronger than Groq's
  (whose honoring is only to be confirmed at WP9).
- No authentication (localhost). No rate-limit headers in normal `ok`
  operation — the absent-headers path (limiter observe-only, empty
  snapshot) is the honest default. Rate-limit headers ARE emitted when a
  rate-limit failure mode is scripted, so WP10 can exercise the D8
  daily-vs-minute classification end-to-end without real Groq traffic.
- `GET /health` — liveness for the load harness readiness probe.

Failure-mode scripting (the WP6 deliverable) via the admin API:

- `POST /_admin/mode` — replaces the MODE section (mode, times,
  retry_after_s, server_error_status, hang_seconds). `times=N` applies
  the mode to the next N chat requests, then auto-reverts to `ok`;
  `times=null` (default) is sticky until changed or reset.
- `POST /_admin/shaping` — sets the sticky SHAPING section (delay_ms,
  first_token_delay_ms, completion_tokens) used by ok-mode responses;
  survives mode changes, cleared only by `/_admin/reset`.
- `GET /_admin/state` — current sections + counters (for load-run logs).
- `POST /_admin/reset` — everything back to defaults.

Modes and the client-side typed outcome they must produce (tested):

- ok                  → 200 (+ usage)
- rate_limited_daily  → 429, x-ratelimit-remaining-requests=0, retry-after
                        → RateLimitDailyError (never retried, D8)
- rate_limited_minute → 429, x-ratelimit-remaining-requests>0, retry-after
                        → RateLimitMinuteError
- server_error        → 5xx (default 503) → ProviderServerError
- malformed           → 200 with a non-JSON body → MalformedRequestError
- hang                → accepts the connection, then sleeps hang_seconds
                        before answering → client-side timeout
                        → TransientNetworkError (real-wire tested)
- abrupt_close        → response BODY starts then ends early: for streams
                        the SSE stream ends without [DONE] and without
                        finish_reason (silent truncation); for non-stream
                        requests the JSON body is truncated. Real mid-
                        stream TCP truncation is additionally covered by
                        a raw-socket test (incomplete chunked encoding).

Counters are exposed via /_admin/state for load-run evidence. The service
imposes no capacity limits of its own — capacity/queue behavior is the
CLIENT-side pipeline under test (WP4 primitives), not the simulator's.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import cycle, islice
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.standin.tokenizer import (
    DEFAULT_COMPLETION_TOKENS,
    TOKEN_RE,
    count_tokens,
    encode,
    prompt_tokens,
)

# Words without punctuation, so " ".join(words) tokenizes to exactly
# len(words) tokens under the documented tokenizer.
WORDLIST: tuple[str, ...] = (
    "relay", "token", "stream", "retry", "breaker", "queue", "budget",
    "window", "request", "provider", "network", "deadline", "signal",
    "health", "probe", "latency",
)

# Deterministic filler words for exact-token simulated responses (tokenizer: \w+ tokens).
_FILLER_WORDS: tuple[str, ...] = (
    "simulated", "standin", "evidence", "pinned", "paper", "context",
    "retrieval", "answer", "source", "grounded", "deterministic", "local",
)

_OK_BODY_PREFIX = '{"choices":[{"mess'  # truncated-JSON body for non-stream abrupt_close


class FailureMode(StrEnum):
    OK = "ok"
    RATE_LIMITED_DAILY = "rate_limited_daily"
    RATE_LIMITED_MINUTE = "rate_limited_minute"
    SERVER_ERROR = "server_error"
    MALFORMED = "malformed"
    HANG = "hang"
    ABRUPT_CLOSE = "abrupt_close"


@dataclass
class ModeSection:
    """Replaced wholesale by every POST /_admin/mode call."""

    mode: FailureMode = FailureMode.OK
    times: int | None = None  # None = sticky until changed/reset
    remaining: int | None = None  # live countdown for times
    retry_after_s: float = 3600.0
    server_error_status: int = 503
    hang_seconds: float = 30.0


@dataclass
class ShapingSection:
    """Sticky ok-mode response shaping; cleared only by /_admin/reset."""

    delay_ms: float = 0.0
    first_token_delay_ms: float = 0.0
    completion_tokens: int | None = None
    completion_text: str | None = None
    stream_chunk_delay_ms: float = 0.0


@dataclass
class Counters:
    requests_total: int = 0
    served_by_mode: dict[str, int] = field(default_factory=dict)


class ModeSpec(BaseModel):
    mode: FailureMode
    times: int | None = Field(default=None, ge=0)
    retry_after_s: float | None = Field(default=None, ge=0)
    server_error_status: int = Field(default=503, ge=500, le=599)
    hang_seconds: float = Field(default=30.0, ge=0)


class ShapingSpec(BaseModel):
    delay_ms: float = Field(default=0.0, ge=0)
    first_token_delay_ms: float = Field(default=0.0, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    completion_text: str | None = Field(default=None)
    """Exact-text override (evaluation aid, WP6/WP7): when set, the service
    returns THIS text verbatim instead of generated word-cycle text;
    completion_tokens is still reported via the documented tokenizer."""
    stream_chunk_delay_ms: float = Field(default=0.0, ge=0)
    """Per-chunk stream pacing (D15 extension, WP8): spaces consecutive
    stream chunks apart, after the first. Delivery pacing only — the
    reported usage and the exact text are unchanged."""


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _stream_pieces(text: str) -> list[str]:
    """Split text into stream chunks that concatenate back to EXACTLY the
    original (fidelity contract): each piece is one tokenizer token plus
    the whitespace that followed it. Word-cycle text chunks the same way
    as before; punctuation keeps its original adjacency ("p.4" never
    becomes "p . 4")."""
    matches = list(TOKEN_RE.finditer(text))
    pieces: list[str] = []
    for index, match in enumerate(matches):
        end = match.end()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pieces.append(text[match.start():next_start])
    return pieces


def _completion_text(target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    return " ".join(islice(cycle(WORDLIST), target_tokens))


def _extract_evidence_refs(messages: list[dict]) -> list[tuple[str, int, str]]:
    """Extract unique (versioned_id, page, title) refs from evidence blocks in messages.

    Only uses identifiers actually present in the supplied prompt/evidence text.
    Never fabricates paper IDs or page numbers.
    """
    refs: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    # Evidence blocks assembled by retriever look like:
    # [1] [Title — arXiv:<id>, p.<n>, §<section>]
    block_re = re.compile(
        r"\[\d+\]\s*\[(?P<title>[^\]]*?)\s*[\u2014\-]\s*arXiv:(?P<id>\d{4}\.\d{4,5}v\d+),\s*p\.(?P<page>\d+)",
        re.IGNORECASE,
    )
    for msg in messages or []:
        content = str(msg.get("content") or "")
        for m in block_re.finditer(content):
            vid = m.group("id")
            page = int(m.group("page"))
            key = (vid, page)
            if key in seen:
                continue
            seen.add(key)
            title = (m.group("title") or "").strip()
            refs.append((vid, page, title))
        # Fallback: any explicit citation-like provenance in text (still from provided evidence)
        if not refs:
            for m in re.finditer(r"arXiv:(?P<id>\d{4}\.\d{4,5}v\d+)[^\n\]]*?p\.\s*(?P<page>\d+)", content, re.IGNORECASE):
                vid = m.group("id")
                page = int(m.group("page"))
                key = (vid, page)
                if key in seen:
                    continue
                seen.add(key)
                refs.append((vid, page, "pinned paper"))
    return refs


def _is_understanding_request(messages: list[dict]) -> bool:
    for msg in messages or []:
        content = str(msg.get("content") or "")
        if "Reply with STRICT JSON only" in content or "sufficient" in content and "refined_search_terms" in content:
            return True
        if msg.get("role") == "system" and "understanding stage" in content.lower():
            return True
    return False


def _fit_to_token_count(text: str, target_tokens: int) -> str:
    """Adjust text to have exactly target_tokens under the documented tokenizer.

    Pads by appending deterministic filler words (preserving the original
    readable text and citation tags). Truncates from the end only if needed.
    """
    if target_tokens <= 0:
        return ""
    current = count_tokens(text)
    if current == target_tokens:
        return text
    if current < target_tokens:
        pad = " ".join(islice(cycle(_FILLER_WORDS), target_tokens - current))
        return (text + " " + pad).strip()
    # Truncate (rare): keep prefix tokens and rebuild; best-effort readability.
    tokens = encode(text)[:target_tokens]
    return " ".join(tokens)


def _json_with_exact_tokens(payload: dict, target_tokens: int) -> str:
    """Produce valid JSON with exactly target_tokens (documented tokenizer)."""
    base = json.dumps(payload, separators=(", ", ": "))
    if target_tokens <= 0:
        return ""
    if count_tokens(base) == target_tokens:
        return base
    # Adjust by padding/trimming the reason string with word tokens.
    reason = str(payload.get("reason", "simulated"))
    reason_tokens = encode(reason)
    delta = target_tokens - count_tokens(base)
    if delta > 0:
        reason_tokens = reason_tokens + list(islice(cycle(_FILLER_WORDS), delta))
    elif delta < 0:
        reason_tokens = reason_tokens[: max(1, len(reason_tokens) + delta)]
    payload = dict(payload)
    payload["reason"] = " ".join(reason_tokens) if reason_tokens else "simulated"
    text = json.dumps(payload, separators=(", ", ": "))
    # Final guard: enforce exact count while keeping parseable JSON best-effort.
    if count_tokens(text) != target_tokens:
        # fall back to token-joined form only as last resort
        return _fit_to_token_count(text, target_tokens)
    return text


def _generate_simulated_answer(messages: list[dict], target_tokens: int | None = None) -> str:
    refs = _extract_evidence_refs(messages)
    if _is_understanding_request(messages):
        if refs:
            payload = {
                "sufficient": True,
                "refined_search_terms": None,
                "reason": "Simulated stand-in assessment provided evidence appears relevant.",
            }
        else:
            payload = {
                "sufficient": False,
                "refined_search_terms": "relevant evidence from pinned papers",
                "reason": "Simulated stand-in assessment insufficient evidence provided.",
            }
        if target_tokens is not None:
            return _json_with_exact_tokens(payload, target_tokens)
        return json.dumps(payload, separators=(", ", ": "))

    if refs:
        cites = " ".join(f"[arXiv:{vid} p.{page}]" for vid, page, _ in refs[:2])
        base = (
            "[Simulated stand-in response] Based on the provided evidence from the pinned papers, "
            f"this deterministic simulator returns a grounded placeholder answer citing supplied sources {cites}. "
            "This is not a real LLM generated answer."
        )
    else:
        base = (
            "[Simulated stand-in response] No evidence passages were supplied to the simulator, "
            "so this deterministic response cannot provide cited claims. "
            "This is not a real LLM generated answer."
        )

    if target_tokens is None:
        return base
    return _fit_to_token_count(base, target_tokens)


def create_app() -> FastAPI:
    app = FastAPI(title="the-relay stand-in", version="0.1.0")
    mode_section = ModeSection()
    shaping = ShapingSection()
    counters = Counters()

    def _arm(spec: ModeSpec) -> None:
        mode_section.mode = spec.mode
        mode_section.times = spec.times
        mode_section.remaining = spec.times
        if spec.retry_after_s is not None:
            mode_section.retry_after_s = spec.retry_after_s
        mode_section.server_error_status = spec.server_error_status
        mode_section.hang_seconds = spec.hang_seconds

    def _consume_mode_slot() -> None:
        """Count down `times` after a request used the armed mode."""
        if mode_section.remaining is not None:
            mode_section.remaining -= 1
            if mode_section.remaining <= 0:
                fresh = ModeSection()  # auto-revert to ok
                mode_section.__dict__.update(fresh.__dict__)

    def _note_mode_served(name: str) -> None:
        counters.requests_total += 1
        counters.served_by_mode[name] = counters.served_by_mode.get(name, 0) + 1

    # ------------------------------------------------------------- admin

    @app.post("/_admin/mode")
    async def set_mode(spec: ModeSpec) -> dict:
        _arm(spec)
        return _state_payload()

    @app.post("/_admin/shaping")
    async def set_shaping(spec: ShapingSpec) -> dict:
        shaping.delay_ms = spec.delay_ms
        shaping.first_token_delay_ms = spec.first_token_delay_ms
        shaping.completion_tokens = spec.completion_tokens
        shaping.completion_text = spec.completion_text
        shaping.stream_chunk_delay_ms = spec.stream_chunk_delay_ms
        return _state_payload()

    @app.get("/_admin/state")
    async def get_state() -> dict:
        return _state_payload()

    @app.post("/_admin/reset")
    async def reset() -> dict:
        mode_section.__dict__.update(ModeSection().__dict__)
        shaping.__dict__.update(ShapingSection().__dict__)
        counters.__dict__.update(Counters().__dict__)
        return _state_payload()

    def _state_payload() -> dict:
        return {
            "mode": {
                "mode": str(mode_section.mode),
                "times": mode_section.times,
                "remaining": mode_section.remaining,
                "retry_after_s": mode_section.retry_after_s,
                "server_error_status": mode_section.server_error_status,
                "hang_seconds": mode_section.hang_seconds,
            },
            "shaping": {
                "delay_ms": shaping.delay_ms,
                "first_token_delay_ms": shaping.first_token_delay_ms,
                "completion_tokens": shaping.completion_tokens,
                "completion_text": shaping.completion_text,
                "stream_chunk_delay_ms": shaping.stream_chunk_delay_ms,
            },
            "counters": {
                "requests_total": counters.requests_total,
                "served_by_mode": dict(counters.served_by_mode),
            },
        }

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "stand-in"}

    # -------------------------------------------------------------- chat

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        raw = await request.body()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return _error_response(400, "request body is not valid JSON")
        if not isinstance(payload, dict):
            return _error_response(400, "request body must be a JSON object")

        model = payload.get("model")
        messages = payload.get("messages")
        if not isinstance(model, str) or not model:
            return _error_response(400, "'model' is required and must be a string")
        if not isinstance(messages, list) or not messages:
            return _error_response(400, "'messages' must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(
                message.get("content", ""), str
            ):
                return _error_response(400, "each message needs a string content")

        _note_mode_served(str(mode_section.mode))
        current = FailureMode(mode_section.mode)

        # ---- scripted failure modes (decided BEFORE any generation) ----
        if current in (FailureMode.RATE_LIMITED_DAILY, FailureMode.RATE_LIMITED_MINUTE):
            remaining_requests = "0" if current == FailureMode.RATE_LIMITED_DAILY else "57"
            headers = {
                "x-ratelimit-limit-requests": "14400",
                "x-ratelimit-remaining-requests": remaining_requests,
                "x-ratelimit-limit-tokens": "6000",
                "x-ratelimit-remaining-tokens": "5320",
                "retry-after": str(mode_section.retry_after_s),
            }
            _consume_mode_slot()
            return JSONResponse(
                status_code=429,
                headers=headers,
                content={
                    "error": {
                        "message": "stand-in scripted rate limit "
                        f"({current})",
                        "type": "rate_limit_error",
                    }
                },
            )
        if current == FailureMode.SERVER_ERROR:
            status = mode_section.server_error_status
            _consume_mode_slot()
            return JSONResponse(
                status_code=status,
                content={"error": {"message": "stand-in scripted server error", "type": "server_error"}},
            )
        if current == FailureMode.MALFORMED:
            _consume_mode_slot()
            return PlainTextResponse("<html>stand-in scripted malformed body</html>")

        # hang: block THIS handler; the client's timeout is the intended victim
        if current == FailureMode.HANG:
            await asyncio.sleep(mode_section.hang_seconds)
            _consume_mode_slot()
            # falls through to the normal ok path after the scripted stall

        if shaping.delay_ms > 0:
            await asyncio.sleep(shaping.delay_ms / 1000.0)

        # ---- deterministic generation ----
        if shaping.completion_text is not None:
            text = shaping.completion_text
            finish_reason = "stop"
        else:
            if shaping.completion_tokens is not None:
                # Admin shaping contract: exact token-count generation using the
                # documented word-cycle (preserved for load-test shaping).
                target = shaping.completion_tokens
                finish_reason = "stop"
                text = _completion_text(target)
            elif payload.get("max_tokens") is not None:
                # Contract: exact completion length target; finish_reason=length.
                # Preserved for existing clients/tests that rely on exact counts.
                target = int(payload["max_tokens"])
                if target < 0:
                    return _error_response(400, "'max_tokens' must be >= 0")
                finish_reason = "length"
                text = _completion_text(target)
            else:
                target = DEFAULT_COMPLETION_TOKENS
                finish_reason = "stop"
                # Default simulated response: readable and, when evidence is
                # provided in the prompt, grounded with valid citation tags.
                # Clearly labeled as simulated (never a real LLM answer).
                text = _generate_simulated_answer(messages, target_tokens=target)
        usage = {
            "prompt_tokens": prompt_tokens(messages),
            "completion_tokens": count_tokens(text),
            "total_tokens": None,  # filled below
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        wants_stream = bool(payload.get("stream"))
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

        if current == FailureMode.ABRUPT_CLOSE:
            _consume_mode_slot()
            if wants_stream:
                first_line = _sse(
                    _chunk_payload(model, {"content": text[:6]}, None)
                )
                # stream ends WITHOUT [DONE] and without finish_reason:
                # silent truncation the orchestrator must detect (WP7)
                return StreamingResponse(
                    _abrupt_stream(first_line), media_type="text/event-stream"
                )
            return StreamingResponse(
                _abrupt_stream(_OK_BODY_PREFIX), media_type="application/json"
            )

        if not wants_stream:
            return JSONResponse(_non_stream_payload(model, text, finish_reason, usage))
        return StreamingResponse(
            _sse_stream(model, text, finish_reason, usage, include_usage),
            media_type="text/event-stream",
        )

    def _error_response(status_code: int, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": {"message": message, "type": "invalid_request_error"}},
        )

    async def _abrupt_stream(first_bytes: str) -> AsyncIterator[str]:
        yield first_bytes  # body starts, then ends early — by design

    async def _sse_stream(
        model: str,
        text: str,
        finish_reason: str,
        usage: dict,
        include_usage: bool,
    ) -> AsyncIterator[str]:
        completion_id = _new_id()
        if shaping.first_token_delay_ms > 0:
            await asyncio.sleep(shaping.first_token_delay_ms / 1000.0)
        pieces = _stream_pieces(text)
        for index, piece in enumerate(pieces):
            if index > 0 and shaping.stream_chunk_delay_ms > 0:
                await asyncio.sleep(shaping.stream_chunk_delay_ms / 1000.0)
            yield _sse(_chunk_payload(model, {"content": piece}, None))
        yield _sse(_chunk_payload(model, {}, finish_reason))
        if include_usage:
            # OpenAI shape: final chunk carries usage with EMPTY choices
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": _created(),
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
        yield "data: [DONE]\n\n"

    def _new_id() -> str:
        return f"chatcmpl-standin-{counters.requests_total}"

    def _created() -> int:
        return int(time.time())

    def _chunk_payload(model: str, delta: dict, finish_reason: str | None) -> dict:
        return {
            "id": _new_id(),
            "object": "chat.completion.chunk",
            "created": _created(),
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    def _non_stream_payload(
        model: str, text: str, finish_reason: str, usage: dict
    ) -> dict:
        return {
            "id": _new_id(),
            "object": "chat.completion",
            "created": _created(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": usage,
        }

    return app


STANDIN_APP = create_app()
