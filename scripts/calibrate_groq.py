"""WP9: real Groq calibration (user-approved Steps A+B+C ONLY).

Run from the project root:

    .venv\\Scripts\\python.exe -m scripts.calibrate_groq --step models
    .venv\\Scripts\\python.exe -m scripts.calibrate_groq --step wire --model <id>
    .venv\\Scripts\\python.exe -m scripts.calibrate_groq --step runner --model <id>

Budget envelope (user-approved 2026-11-12): <=100 requests / ~50K tokens.
Steps: A = 1 GET /models (0 tokens); B = 2 tiny chat calls (~10 tokens
each); C = 3 full RelayRunner requests (~2-4K tokens in + ~200 out each).
NO forced 429s ever; NO load on real Groq. Every measured fact is
appended to results/logs/groq_calibration.jsonl (values only, no key).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.core.logging_setup import configure_logging, force_utf8_stdio
from app.core.settings import AppCapacity, get_settings
from app.orchestrator.events import EventSink
from app.orchestrator.runner import RelayRunner
from app.providers import ChatMessage, GROQ_DEFAULT_BASE_URL
from app.providers.groq import GroqProvider
from app.reliability.breaker import CircuitBreaker
from app.reliability.capacity import RequestGate
from app.reliability.limiter import HeaderQuotaLimiter
from app.reliability.recorder import RequestRecorder, utc_now_iso_ms
from app.retrieval.retriever import Retriever

CALIB_LOG = Path("results") / "logs" / "groq_calibration.jsonl"

# user-approved counts per step — the script refuses to exceed these
STEP_BUDGET = {"models": 1, "wire": 2, "runner": 3}

WIRE_PROMPT = "Reply with the single word: ok"

# one calibration question per pinned paper (from the WP3 spot-check probes)
RUNNER_QUESTIONS = (
    "How does multi-head attention extend the attention mechanism?",
    "How does Moshi talk and listen at the same time?",
    "How are chunk boundaries selected robustly?",
)


def _log(step: str, kind: str, **data: object) -> None:
    """Append one measured-fact line; key never enters here."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "step": step,
        "kind": kind,
        **data,
    }
    CALIB_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CALIB_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _print_limits(snapshot) -> None:
    fields = (
        "rpd_limit",
        "rpd_remaining",
        "tpm_limit",
        "tpm_remaining",
        "rpm_limit",
        "rpm_remaining",
    )
    print("  observed limits: " + json.dumps({f: getattr(snapshot, f) for f in fields}))


async def step_models() -> None:
    print(f"Step A: GET {GROQ_DEFAULT_BASE_URL}/models (1 request, 0 tokens)")
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.get(
            f"{GROQ_DEFAULT_BASE_URL}/models",
            headers={"Authorization": f"Bearer {_key()}"},
        )
    print(f"  status: {response.status_code}")
    ratelimit_headers = {
        k: v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit")
    }
    if ratelimit_headers:
        print("  x-ratelimit-* on /models: " + json.dumps(ratelimit_headers))
    response.raise_for_status()
    data = response.json()["data"]
    ids = sorted(m["id"] for m in data)
    print(f"  models available to this account: {len(ids)}")
    for model_id in ids:
        print(f"    - {model_id}")
    _log("models", "list_ok", n_models=len(ids), status=response.status_code)


def _key() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    import os

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set — create .env first")
    return key


async def step_wire(model: str) -> None:
    print(f"Step B: wire probe on model={model} (2 requests, ~10 tokens each)")
    provider = GroqProvider()

    # --- non-streaming -------------------------------------------------
    started = time.perf_counter()
    response = await provider.complete(
        [ChatMessage(role="user", content=WIRE_PROMPT)],
        model=model,
        temperature=0.0,
        max_tokens=64,
    )
    non_stream_ms = (time.perf_counter() - started) * 1000.0
    print("[1/2 non-stream]")
    print(f"  content: {response.content!r}")
    print(f"  finish_reason: {response.finish_reason} | transport_ms: {response.transport_ms:.1f} | wall_ms: {non_stream_ms:.1f}")
    print(f"  usage: prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens} total={response.usage.total_tokens}")
    _print_limits(response.limits)
    _log(
        "wire",
        "complete_ok",
        model=model,
        transport_ms=round(response.transport_ms, 1),
        wall_ms=round(non_stream_ms, 1),
        finish_reason=response.finish_reason,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        **{f: getattr(response.limits, f) for f in ("rpd_limit", "rpd_remaining", "tpm_limit", "tpm_remaining", "rpm_limit", "rpm_remaining")},
    )

    # --- streaming ------------------------------------------------------
    started = time.perf_counter()
    ttft_ms = None
    pieces: list[str] = []
    finish_reason = None
    usage = None
    async for chunk in provider.stream(
        [ChatMessage(role="user", content=WIRE_PROMPT)],
        model=model,
        temperature=0.0,
        max_tokens=64,
    ):
        if chunk.content_delta and ttft_ms is None:
            ttft_ms = (time.perf_counter() - started) * 1000.0
        if chunk.content_delta:
            pieces.append(chunk.content_delta)
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
    total_ms = (time.perf_counter() - started) * 1000.0
    print("[2/2 stream]")
    print(f"  reassembled: {''.join(pieces)!r}")
    print(f"  ttft_ms: {None if ttft_ms is None else round(ttft_ms, 1)} | total_ms: {total_ms:.1f} | finish_reason: {finish_reason}")
    if usage is not None:
        print(f"  usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
        _log(
            "wire",
            "stream_ok",
            model=model,
            ttft_ms=None if ttft_ms is None else round(ttft_ms, 1),
            total_ms=round(total_ms, 1),
            finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
    await provider.aclose()
    print("Step B done (2 requests).")


async def step_runner(model: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    from scripts.run_api import _build_retrievers

    retrievers = _build_retrievers(Path(settings.papers_dir))
    # warmup so the measured TTFT is pipeline-only (same rule as run_api)
    for setting, retriever in retrievers.items():
        await retriever.retrieve(f"boot warmup probe ({setting})")

    provider = GroqProvider()
    capacity = AppCapacity(
        in_flight_limit=settings.capacity.real_in_flight_limit,
        queue_limit=settings.capacity.queue_limit,
    )
    gate = RequestGate(capacity)
    breaker = CircuitBreaker(settings.breaker)
    limiter = HeaderQuotaLimiter()
    recorder = RequestRecorder(settings.results_dir / "results.csv")
    sink = EventSink(settings.results_dir / "events")
    runner = RelayRunner(
        retrievers=retrievers,
        provider=provider,
        model=model,
        gate=gate,
        breaker=breaker,
        limiter=limiter,
        pipeline=settings.pipeline,
        retry_policy=settings.retry,
        recorder=recorder,
        sink=sink,
    )

    print(f"Step C: full-runner calibration on model={model} (3 requests, real rows)")
    for index, question in enumerate(RUNNER_QUESTIONS, start=1):
        request_id = f"wp9-calib-{index}"
        outcome = await runner.run(
            request_id=request_id,
            scenario="wp9-calib",
            question=question,
            retrieval_setting="B",
            start_time=utc_now_iso_ms(),
        )
        print(f"[{index}/3] {request_id}: {question[:60]}")
        print(
            f"  status={outcome.status} kind={outcome.outcome_kind} refused={outcome.refused}"
            f" citations_valid={outcome.citations_valid}"
        )
        print(
            f"  first_token_ms={outcome.first_token_ms} total_ms={round(outcome.total_ms, 1)}"
            f" retries={outcome.retries} tokens_in={outcome.tokens_in} tokens_out={outcome.tokens_out}"
        )
        print(f"  answer: {outcome.answer[:140]}")
        _log(
            "runner",
            "run_outcome",
            request_id=request_id,
            model=model,
            status=outcome.status,
            outcome_kind=outcome.outcome_kind,
            refused=outcome.refused,
            citations_valid=outcome.citations_valid,
            first_token_ms=outcome.first_token_ms,
            total_ms=round(outcome.total_ms, 1),
            retries=outcome.retries,
            tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out,
            retrieval_setting=outcome.retrieval_setting,
        )
    print(f"rows -> {recorder.path}")
    print(f"events -> {sink.directory}")
    await provider.aclose()
    print("Step C done (3 requests).")


async def _main() -> None:
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description="WP9 Groq calibration")
    parser.add_argument("--step", choices=("models", "wire", "runner"), required=True)
    parser.add_argument(
        "--model",
        default=None,
        help="required for wire/runner; see Step A output for available ids",
    )
    args = parser.parse_args()

    if args.step in ("wire", "runner") and not args.model:
        raise SystemExit("--model is required for this step (no model is assumed)")
    print(f"approved budget for this step: {STEP_BUDGET[args.step]} request(s)\n")
    if args.step == "models":
        await step_models()
    elif args.step == "wire":
        await step_wire(args.model)
    else:
        await step_runner(args.model)


if __name__ == "__main__":
    asyncio.run(_main())
