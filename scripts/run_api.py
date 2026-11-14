"""Run the-relay HTTP API (WP8) with the REAL retrieval indexes.

Usage (from the project root):
    .venv\\Scripts\\python.exe -m scripts.run_api
    .venv\\Scripts\\python.exe -m scripts.run_api --provider groq --model <id>

Defaults to the stand-in provider (http://127.0.0.1:8100 — start it with
`python -m scripts.run_standin`). `--provider groq` needs GROQ_API_KEY in
.env and an explicit `--model`: no model is assumed before WP9 measurement
(DOCUMENTED in settings.groq.default_model = "").

The real Groq gate applies here too: nothing in WP8 sends live Groq
traffic; choosing groq only re-wires the transport. With groq, the app's
in-flight cap drops to the real-tier limit (settings.capacity.
real_in_flight_limit).

rows  -> results/results.csv   (D13 contract, same recorder as the harness)
events-> results/events/       (per-request JSONL, D13 supporting logs)
retrieval_setting A|B|C resolves via results/retrieval_settings.json
(written at boot if missing).
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn

from app.api.service import create_api_app
from app.core.logging_setup import configure_logging, force_utf8_stdio
from app.core.settings import AppCapacity, get_settings
from app.orchestrator.runner import RelayRunner
from app.providers.groq import GroqProvider
from app.providers.standin import StandInProvider
from app.reliability.breaker import CircuitBreaker
from app.reliability.capacity import RequestGate
from app.reliability.limiter import HeaderQuotaLimiter
from app.reliability.recorder import RequestRecorder, save_retrieval_settings
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever

# identical to scripts/retrieval_spotcheck.py: dir + whether rerank applies
CONFIG_DIRS = {
    "A": ("config_a", False),
    "B": ("config_b", False),
    "C": ("config_c", True),
}


def _build_retrievers(papers_dir: Path) -> dict[str, Retriever]:
    indexes_dir = papers_dir / "indexes"
    embedder = Embedder()  # default all-MiniLM-L6-v2, matches built indexes
    retrievers: dict[str, Retriever] = {}
    for setting, (dir_name, rerank) in CONFIG_DIRS.items():
        index_dir = indexes_dir / dir_name
        if not (index_dir / "meta.json").is_file():
            raise SystemExit(
                f"MISSING index at {index_dir}\nRun: .venv\\Scripts\\python.exe -m scripts.build_index"
            )
        retrievers[setting] = Retriever(
            index_dir, embedder, reranker=CrossEncoderReranker() if rerank else None
        )
    return retrievers


async def _warmup(retrievers: dict[str, Retriever]) -> None:
    """Force the lazy index/embedding-model load BEFORE serving, so the
    first real request's TTFT measures the pipeline, not a ~25 s model
    cold load (measured at the WP8 smoke run; same trick as the
    retrieval spot-check's warmup probe)."""
    for setting, retriever in retrievers.items():
        await retriever.retrieve(f"boot warmup probe ({setting})")


def build_relay_app(
    *,
    provider_name: str,
    model: str | None,
    standin_base_url: str | None = None,
) -> tuple["FastAPI", dict]:
    """Wire the FULL relay app (real retrievers + reliability stack).

    Shared by `main()` (CLI server) and the WP10 load harness, so both
    exercise the identical boot path. Returns (app, components) where
    components carries recorder/sink/provider/gate for callers that need
    introspection.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    retrievers = _build_retrievers(settings.papers_dir)

    if provider_name == "groq":
        if not model:
            raise SystemExit(
                "model is required with provider=groq (no model is assumed "
                "before WP9 measures real behavior)"
            )
        provider = GroqProvider()  # ConfigurationError BEFORE any traffic if key absent
        capacity = AppCapacity(
            in_flight_limit=settings.capacity.real_in_flight_limit,
            queue_limit=settings.capacity.queue_limit,
        )
        model_used, source_label = model, "real"
    else:
        provider = StandInProvider(
            base_url=standin_base_url or settings.standin.base_url
        )
        capacity = AppCapacity(
            in_flight_limit=settings.capacity.in_flight_limit,
            queue_limit=settings.capacity.queue_limit,
        )
        model_used, source_label = model or "standin-sim", "stand-in"

    gate = RequestGate(capacity)
    breaker = CircuitBreaker(settings.breaker)
    limiter = HeaderQuotaLimiter()
    recorder = RequestRecorder(settings.results_dir / "results.csv")
    from app.orchestrator.events import EventSink

    sink = EventSink(settings.results_dir / "events")
    save_retrieval_settings(settings.results_dir / "retrieval_settings.json")

    runner = RelayRunner(
        retrievers=retrievers,
        provider=provider,
        model=model_used,
        gate=gate,
        breaker=breaker,
        limiter=limiter,
        pipeline=settings.pipeline,
        retry_policy=settings.retry,
        recorder=recorder,
        sink=sink,
    )
    app = create_api_app(
        runner=runner, gate=gate, breaker=breaker, limiter=limiter, recorder=recorder
    )
    return app, {
        "recorder": recorder,
        "sink": sink,
        "provider": provider,
        "gate": gate,
        "breaker": breaker,
        "limiter": limiter,
        "retrievers": retrievers,
        "source_label": source_label,
        "model": model_used,
    }


def main() -> None:
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description="the-relay API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--provider", choices=("standin", "groq"), default="standin")
    parser.add_argument(
        "--model",
        default=None,
        help="provider model id; REQUIRED with --provider groq (no model is assumed)",
    )
    args = parser.parse_args()

    app, parts = build_relay_app(
        provider_name=args.provider, model=args.model
    )

    print(
        f"the-relay API on http://{args.host}:{args.port} | provider={args.provider} "
        f"(source={parts['source_label']}, model={parts['model']}) | settings A/B/C | "
        f"rows={parts['recorder'].path} | events={parts['sink'].directory}"
    )
    asyncio.run(_warmup(parts["retrievers"]))  # measured: cold model load ~25 s on this machine
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
