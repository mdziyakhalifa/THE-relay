"""WP1 smoke run — visible demonstration that core observability works.

Run:  .venv\\Scripts\\python.exe scripts\\smoke_logging.py

Shows: JSON log lines stamped with one request_id, a stage-timing snapshot
measured with the real clock, and redaction scrubbing a fake credential.
"""
from __future__ import annotations

import asyncio

from app.core.logging_setup import configure_logging, get_logger, redact
from app.core.request_context import request_scope
from app.core.settings import get_settings
from app.core.stage_metrics import StageMetrics


async def main() -> None:
    configure_logging("INFO")
    log = get_logger("relay.smoke")
    settings = get_settings()

    with request_scope() as request_id:
        log.info(
            "smoke_start",
            extra={
                "component": "core",
                "max_model_calls": settings.pipeline.max_model_calls,
                "deadline_ms": settings.pipeline.request_deadline_ms,
            },
        )
        metrics = StageMetrics()  # real clock here
        async with metrics.timed("fake_io"):
            await asyncio.sleep(0.02)
        async with metrics.timed("fake_generation"):
            await asyncio.sleep(0.05)
        log.info("stage_snapshot", extra={"stages": metrics.snapshot(),
                                          "total_ms": metrics.total_ms})
        log.info("redaction_demo",
                 extra={"header_example": redact("Authorization: Bearer gsk_DEMO_key_01234")})

    print(f"[smoke] OK request_id={request_id}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
