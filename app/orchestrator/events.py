"""Per-request structured event logs (WP7) — the D13 supporting evidence.

`results.csv` alone cannot explain internal latency, so every request gets
its own JSONL event file recording: queue wait/rejection, stage durations,
retrieval details, per-attempt provider calls (status, transport time,
reported usage, observed limit headers), retry events, first-token and
completion timestamps, cancellation, token-accounting bookings, and the
typed terminal outcome. One file per request keeps joins with results.csv
trivial (same request_id).

Nothing here fabricates: values are measured or explicitly absent (None).
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging_setup import get_logger

log = get_logger("relay.orchestrator.events")

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def utc_now_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class RequestEventLog:
    """Append-only JSONL event stream for ONE request."""

    def __init__(self, path: Path, request_id: str) -> None:
        self.path = path
        self.request_id = request_id
        self._started_perf = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "a", encoding="utf-8", newline="\n")

    def emit(self, kind: str, **fields: object) -> None:
        record = {
            "ts": utc_now_iso_ms(),
            "elapsed_ms": round((time.perf_counter() - self._started_perf) * 1000, 1),
            "request_id": self.request_id,
            "kind": kind,
        }
        record.update(fields)
        self._handle.write(
            f"{json_dumps(record)}\n"
        )
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # never let bookkeeping mask a real outcome
            log.warning("event_log_close_failed", extra={"request_id": self.request_id})


def json_dumps(record: dict) -> str:
    import json

    return json.dumps(record, default=str, separators=(",", ":"), ensure_ascii=False)


class EventSink:
    """Creates per-request event files under one directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def open(self, request_id: str) -> RequestEventLog:
        safe = _SAFE_ID.sub("_", request_id)
        if not safe:
            safe = "unnamed_request"
        return RequestEventLog(self.directory / f"{safe}.jsonl", request_id)


class NullEventSink:
    """No-op sink (used when event files are explicitly disabled)."""

    def open(self, request_id: str) -> RequestEventLog:  # pragma: no cover
        raise NotImplementedError("NullEventSink must never be opened")
