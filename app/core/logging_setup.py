"""Structured JSON logging with secret redaction.

Every event is one parseable JSON line. Automatic fields: ``ts``, ``level``,
``logger``, ``msg``, ``request_id`` (from the request context). Any keyword
passed via ``extra=`` merges into the record, so business fields (stage,
provider, retry count, duration_ms ...) arrive top-level — no log parsing
games when we later "trace one slow request".

``redact()`` scrubs credential-shaped substrings from anything that might
carry provider headers or error bodies before they reach a log.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from datetime import datetime, timezone

from app.core.request_context import get_request_id

# Attributes every logging.LogRecord carries by default; anything else that
# ends up in record.__dict__ came from our own ``extra=`` and is kept.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime", "stacklevel",
}

_JSON_HANDLER_ATTR = "relay_json_handler"

_BEARER_ASSIGN = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?([A-Za-z0-9\-._~+/=]{8,})"
)
_KEY_ASSIGN = re.compile(
    r"(?i)((?:api[_-]?key|x-api-key)\s*[:=]\s*)[A-Za-z0-9\-._+/]{6,}"
)
_SAK_PREFIX = re.compile(r"(?i)gsk_[A-Za-z0-9\-_.]{10,}")


def redact(text: str | None) -> str:
    """Replace credential-shaped substrings with ``<redacted>``."""
    if not text:
        return ""
    text = _BEARER_ASSIGN.sub(r"\1<redacted>", text)
    text = _KEY_ASSIGN.sub(r"\1<redacted>", text)
    text = _SAK_PREFIX.sub("<redacted>", text)
    return text


class JsonFormatter(logging.Formatter):
    """Render a record as one compact JSON object line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exc"] = "".join(
                traceback.format_exception(*record.exc_info)
            )[-2000:]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """Attach the JSON handler to the root logger exactly once (idempotent,
    so pytest re-entry does not duplicate lines)."""
    root = logging.getLogger()
    if not any(getattr(h, _JSON_HANDLER_ATTR, False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        setattr(handler, _JSON_HANDLER_ATTR, True)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(level)
    return logging.getLogger("relay")


def get_logger(name: str) -> logging.Logger:
    """Namespaced child logger, e.g. ``get_logger('relay.providers.groq')``."""
    if not name.startswith("relay"):
        name = f"relay.{name}"
    return logging.getLogger(name)


def force_utf8_stdio() -> None:
    """Windows console safety: write UTF-8 with replacement instead of raising
    UnicodeEncodeError on characters cp1252 cannot map (paper text has ∗, µ, ...)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # non-reconfigurable stream (tests, IDEs)
            pass
