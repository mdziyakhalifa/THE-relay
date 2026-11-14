"""Request-scoped identity for async pipelines.

A ``ContextVar`` carries ``request_id`` through queueing, worker tasks, retry
loops and streaming generators without threading it through every signature.
ContextVars are task-local by default, which is exactly the async-safe
behaviour we need; child tasks (with ``asyncio.create_task``) inherit the
value set before the spawn.
"""
from __future__ import annotations

import contextlib
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("relay_request_id", default=None)


def generate_request_id() -> str:
    """Short, collision-safe id used consistently in logs, CSV rows and event
    logs (format: ``req-<12 hex chars>``)."""
    return f"req-{uuid.uuid4().hex[:12]}"


@contextlib.contextmanager
def request_scope(request_id: str | None = None):
    """Bind a request_id to the current context for the duration of the block."""
    rid = request_id or generate_request_id()
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


def get_request_id() -> str:
    """Current request id, or ``-`` when running outside any request scope
    (e.g. startup ticks) so log fields are never missing."""
    rid = _request_id.get()
    return rid if rid is not None else "-"
