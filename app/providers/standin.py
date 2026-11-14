"""Stand-in provider client — talks to OUR local simulator (WP6 service).

Contract the stand-in service must fulfill (WP6 implements it):
- OpenAI-compatible POST /chat/completions (non-streaming and SSE streaming).
- A `usage` block computed with the service's DOCUMENTED tokenizer, reported
  on non-streaming responses and (when include_usage is honored) on the
  stream's final chunk. The client reads reported usage only — it never
  estimates on behalf of the service.
- No authentication (localhost); rate-limit headers optional — absent
  headers degrade to an empty snapshot (honest unknowns, see D12).

Deliberately the same wire behavior as the Groq client (shared transport),
so load experiments exercise the identical code path apart from the URL.
"""
from __future__ import annotations

import httpx

from app.core.logging_setup import get_logger
from app.providers.compat import OpenAICompatTransport

log = get_logger("relay.providers.standin")


class StandInProvider(OpenAICompatTransport):
    name = "stand-in"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        request_timeout_s: float = 20.0,
    ) -> None:
        if base_url is None:
            from app.core.settings import get_settings

            base_url = get_settings().standin.base_url
        super().__init__(
            base_url=base_url,
            http=http,
            request_timeout_s=request_timeout_s,
            headers=None,  # local service; no auth
        )
