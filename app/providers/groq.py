"""Groq provider — HTTPX against the documented OpenAI-compatible endpoint (D1).

Raw HTTP (not the vendor SDK) because we must capture rate-limit headers
verbatim, raw error bodies, and measure transport boundaries precisely.
Wire-format behavior lives in `compat.py`; this file adds only credentials.

The API key is read from the environment (`.env` via python-dotenv) at
construction; it exists only inside the Authorization header — never
logged, never returned, never stored in settings.
"""
from __future__ import annotations

import os

import httpx

from app.core.logging_setup import get_logger
from app.providers import GROQ_DEFAULT_BASE_URL
from app.providers.compat import OpenAICompatTransport
from app.reliability.errors import ConfigurationError

log = get_logger("relay.providers.groq")


class GroqProvider(OpenAICompatTransport):
    name = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        request_timeout_s: float = 20.0,
    ) -> None:
        from dotenv import load_dotenv

        load_dotenv(override=False)  # .env supplies the key when present
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set; put it in .env (never in code/chat/logs)"
            )
        super().__init__(
            base_url=base_url or GROQ_DEFAULT_BASE_URL,
            http=http,
            request_timeout_s=request_timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )
