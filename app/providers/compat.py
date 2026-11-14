"""Shared OpenAI-compatible transport used by both providers (WP5).

One implementation of: request building, SSE parsing, status->typed-error
mapping, rate-limit header capture, and transport timing. Groq and the
stand-in differ ONLY in credentials/base URL — nothing else, so their
behavior cannot drift apart.

Pure transport: no retries, no breaker, no limiter calls here (WP7 wraps).
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from app.core.logging_setup import get_logger
from app.providers import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    ChatUsage,
    LLMProvider,
    limits_from_headers,
    raise_for_status,
    wrap_transport_error,
)
from app.reliability.limiter import ProviderLimitsSnapshot

log = get_logger("relay.providers.compat")

_STREAM_DONE = "[DONE]"


class OpenAICompatTransport(LLMProvider):
    name: str = "openai-compat"

    def __init__(
        self,
        *,
        base_url: str,
        http: httpx.AsyncClient | None = None,
        request_timeout_s: float = 20.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_timeout_s = request_timeout_s
        self._extra_headers = dict(headers or {})
        self._owns_client = http is None
        self._http = http or httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(request_timeout_s)
        )
        self.last_limits: ProviderLimitsSnapshot | None = None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    # ------------------------------------------------------------ complete

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> ChatResponse:
        payload: dict = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        started = time.perf_counter()
        try:
            response = await self._http.post(
                "/chat/completions",
                json=payload,
                headers=self._extra_headers,
                timeout=timeout_s or self.default_timeout_s,
            )
        except httpx.HTTPError as error:
            raise wrap_transport_error(error) from error
        transport_ms = (time.perf_counter() - started) * 1000.0

        body = response.text
        raise_for_status(response.status_code, response.headers, body)
        self.last_limits = limits_from_headers(response.headers)
        try:
            data = response.json()
        except json.JSONDecodeError as error:
            from app.reliability.errors import MalformedRequestError

            raise MalformedRequestError(
                "provider returned non-JSON body", body=body[:500]
            ) from error
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage_raw = data.get("usage")
        return ChatResponse(
            content=str(message.get("content") or ""),
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            usage=ChatUsage(**usage_raw) if isinstance(usage_raw, dict) else None,
            limits=self.last_limits,
            transport_ms=transport_ms,
        )

    # -------------------------------------------------------------- stream

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        payload: dict = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            async with self._http.stream(
                "POST", "/chat/completions", json=payload, headers=self._extra_headers
            ) as response:
                body = ""
                if response.status_code != 200:
                    raw = await response.aread()
                    body = raw.decode("utf-8", errors="replace")
                raise_for_status(response.status_code, response.headers, body)
                self.last_limits = limits_from_headers(response.headers)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == _STREAM_DONE:
                        return
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        log.warning("unparseable_stream_event")
                        continue
                    choices = event.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    usage_raw = event.get("usage")
                    yield ChatChunk(
                        content_delta=str(delta.get("content") or ""),
                        finish_reason=choices[0].get("finish_reason") if choices else None,
                        usage=ChatUsage(**usage_raw) if isinstance(usage_raw, dict) else None,
                    )
        except httpx.HTTPError as error:
            raise wrap_transport_error(error) from error
