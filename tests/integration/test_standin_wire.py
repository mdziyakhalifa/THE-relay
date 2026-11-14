"""Real-wire failure-mode tests (raw asyncio sockets; WP6).

MockTransport can fake status codes, but NOT network-level faults. These
tests run a raw TCP server on an ephemeral localhost port to produce REAL
connection behavior, then assert the provider maps it to the honest
typed outcome:

- hang (accept, never answer)  -> client timeout -> TransientNetworkError
- mid-stream truncation (chunked body closed early)
                               -> incomplete read -> TransientNetworkError
"""
import asyncio

import httpx
import pytest

from app.providers import ChatMessage
from app.providers.standin import StandInProvider
from app.reliability.errors import TransientNetworkError

MESSAGES = [ChatMessage(role="user", content="hello")]


async def _start_raw_server(handler) -> tuple[asyncio.AbstractServer, int]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _read_request(reader: asyncio.StreamReader) -> None:
    """Read headers, then the body (content-length framed)."""
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        buffer += await asyncio.wait_for(reader.read(65536), timeout=5.0)
    headers, _, rest = buffer.partition(b"\r\n\r\n")
    length = 0
    for line in headers.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    while len(rest) < length:
        rest += await asyncio.wait_for(reader.read(65536), timeout=5.0)


async def test_hang_then_client_timeout_maps_to_transient_network_error():
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _read_request(reader)
            await asyncio.sleep(5.0)  # accept, then never answer in time
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
            await writer.drain()
        except (asyncio.CancelledError, ConnectionError):
            pass

    server, port = await _start_raw_server(handler)
    try:
        provider = StandInProvider(
            base_url=f"http://127.0.0.1:{port}", request_timeout_s=0.5
        )
        try:
            with pytest.raises(TransientNetworkError):
                await provider.complete(MESSAGES, model="m")
        finally:
            await provider.aclose()
    finally:
        server.close()


async def test_midstream_truncation_maps_to_transient_network_error():
    partial_chunk = b'data: {"choices": [{"delta": {"content": "par'

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _read_request(reader)
            # announce 200 chunk bytes, deliver only a prefix, then close:
            # h11 sees an INCOMPLETE chunked body -> RemoteProtocolError
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: close\r\n"
                b"\r\n"
                b"C8\r\n" + partial_chunk
            )
            await writer.drain()
            writer.close()
        except (asyncio.CancelledError, ConnectionError):
            pass

    server, port = await _start_raw_server(handler)
    try:
        client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(5.0)
        )
        provider = StandInProvider(
            base_url=f"http://127.0.0.1:{port}", http=client
        )
        try:
            consumed = 0
            with pytest.raises(TransientNetworkError):
                async for _chunk in provider.stream(MESSAGES, model="m"):
                    consumed += 1
            assert consumed >= 0
        finally:
            await client.aclose()
            await provider.aclose()
    finally:
        server.close()
        await server.wait_closed()
