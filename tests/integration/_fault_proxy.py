"""Fault-injection HTTP proxy in front of Ollama for integration tests.

Sits between marshal and the real Ollama, transparently forwarding
requests by default but with hooks to selectively fail/delay/disconnect
or return canned responses. Required for tests that need controllable
Ollama failures (Surface A retry, failed-preload sentinel, Surface C2
unexpected-unload via fake /api/ps) — all of which can't be reproduced
reliably by toggling the real daemon.

Implementation: bare ``asyncio.start_server`` HTTP/1.1 parser, with an
``httpx.AsyncClient`` for the upstream forward leg. No aiohttp
dependency. ~200 LOC.

Usage::

    from tests.integration._fault_proxy import fault_proxy

    async with fault_proxy() as proxy:
        proxy.fail_next("/api/generate", times=2, status=503)
        # marshal_config.ollama.host = proxy.url
        ...

Hooks:
- ``fail_next(path, times, status)`` — return ``status`` ``times``
  times for any request whose path starts with ``path``, then resume
  pass-through.
- ``disconnect_next(path, times)`` — close the connection without
  sending headers (triggers ``httpx.ConnectError`` / ``RemoteProtocolError``
  on the client side).
- ``delay_next(path, seconds)`` — sleep before forwarding (use to
  trigger ``httpx.ReadTimeout`` when the client has a tight read deadline).
- ``fake_response(path, body, times=None)`` — return ``body`` (dict or
  bytes) ``times`` times (or indefinitely if ``times is None``).

All hooks fire FIFO — multiple queued faults on the same path apply
in order.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

# Bytes returned to the client for status-only fault responses.
_REASON_PHRASES = {
    500: b"Internal Server Error",
    502: b"Bad Gateway",
    503: b"Service Unavailable",
    504: b"Gateway Timeout",
}


@dataclass
class _FailSpec:
    """Return ``status`` with optional ``body`` instead of forwarding."""

    status: int
    body: bytes = b""


@dataclass
class _FakeSpec:
    """Return canned ``body`` (raw bytes) instead of forwarding.

    ``times`` of None means "indefinitely" — useful for tests that
    fake /api/ps on every poll for the test's duration.
    """

    body: bytes
    remaining: int | None  # None = forever, int = countdown


@dataclass
class _DisconnectSpec:
    """Close the socket without writing any HTTP response."""


@dataclass
class _DelaySpec:
    """Sleep ``seconds`` before forwarding."""

    seconds: float


_Spec = _FailSpec | _FakeSpec | _DisconnectSpec | _DelaySpec


class FaultProxy:
    """Test-side HTTP proxy with fault-injection hooks. See module docs."""

    def __init__(self, upstream: str = "http://localhost:11434") -> None:
        self._upstream = upstream.rstrip("/")
        # Path-prefix → FIFO queue of fault specs. Empty queue → pass through.
        self._queues: dict[str, deque[_Spec]] = defaultdict(deque)
        self._server: asyncio.base_events.Server | None = None
        self._port: int = 0

    # -- public API ---------------------------------------------------

    @property
    def url(self) -> str:
        """The proxy's base URL (e.g. ``http://localhost:54321``)."""
        if self._port == 0:
            msg = "fault proxy not started — use `async with fault_proxy()`"
            raise RuntimeError(msg)
        return f"http://127.0.0.1:{self._port}"

    def fail_next(self, path: str, times: int = 1, status: int = 503) -> None:
        """Queue ``times`` failures (returning ``status``) for ``path``."""
        for _ in range(times):
            self._queues[path].append(_FailSpec(status=status))

    def disconnect_next(self, path: str, times: int = 1) -> None:
        """Queue ``times`` socket-close-without-response events for ``path``."""
        for _ in range(times):
            self._queues[path].append(_DisconnectSpec())

    def delay_next(self, path: str, seconds: float) -> None:
        """Sleep ``seconds`` before forwarding the next request to ``path``."""
        self._queues[path].append(_DelaySpec(seconds=seconds))

    def fake_response(
        self, path: str, body: dict[str, Any] | bytes, times: int | None = None
    ) -> None:
        """Return ``body`` (dict → JSON, or raw bytes) for ``path``.

        ``times=None`` means indefinitely.
        """
        payload = json.dumps(body).encode() if isinstance(body, dict) else body
        self._queues[path].append(_FakeSpec(body=payload, remaining=times))

    # -- lifecycle ----------------------------------------------------

    async def _start(self) -> None:
        """Bind to an ephemeral port and start accepting connections."""
        # Reserve a free port. We bind, read, close, then start the
        # asyncio server on the same port. Tiny race window but
        # acceptable for test infrastructure.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            self._port = sock.getsockname()[1]
        self._server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", self._port
        )

    async def _stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # -- request handling --------------------------------------------

    def _next_spec(self, path: str) -> _Spec | None:
        """Pop the next queued spec for any prefix matching ``path``.

        Longest-prefix match wins, so a more specific queued path
        takes precedence over a broader one. Returns None when no
        prefix has a queued spec.
        """
        candidates = sorted(
            (p for p in self._queues if path.startswith(p)),
            key=len,
            reverse=True,
        )
        for prefix in candidates:
            queue = self._queues[prefix]
            if not queue:
                continue
            spec = queue[0]
            # _FakeSpec with remaining=None stays in queue; everything
            # else pops once consumed.
            if isinstance(spec, _FakeSpec) and spec.remaining is None:
                return spec
            if isinstance(spec, _FakeSpec) and spec.remaining is not None:
                spec.remaining -= 1
                if spec.remaining <= 0:
                    queue.popleft()
                return spec
            queue.popleft()
            return spec
        return None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Per-connection handler. One request → one response → close."""
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            try:
                method, path, _version = request_line.decode().split(" ", 2)
            except ValueError:
                return

            # Read headers until empty line.
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                key, _, val = line.decode().partition(":")
                headers[key.strip().lower()] = val.strip()

            # Read body if Content-Length is set.
            content_length = int(headers.get("content-length", "0") or "0")
            body = await reader.readexactly(content_length) if content_length else b""

            spec = self._next_spec(path)
            if spec is None:
                await self._forward(method, path, headers, body, writer)
            elif isinstance(spec, _FailSpec):
                await self._write_error(writer, spec.status)
            elif isinstance(spec, _DisconnectSpec):
                # Close without writing anything — client sees ConnectError
                # or RemoteProtocolError depending on timing.
                pass
            elif isinstance(spec, _DelaySpec):
                await asyncio.sleep(spec.seconds)
                await self._forward(method, path, headers, body, writer)
            elif isinstance(spec, _FakeSpec):
                await self._write_json(writer, 200, spec.body)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    async def _forward(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Pass through to the real Ollama and stream the response back."""
        url = f"{self._upstream}{path}"
        # Drop hop-by-hop headers + host (httpx sets its own).
        forward_headers = {
            k: v
            for k, v in headers.items()
            if k not in ("host", "content-length", "connection")
        }
        try:
            async with (
                httpx.AsyncClient(timeout=300) as client,
                client.stream(
                    method, url, headers=forward_headers, content=body
                ) as resp,
            ):
                # Write status line + headers (close-delimited body).
                writer.write(f"HTTP/1.1 {resp.status_code} OK\r\n".encode())
                writer.write(b"Connection: close\r\n")
                # Forward content-type so client parses correctly.
                if "content-type" in resp.headers:
                    writer.write(
                        f"Content-Type: {resp.headers['content-type']}\r\n".encode()
                    )
                writer.write(b"\r\n")
                async for chunk in resp.aiter_raw():
                    writer.write(chunk)
                    await writer.drain()
        except httpx.HTTPError:
            # Upstream failure — surface as 502 to the test's marshal.
            await self._write_error(writer, 502)

    @staticmethod
    async def _write_error(writer: asyncio.StreamWriter, status: int) -> None:
        reason = _REASON_PHRASES.get(status, b"Error")
        writer.write(f"HTTP/1.1 {status} ".encode() + reason + b"\r\n")
        writer.write(b"Connection: close\r\n")
        writer.write(b"Content-Length: 0\r\n\r\n")
        await writer.drain()

    @staticmethod
    async def _write_json(
        writer: asyncio.StreamWriter, status: int, body: bytes
    ) -> None:
        writer.write(f"HTTP/1.1 {status} OK\r\n".encode())
        writer.write(b"Connection: close\r\n")
        writer.write(b"Content-Type: application/json\r\n")
        writer.write(f"Content-Length: {len(body)}\r\n\r\n".encode())
        writer.write(body)
        await writer.drain()


@asynccontextmanager
async def fault_proxy(
    upstream: str = "http://localhost:11434",
) -> AsyncIterator[FaultProxy]:
    """Async context manager — yields a started FaultProxy.

    Bound to an ephemeral 127.0.0.1 port. Use ``proxy.url`` for the
    address to point marshal at.
    """
    proxy = FaultProxy(upstream=upstream)
    await proxy._start()
    try:
        yield proxy
    finally:
        await proxy._stop()
