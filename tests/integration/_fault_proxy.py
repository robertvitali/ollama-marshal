"""Fault-injection HTTP proxy in front of Ollama for integration tests.

Sits between marshal and the real Ollama, transparently forwarding
requests by default but with hooks to selectively fail/delay/disconnect
or return canned responses. Required for tests that need controllable
Ollama failures (Surface A retry, failed-preload sentinel, Surface C2
unexpected-unload via fake /api/ps) — all of which can't be reproduced
reliably by toggling the real daemon.

Implementation: bare ``asyncio.start_server`` HTTP/1.1 parser, with an
``httpx.AsyncClient`` for the upstream forward leg. No aiohttp
dependency. Bound to ``127.0.0.1`` (IPv4-only) on an ephemeral port.

Usage::

    from tests.integration._fault_proxy import fault_proxy

    async with fault_proxy() as proxy:
        proxy.fail_next("/api/generate", times=2, status=503)
        # marshal_config.ollama.host = proxy.url
        ...

Hooks (all match on the canonical request path — query string and
trailing slash stripped — with EXACT equality, not prefix):
- ``fail_next(path, times, status)`` — return ``status`` ``times``
  times for ``path``, then resume pass-through.
- ``disconnect_next(path, times)`` — close the connection without
  sending headers (surfaces on the client side as
  ``httpx.RemoteProtocolError`` after a clean TCP FIN).
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
import re
import socket
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

# Bytes returned to the client for status-only fault responses.
_REASON_PHRASES = {
    200: b"OK",
    400: b"Bad Request",
    411: b"Length Required",
    413: b"Payload Too Large",
    500: b"Internal Server Error",
    502: b"Bad Gateway",
    503: b"Service Unavailable",
    504: b"Gateway Timeout",
}

# Strict integer regex for Content-Length. Rejects signed forms
# (``+5``, ``-1``), hex (``0x0a``), whitespace inside the value, and
# any non-digit characters. The HTTP spec (RFC 7230 §3.3.2) requires
# Content-Length be a sequence of OCTECT digits with no other forms.
_CONTENT_LENGTH_RE = re.compile(r"^[0-9]+$")


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
        self._client: httpx.AsyncClient | None = None
        # Track in-flight request handler tasks so ``_stop`` can drain
        # them. Without this, a handler still inside ``_forward`` when
        # the test exits will continue running, try to use the
        # already-closed httpx client, and raise
        # ``RuntimeError("client has been closed.")`` — surfacing as
        # "Task exception was never retrieved" warnings in pytest.
        self._handler_tasks: set[asyncio.Task[None]] = set()

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
        key = self._normalize_path(path)
        for _ in range(times):
            self._queues[key].append(_FailSpec(status=status))

    def disconnect_next(self, path: str, times: int = 1) -> None:
        """Queue ``times`` socket-close-without-response events for ``path``."""
        key = self._normalize_path(path)
        for _ in range(times):
            self._queues[key].append(_DisconnectSpec())

    def delay_next(self, path: str, seconds: float) -> None:
        """Sleep ``seconds`` before forwarding the next request to ``path``."""
        self._queues[self._normalize_path(path)].append(_DelaySpec(seconds=seconds))

    def fake_response(
        self, path: str, body: dict[str, Any] | bytes, times: int | None = None
    ) -> None:
        """Return ``body`` (dict → JSON, or raw bytes) for ``path``.

        ``times=None`` means indefinitely.
        """
        payload = json.dumps(body).encode() if isinstance(body, dict) else body
        self._queues[self._normalize_path(path)].append(
            _FakeSpec(body=payload, remaining=times)
        )

    # -- lifecycle ----------------------------------------------------

    async def _start(self) -> None:
        """Bind to an ephemeral port and start accepting connections.

        Atomic: ``start_server(port=0)`` lets the OS pick a free port,
        then we read it back from the bound socket. No bind/close/rebind
        race window where another process could steal the port.

        ``family=socket.AF_INET`` forces IPv4-only — without it,
        macOS may bind both IPv4 and IPv6 and ``sockets[0]`` could
        report either, while ``proxy.url`` always uses ``127.0.0.1``.
        Pinning to AF_INET removes the platform-dependent ambiguity.
        """
        self._server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", 0, family=socket.AF_INET
        )
        # `sockets[0]` always exists for a listening server.
        self._port = self._server.sockets[0].getsockname()[1]
        # Single shared upstream client — pooled connections, much
        # faster than opening a new TCP session per forwarded request
        # (matters for retry tests that fire 99 sequential requests).
        self._client = httpx.AsyncClient(timeout=300)

    async def _stop(self) -> None:
        # Stop accepting new connections.
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        # Drain in-flight handler tasks BEFORE closing the upstream
        # client. If a handler is mid-``_forward`` when the client
        # closes, its next ``self._client.stream`` call raises
        # ``RuntimeError("client has been closed.")`` — catch via
        # cancellation instead. 5s grace before forcible cancel; that's
        # plenty for any in-flight request to either finish naturally
        # or surface its own error path.
        if self._handler_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._handler_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except TimeoutError:
                for task in self._handler_tasks:
                    task.cancel()
                # Final gather absorbs CancelledError so the test
                # doesn't see "Task exception was never retrieved".
                await asyncio.gather(*self._handler_tasks, return_exceptions=True)
            self._handler_tasks.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- request handling --------------------------------------------

    @staticmethod
    def _normalize_path(raw: str) -> str:
        """Strip query string and trailing slash for canonical matching."""
        path = raw.split("?", 1)[0]
        return path.rstrip("/") or "/"

    def _next_spec(self, request_path: str) -> _Spec | None:
        """Pop the next queued spec matching ``request_path``.

        EXACT match on the canonical path (query string and trailing
        slash stripped). Originally used ``startswith`` but that
        over-matched: a queued ``/api/chat`` fault would also fire on
        ``/api/chatty`` or any future longer endpoint sharing that
        prefix. Exact match is unambiguous and matches every queued
        path in the current suite.
        """
        canonical = self._normalize_path(request_path)
        queue = self._queues.get(canonical)
        if not queue:
            return None
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

    # Cap on request body size. 16 MiB is plenty for /api/chat
    # message arrays even at huge num_ctx; rejects malformed
    # Content-Length values that would otherwise OOM-allocate.
    _MAX_BODY_BYTES = 16 * 1024 * 1024

    # Read timeout per phase. A partial request (client disconnects
    # mid-headers, or stops sending bytes mid-body) would otherwise
    # leave the handler task alive past test teardown — the test's
    # own asyncio loop logs "Task exception was never retrieved"
    # noise. Cap each read at 30s so the handler unwinds cleanly.
    _READ_TIMEOUT_S = 30.0

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Per-connection handler. One request → one response → close.

        Catches malformed-request exceptions silently to keep test
        output clean (asyncio's "Task exception was never retrieved"
        noise from `IncompleteReadError`/`UnicodeDecodeError`/
        `ValueError` would otherwise pollute every test run with a
        slightly-misbehaving client).

        Self-registers with ``self._handler_tasks`` so ``_stop`` can
        drain in-flight handlers before closing the upstream client.
        """
        task = asyncio.current_task()
        if task is not None:
            self._handler_tasks.add(task)
        try:
            try:
                request_line = await asyncio.wait_for(
                    reader.readline(), timeout=self._READ_TIMEOUT_S
                )
            except TimeoutError:
                return
            if not request_line:
                return
            try:
                method, path, _version = request_line.decode().split(" ", 2)
            except (ValueError, UnicodeDecodeError):
                return

            # Read headers until empty line. Track Content-Length
            # occurrences separately so we can reject duplicates
            # (HTTP request smuggling vector — RFC 7230 §3.3.3).
            headers: dict[str, str] = {}
            content_length_count = 0
            while True:
                try:
                    line = await asyncio.wait_for(
                        reader.readline(), timeout=self._READ_TIMEOUT_S
                    )
                except TimeoutError:
                    return
                if not line or line in (b"\r\n", b"\n"):
                    break
                try:
                    key, _, val = line.decode().partition(":")
                except UnicodeDecodeError:
                    return
                key_lc = key.strip().lower()
                if key_lc == "content-length":
                    content_length_count += 1
                headers[key_lc] = val.strip()

            # Reject Transfer-Encoding: chunked. The bare HTTP/1.1 parser
            # does not implement chunked decoding on the inbound side;
            # silently treating chunked bodies as Content-Length=0 (the
            # old behavior) leaks the chunked body bytes onto the next
            # read of the stream and forwards a different request than
            # the client sent. 411 Length Required is the canonical
            # rejection for this case.
            if headers.get("transfer-encoding"):
                await self._write_error(writer, 411)
                return

            # Reject duplicate Content-Length headers — request smuggling
            # vector. Different downstream parsers may pick the first vs
            # last header value and route the body differently. The
            # spec (RFC 7230 §3.3.3 rule 4) says reject with 400.
            if content_length_count > 1:
                await self._write_error(writer, 400)
                return

            # Strict Content-Length parsing. Reject signed values
            # (``+5``, ``-1``), hex, whitespace inside the value, and
            # any non-digit characters per RFC 7230 §3.3.2.
            cl_raw = headers.get("content-length", "").strip()
            if cl_raw and not _CONTENT_LENGTH_RE.match(cl_raw):
                await self._write_error(writer, 400)
                return
            content_length = int(cl_raw) if cl_raw else 0
            if content_length > self._MAX_BODY_BYTES:
                await self._write_error(writer, 413)
                return
            try:
                body = (
                    await asyncio.wait_for(
                        reader.readexactly(content_length),
                        timeout=self._READ_TIMEOUT_S,
                    )
                    if content_length
                    else b""
                )
            except (TimeoutError, asyncio.IncompleteReadError):
                return

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
        except (ConnectionError, BrokenPipeError):
            # Client closed mid-handler. Nothing actionable; stay quiet.
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
            if task is not None:
                self._handler_tasks.discard(task)

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
        # Drop hop-by-hop headers + host (httpx sets its own). The
        # full hop-by-hop list per RFC 7230 §6.1 — leaking these from
        # client to upstream can break keep-alive negotiation.
        hop_by_hop = {
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
            "upgrade",
            "proxy-connection",
            "te",
            "keep-alive",
            "trailer",
        }
        forward_headers = {k: v for k, v in headers.items() if k not in hop_by_hop}
        # Track whether we've started writing the response to the
        # client. If the upstream fails AFTER we've already sent the
        # status line + headers + partial body, writing a fresh 502 on
        # top would produce malformed wire bytes the client would
        # misparse. Instead we just close the writer in that case —
        # close-delimited framing means the client correctly sees a
        # truncated body and surfaces the right error class.
        started_response = False
        try:
            assert self._client is not None, "fault proxy not started"
            async with self._client.stream(
                method, url, headers=forward_headers, content=body
            ) as resp:
                # Write status line + headers (close-delimited body).
                # Use the upstream's actual reason phrase rather than
                # a stock "OK" — a forwarded 503 should read
                # "HTTP/1.1 503 Service Unavailable", not
                # "HTTP/1.1 503 OK". Some lenient parsers tolerate
                # the wrong reason but it can mask retry-test bugs.
                reason = (
                    resp.reason_phrase
                    or _REASON_PHRASES.get(resp.status_code, b"OK").decode()
                )
                writer.write(f"HTTP/1.1 {resp.status_code} {reason}\r\n".encode())
                writer.write(b"Connection: close\r\n")
                # Forward content-type so client parses correctly.
                if "content-type" in resp.headers:
                    writer.write(
                        f"Content-Type: {resp.headers['content-type']}\r\n".encode()
                    )
                writer.write(b"\r\n")
                started_response = True
                async for chunk in resp.aiter_raw():
                    writer.write(chunk)
                    await writer.drain()
        except httpx.HTTPError:
            if started_response:
                # Already streaming to client — closing the writer
                # produces a clean truncation, vs writing 502 which
                # would corrupt the response framing.
                return
            # Upstream failure before any bytes hit the wire — safe to
            # surface as 502 to the test's marshal.
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
        reason = _REASON_PHRASES.get(status, b"OK")
        writer.write(f"HTTP/1.1 {status} ".encode() + reason + b"\r\n")
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
