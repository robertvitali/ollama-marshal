"""Streaming response proxy for Ollama's NDJSON format."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from ollama_marshal.lifecycle import ModelLifecycle

logger = structlog.get_logger()


# Default Hop 2 forward timeout used when call sites don't pass an
# explicit ``timeout_s`` (legacy unit-test paths). Production paths
# always thread the resolved value (header → config) through the
# scheduler — see ``Scheduler._forward_single_inner`` and
# ``RequestEnvelope.ollama_forward_timeout_s``.
_DEFAULT_FORWARD_TIMEOUT_S = 3600


async def forward_request(
    ollama_host: str,
    endpoint: str,
    request_body: dict[str, Any],
    stream: bool = False,
    timeout_s: int = _DEFAULT_FORWARD_TIMEOUT_S,
) -> httpx.Response | AsyncIterator[bytes]:
    """Forward a request to Ollama, handling both streaming and non-streaming.

    For non-streaming requests, returns the full response. For streaming
    requests, returns an async iterator of response chunks.

    Args:
        ollama_host: Ollama API base URL.
        endpoint: The API endpoint path (e.g., '/api/chat').
        request_body: The request body to forward.
        stream: Whether to stream the response.
        timeout_s: Wall-clock budget for the Ollama call in seconds.
            Default 3600 matches the config default; production call
            sites pass the per-envelope value resolved from the
            ``X-Request-Timeout`` header or
            ``scheduler.ollama_forward_timeout_s``.

    Returns:
        httpx.Response for non-streaming, AsyncIterator[bytes] for streaming.
    """
    body = ModelLifecycle.override_keep_alive(request_body)

    if stream:
        return _stream_response(ollama_host, endpoint, body, timeout_s=timeout_s)
    return await _forward_response(ollama_host, endpoint, body, timeout_s=timeout_s)


async def _forward_response(
    ollama_host: str,
    endpoint: str,
    body: dict[str, Any],
    timeout_s: int = _DEFAULT_FORWARD_TIMEOUT_S,
) -> httpx.Response:
    """Forward a non-streaming request and return the full response.

    Args:
        ollama_host: Ollama API base URL.
        endpoint: The API endpoint path.
        body: The request body.
        timeout_s: Wall-clock budget for the Ollama call in seconds.

    Returns:
        The full httpx.Response from Ollama.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ollama_host}{endpoint}",
            json=body,
            timeout=timeout_s,
        )
        return resp


async def _stream_response(
    ollama_host: str,
    endpoint: str,
    body: dict[str, Any],
    timeout_s: int = _DEFAULT_FORWARD_TIMEOUT_S,
) -> AsyncIterator[bytes]:
    """Stream a response from Ollama as an async byte iterator.

    Ollama streams NDJSON — one JSON object per line. This function
    yields raw bytes as they arrive from Ollama.

    Args:
        ollama_host: Ollama API base URL.
        endpoint: The API endpoint path.
        body: The request body.
        timeout_s: Read-side budget for the streaming call. Connect/
            write/pool stay short (10s) so a dead Ollama fails fast
            even when ``timeout_s`` is hours.

    Yields:
        Raw byte chunks from Ollama's streaming response.
    """
    async with (
        httpx.AsyncClient() as client,
        client.stream(
            "POST",
            f"{ollama_host}{endpoint}",
            json=body,
            timeout=httpx.Timeout(connect=10, read=timeout_s, write=10, pool=10),
        ) as resp,
    ):
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk


async def forward_passthrough(
    ollama_host: str,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: int = _DEFAULT_FORWARD_TIMEOUT_S,
) -> httpx.Response:
    """Pass a request through to Ollama without any modification.

    Used for non-inference endpoints that don't need scheduling.

    Args:
        ollama_host: Ollama API base URL.
        method: HTTP method (GET, POST, etc.).
        path: The full request path.
        body: Raw request body bytes, if any.
        headers: Additional headers to forward.
        timeout_s: Wall-clock budget for the passthrough call in
            seconds. Defaults to the same Hop 2 default
            (``_DEFAULT_FORWARD_TIMEOUT_S``).

    Returns:
        The full httpx.Response from Ollama.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=method,
            url=f"{ollama_host}{path}",
            content=body,
            headers=headers,
            timeout=timeout_s,
        )
        return resp
