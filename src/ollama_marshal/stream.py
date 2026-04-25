"""Streaming response proxy for Ollama's NDJSON format."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from ollama_marshal.lifecycle import ModelLifecycle

logger = structlog.get_logger()


async def forward_request(
    ollama_host: str,
    endpoint: str,
    request_body: dict[str, Any],
    stream: bool = False,
) -> httpx.Response | AsyncIterator[bytes]:
    """Forward a request to Ollama, handling both streaming and non-streaming.

    For non-streaming requests, returns the full response. For streaming
    requests, returns an async iterator of response chunks.

    Args:
        ollama_host: Ollama API base URL.
        endpoint: The API endpoint path (e.g., '/api/chat').
        request_body: The request body to forward.
        stream: Whether to stream the response.

    Returns:
        httpx.Response for non-streaming, AsyncIterator[bytes] for streaming.
    """
    body = ModelLifecycle.override_keep_alive(request_body)

    if stream:
        return _stream_response(ollama_host, endpoint, body)
    return await _forward_response(ollama_host, endpoint, body)


async def _forward_response(
    ollama_host: str,
    endpoint: str,
    body: dict[str, Any],
) -> httpx.Response:
    """Forward a non-streaming request and return the full response.

    Args:
        ollama_host: Ollama API base URL.
        endpoint: The API endpoint path.
        body: The request body.

    Returns:
        The full httpx.Response from Ollama.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ollama_host}{endpoint}",
            json=body,
            timeout=300,
        )
        return resp


async def _stream_response(
    ollama_host: str,
    endpoint: str,
    body: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Stream a response from Ollama as an async byte iterator.

    Ollama streams NDJSON — one JSON object per line. This function
    yields raw bytes as they arrive from Ollama.

    Args:
        ollama_host: Ollama API base URL.
        endpoint: The API endpoint path.
        body: The request body.

    Yields:
        Raw byte chunks from Ollama's streaming response.
    """
    async with (
        httpx.AsyncClient() as client,
        client.stream(
            "POST",
            f"{ollama_host}{endpoint}",
            json=body,
            timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10),
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
) -> httpx.Response:
    """Pass a request through to Ollama without any modification.

    Used for non-inference endpoints that don't need scheduling.

    Args:
        ollama_host: Ollama API base URL.
        method: HTTP method (GET, POST, etc.).
        path: The full request path.
        body: Raw request body bytes, if any.
        headers: Additional headers to forward.

    Returns:
        The full httpx.Response from Ollama.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=method,
            url=f"{ollama_host}{path}",
            content=body,
            headers=headers,
            timeout=300,
        )
        return resp
