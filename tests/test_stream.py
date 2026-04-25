from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollama_marshal.stream import (
    _forward_response,
    _stream_response,
    forward_passthrough,
    forward_request,
)

OLLAMA_HOST = "http://localhost:11434"


def _make_mock_client(response=None):
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    if response is not None:
        mock_client.post = AsyncMock(return_value=response)
        mock_client.request = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# forward_request — non-streaming
# ---------------------------------------------------------------------------


class TestForwardRequestNonStreaming:
    async def test_returns_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            result = await forward_request(
                OLLAMA_HOST,
                "/api/chat",
                {"model": "llama3", "prompt": "hi"},
                stream=False,
            )

        assert result is mock_response

    async def test_applies_keep_alive_override(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await forward_request(
                OLLAMA_HOST, "/api/chat", {"model": "llama3"}, stream=False
            )

        sent_body = mock_client.post.call_args[1]["json"]
        assert sent_body["keep_alive"] == "24h"
        assert sent_body["model"] == "llama3"

    async def test_posts_to_correct_url(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await forward_request(
                OLLAMA_HOST, "/api/generate", {"model": "m"}, stream=False
            )

        url = mock_client.post.call_args[0][0]
        assert url == f"{OLLAMA_HOST}/api/generate"


# ---------------------------------------------------------------------------
# forward_request — streaming
# ---------------------------------------------------------------------------


class TestForwardRequestStreaming:
    async def test_returns_async_iterator(self):
        body = {"model": "llama3", "prompt": "hi"}

        with patch("ollama_marshal.stream.httpx.AsyncClient"):
            result = forward_request(OLLAMA_HOST, "/api/chat", body, stream=True)
            # When stream=True, forward_request returns the generator directly
            # (it's not awaited — _stream_response is an async generator)
            actual = await result
            # The result should be an async iterator (generator)
            assert hasattr(actual, "__aiter__")


# ---------------------------------------------------------------------------
# _forward_response
# ---------------------------------------------------------------------------


class TestForwardResponse:
    async def test_returns_httpx_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            result = await _forward_response(
                OLLAMA_HOST, "/api/chat", {"model": "llama3"}
            )

        assert result is mock_response

    async def test_sends_json_body(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)
        body = {"model": "llama3", "messages": [], "keep_alive": "24h"}

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await _forward_response(OLLAMA_HOST, "/api/chat", body)

        assert mock_client.post.call_args[1]["json"] == body

    async def test_timeout_is_set(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await _forward_response(OLLAMA_HOST, "/api/chat", {"model": "m"})

        assert mock_client.post.call_args[1]["timeout"] == 300


# ---------------------------------------------------------------------------
# _stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    async def test_yields_chunks(self):
        chunks = [b'{"done":false}\n', b'{"done":true}\n']

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()

        async def aiter_bytes():
            for c in chunks:
                yield c

        mock_resp.aiter_bytes = aiter_bytes
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.stream = MagicMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        collected = []
        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            async for chunk in _stream_response(
                OLLAMA_HOST, "/api/chat", {"model": "m"}
            ):
                collected.append(chunk)

        assert collected == chunks

    async def test_raises_on_bad_status(self):
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.stream = MagicMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            async for _ in _stream_response(OLLAMA_HOST, "/api/chat", {"model": "m"}):
                pass


# ---------------------------------------------------------------------------
# forward_passthrough
# ---------------------------------------------------------------------------


class TestForwardPassthrough:
    async def test_get_request(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            result = await forward_passthrough(OLLAMA_HOST, "GET", "/api/tags")

        assert result is mock_response
        call_kwargs = mock_client.request.call_args[1]
        assert call_kwargs["method"] == "GET"
        assert call_kwargs["url"] == f"{OLLAMA_HOST}/api/tags"
        assert call_kwargs["content"] is None
        assert call_kwargs["headers"] is None

    async def test_post_request_with_body(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)
        body_bytes = b'{"name": "llama3"}'

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            result = await forward_passthrough(
                OLLAMA_HOST, "POST", "/api/show", body=body_bytes
            )

        assert result is mock_response
        call_kwargs = mock_client.request.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["content"] == body_bytes

    async def test_post_with_headers(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)
        custom_headers = {"X-Custom": "value", "Content-Type": "application/json"}

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await forward_passthrough(
                OLLAMA_HOST,
                "POST",
                "/api/pull",
                body=b'{"name":"m"}',
                headers=custom_headers,
            )

        call_kwargs = mock_client.request.call_args[1]
        assert call_kwargs["headers"] == custom_headers

    async def test_no_body_no_headers(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await forward_passthrough(OLLAMA_HOST, "DELETE", "/api/delete")

        call_kwargs = mock_client.request.call_args[1]
        assert call_kwargs["content"] is None
        assert call_kwargs["headers"] is None

    async def test_timeout_is_set(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await forward_passthrough(OLLAMA_HOST, "GET", "/api/ps")

        call_kwargs = mock_client.request.call_args[1]
        assert call_kwargs["timeout"] == 300

    async def test_constructs_correct_url(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_client = _make_mock_client(mock_response)
        host = "http://remote:9999"

        with patch("ollama_marshal.stream.httpx.AsyncClient", return_value=mock_client):
            await forward_passthrough(host, "GET", "/api/version")

        call_kwargs = mock_client.request.call_args[1]
        assert call_kwargs["url"] == "http://remote:9999/api/version"
