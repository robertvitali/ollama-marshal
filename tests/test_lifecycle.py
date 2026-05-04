from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollama_marshal.lifecycle import ModelLifecycle

OLLAMA_HOST = "http://localhost:11434"


@pytest.fixture
def lifecycle():
    return ModelLifecycle(OLLAMA_HOST)


# ---------------------------------------------------------------------------
# override_keep_alive (static, pure)
# ---------------------------------------------------------------------------


class TestOverrideKeepAlive:
    def test_sets_keep_alive(self):
        body = {"model": "llama3", "prompt": "hi"}
        result = ModelLifecycle.override_keep_alive(body)
        assert result["keep_alive"] == "24h"

    def test_preserves_other_fields(self):
        body = {"model": "llama3", "prompt": "hi", "stream": True}
        result = ModelLifecycle.override_keep_alive(body)
        assert result["model"] == "llama3"
        assert result["prompt"] == "hi"
        assert result["stream"] is True

    def test_overwrites_existing_keep_alive(self):
        body = {"model": "llama3", "keep_alive": "5m"}
        result = ModelLifecycle.override_keep_alive(body)
        assert result["keep_alive"] == "24h"

    def test_does_not_mutate_original(self):
        body = {"model": "llama3"}
        result = ModelLifecycle.override_keep_alive(body)
        assert "keep_alive" not in body
        assert result["keep_alive"] == "24h"

    def test_empty_body(self):
        result = ModelLifecycle.override_keep_alive({})
        assert result == {"keep_alive": "24h"}


# ---------------------------------------------------------------------------
# preload
# ---------------------------------------------------------------------------


class TestPreload:
    async def test_preload_success(self, lifecycle):
        mock_post_response = MagicMock(spec=httpx.Response)
        mock_ps_response = MagicMock(spec=httpx.Response)
        mock_ps_response.json.return_value = {
            "models": [{"name": "llama3:latest", "model": "llama3:latest"}]
        }
        mock_ps_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.get = AsyncMock(return_value=mock_ps_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            result = await lifecycle.preload("llama3:latest")

        assert result is True
        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["model"] == "llama3:latest"
        assert call_kwargs[1]["json"]["prompt"] == ""
        assert call_kwargs[1]["json"]["keep_alive"] == "24h"

    async def test_preload_model_matched_by_model_field(self, lifecycle):
        mock_post_response = MagicMock(spec=httpx.Response)
        mock_ps_response = MagicMock(spec=httpx.Response)
        mock_ps_response.json.return_value = {
            "models": [{"name": "other-name", "model": "llama3:latest"}]
        }
        mock_ps_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.get = AsyncMock(return_value=mock_ps_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            result = await lifecycle.preload("llama3:latest")

        assert result is True

    async def test_preload_timeout_model_never_appears(self, lifecycle):
        mock_post_response = MagicMock(spec=httpx.Response)
        mock_ps_response = MagicMock(spec=httpx.Response)
        mock_ps_response.json.return_value = {"models": []}
        mock_ps_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.get = AsyncMock(return_value=mock_ps_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
            ),
            patch("ollama_marshal.lifecycle._PS_POLL_MAX_WAIT", 2),
            patch("ollama_marshal.lifecycle._PS_POLL_INTERVAL", 1),
            patch("ollama_marshal.lifecycle.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await lifecycle.preload("llama3:latest")

        assert result is False

    async def test_preload_http_error(self, lifecycle):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            result = await lifecycle.preload("llama3:latest")

        assert result is False


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------


class TestUnload:
    async def test_unload_success(self, lifecycle):
        mock_response = MagicMock(spec=httpx.Response)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            result = await lifecycle.unload("llama3:latest")

        assert result is True
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["model"] == "llama3:latest"
        assert call_kwargs[1]["json"]["keep_alive"] == "0"
        assert call_kwargs[1]["json"]["prompt"] == ""

    async def test_unload_http_error(self, lifecycle):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            result = await lifecycle.unload("llama3:latest")

        assert result is False

    async def test_unload_posts_to_correct_url(self, lifecycle):
        mock_response = MagicMock(spec=httpx.Response)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            await lifecycle.unload("mistral:latest")

        url = mock_client.post.call_args[0][0]
        assert url == f"{OLLAMA_HOST}/api/generate"


# ---------------------------------------------------------------------------
# unload_all
# ---------------------------------------------------------------------------


class TestUnloadAll:
    async def test_unload_all_multiple_models(self, lifecycle):
        mock_response = MagicMock(spec=httpx.Response)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        models = ["llama3:latest", "mistral:latest", "codellama:latest"]

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            await lifecycle.unload_all(models)

        assert mock_client.post.await_count == 3

    async def test_unload_all_empty_list(self, lifecycle):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            await lifecycle.unload_all([])

        mock_client.post.assert_not_awaited()

    async def test_unload_all_continues_after_failure(self, lifecycle):
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.HTTPError("fail on second")
            return MagicMock(spec=httpx.Response)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=side_effect)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "ollama_marshal.lifecycle.httpx.AsyncClient", return_value=mock_client
        ):
            await lifecycle.unload_all(["a", "b", "c"])

        # All three models attempted even though second fails
        assert call_count == 3


# ---------------------------------------------------------------------------
# _wait_for_model
# ---------------------------------------------------------------------------


class TestWaitForModel:
    async def test_model_found_immediately(self, lifecycle):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "models": [{"name": "llama3:latest", "model": "llama3:latest"}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await lifecycle._wait_for_model(
            mock_client, "llama3:latest", "http://localhost:11434"
        )
        assert result is True

    async def test_model_found_after_retries(self, lifecycle):
        empty_response = MagicMock(spec=httpx.Response)
        empty_response.json.return_value = {"models": []}
        empty_response.raise_for_status = MagicMock()

        found_response = MagicMock(spec=httpx.Response)
        found_response.json.return_value = {
            "models": [{"name": "llama3:latest", "model": "llama3:latest"}]
        }
        found_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=[empty_response, empty_response, found_response]
        )

        with patch("ollama_marshal.lifecycle.asyncio.sleep", new_callable=AsyncMock):
            result = await lifecycle._wait_for_model(
                mock_client, "llama3:latest", "http://localhost:11434"
            )

        assert result is True
        assert mock_client.get.await_count == 3

    async def test_model_not_found_timeout(self, lifecycle):
        empty_response = MagicMock(spec=httpx.Response)
        empty_response.json.return_value = {"models": []}
        empty_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=empty_response)

        with (
            patch("ollama_marshal.lifecycle._PS_POLL_MAX_WAIT", 3),
            patch("ollama_marshal.lifecycle._PS_POLL_INTERVAL", 1),
            patch("ollama_marshal.lifecycle.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await lifecycle._wait_for_model(
                mock_client, "llama3:latest", "http://localhost:11434"
            )

        assert result is False

    async def test_wait_survives_http_error_during_poll(self, lifecycle):
        error_response = httpx.HTTPError("poll failed")
        found_response = MagicMock(spec=httpx.Response)
        found_response.json.return_value = {
            "models": [{"name": "llama3:latest", "model": "llama3:latest"}]
        }
        found_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=[error_response, found_response])

        with patch("ollama_marshal.lifecycle.asyncio.sleep", new_callable=AsyncMock):
            result = await lifecycle._wait_for_model(
                mock_client, "llama3:latest", "http://localhost:11434"
            )

        assert result is True

    async def test_wait_matches_by_model_field(self, lifecycle):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "models": [{"name": "some-alias", "model": "llama3:latest"}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await lifecycle._wait_for_model(
            mock_client, "llama3:latest", "http://localhost:11434"
        )
        assert result is True

    async def test_wait_no_models_key_in_response(self, lifecycle):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("ollama_marshal.lifecycle._PS_POLL_MAX_WAIT", 1),
            patch("ollama_marshal.lifecycle._PS_POLL_INTERVAL", 1),
            patch("ollama_marshal.lifecycle.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await lifecycle._wait_for_model(
                mock_client, "llama3:latest", "http://localhost:11434"
            )

        assert result is False


# ---------------------------------------------------------------------------
# ensure_loaded
# ---------------------------------------------------------------------------


class TestEnsureLoaded:
    async def test_already_loaded_skips_preload(self, lifecycle):
        with patch.object(lifecycle, "preload", new_callable=AsyncMock) as mock_preload:
            result = await lifecycle.ensure_loaded("llama3:latest", {"llama3:latest"})

        assert result is True
        mock_preload.assert_not_awaited()

    async def test_not_loaded_triggers_preload(self, lifecycle):
        with patch.object(
            lifecycle, "preload", new_callable=AsyncMock, return_value=True
        ) as mock_preload:
            result = await lifecycle.ensure_loaded("llama3:latest", set())

        assert result is True
        mock_preload.assert_awaited_once_with(
            "llama3:latest",
            num_ctx=None,
            instance_url=None,
            load_timeout_s=None,
        )

    async def test_not_loaded_preload_fails(self, lifecycle):
        with patch.object(
            lifecycle, "preload", new_callable=AsyncMock, return_value=False
        ):
            result = await lifecycle.ensure_loaded("llama3:latest", set())

        assert result is False

    async def test_different_model_not_in_set(self, lifecycle):
        with patch.object(
            lifecycle, "preload", new_callable=AsyncMock, return_value=True
        ) as mock_preload:
            result = await lifecycle.ensure_loaded("mistral:latest", {"llama3:latest"})

        assert result is True
        mock_preload.assert_awaited_once_with(
            "mistral:latest",
            num_ctx=None,
            instance_url=None,
            load_timeout_s=None,
        )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestModelLifecycleInit:
    def test_default_host(self):
        lc = ModelLifecycle()
        assert lc._ollama_host == "http://localhost:11434"

    def test_custom_host(self):
        lc = ModelLifecycle("http://remote:8080")
        assert lc._ollama_host == "http://remote:8080"
