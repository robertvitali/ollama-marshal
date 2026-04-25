from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from starlette.routing import Route

import ollama_marshal.server as server_mod
from ollama_marshal.config import MarshalConfig, ShutdownConfig, ShutdownMode
from ollama_marshal.memory import LoadedModel, MemoryBudget
from ollama_marshal.queue import ModelQueues
from ollama_marshal.server import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_memory():
    mem = MagicMock()
    mem.get_loaded_models.return_value = {
        "llama3:latest": LoadedModel(name="llama3:latest", size_vram=4_000_000_000),
    }
    mem.available_vram.return_value = 50 * 1024**3
    mem.used_vram.return_value = 4_000_000_000
    mem.budget = MemoryBudget(
        total_ram=64 * 1024**3,
        os_overhead=4 * 1024**3,
        safety_margin=2 * 1024**3,
    )
    return mem


def _mock_queues():
    q = MagicMock(spec=ModelQueues)
    q.pending_by_model = AsyncMock(return_value={"llama3:latest": 2})
    q.total_pending = AsyncMock(return_value=2)
    q.enqueue = AsyncMock()
    return q


def _mock_scheduler():
    sched = MagicMock()
    sched.metrics = MagicMock(
        requests_served=10,
        model_swaps=3,
        evictions=1,
        average_wait_ms=42.5,
    )
    return sched


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_returns_fastapi_instance(self):
        config = MarshalConfig()
        app = create_app(config)
        assert isinstance(app, FastAPI)

    def test_app_title_and_version(self):
        config = MarshalConfig()
        app = create_app(config)
        assert app.title == "ollama-marshal"
        assert app.version == "0.1.0"

    def test_stores_config_in_state(self):
        config = MarshalConfig()
        app = create_app(config)
        assert app.state.config is config

    def test_creates_with_default_config(self):
        with patch("ollama_marshal.server.load_config") as mock_load:
            mock_load.return_value = MarshalConfig()
            app = create_app(None)
            assert isinstance(app, FastAPI)
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


class TestRouteRegistration:
    def _get_routes(self):
        config = MarshalConfig()
        app = create_app(config)
        return {route.path for route in app.routes if hasattr(route, "path")}

    def test_api_chat_route_exists(self):
        assert "/api/chat" in self._get_routes()

    def test_api_generate_route_exists(self):
        assert "/api/generate" in self._get_routes()

    def test_api_embeddings_route_exists(self):
        assert "/api/embeddings" in self._get_routes()

    def test_openai_chat_completions_route_exists(self):
        assert "/v1/chat/completions" in self._get_routes()

    def test_openai_completions_route_exists(self):
        assert "/v1/completions" in self._get_routes()

    def test_openai_embeddings_route_exists(self):
        assert "/v1/embeddings" in self._get_routes()

    def test_marshal_status_route_exists(self):
        assert "/api/marshal/status" in self._get_routes()

    def test_passthrough_route_exists(self):
        assert "/api/{path:path}" in self._get_routes()


# ---------------------------------------------------------------------------
# _enqueue_inference
# ---------------------------------------------------------------------------


class TestEnqueueInference:
    async def test_creates_envelope_and_enqueues(self):
        queues = ModelQueues()
        original_module_queues = getattr(server_mod, "_queues", None)
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {"x-program-id": "test-prog"}

        body = {"model": "llama3:latest", "stream": False}

        # Complete the envelope from a separate task
        async def complete_after_enqueue():
            # Wait for the envelope to be enqueued
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:
                all_envs[0].complete({"response": "ok"})

        task = asyncio.create_task(complete_after_enqueue())

        try:
            result = await server_mod._enqueue_inference(request, body, "/api/chat")
            assert result is not None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if original_module_queues is not None:
                server_mod._queues = original_module_queues

    async def test_returns_502_on_error(self):
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {}

        body = {"model": "llama3:latest"}

        async def fail_after_enqueue():
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:
                all_envs[0].fail(RuntimeError("ollama down"))

        task = asyncio.create_task(fail_after_enqueue())

        try:
            result = await server_mod._enqueue_inference(request, body, "/api/chat")
            # Should be a 502 JSONResponse
            assert result.status_code == 502
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_streaming_response(self):
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {}

        body = {"model": "llama3:latest", "stream": True}

        async def complete_stream():
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:
                # Simulate a streaming async iterator response
                async def fake_stream():
                    yield b'{"data": "chunk"}'

                all_envs[0].complete(fake_stream())

        task = asyncio.create_task(complete_stream())

        try:
            result = await server_mod._enqueue_inference(request, body, "/api/chat")
            from fastapi.responses import StreamingResponse

            assert isinstance(result, StreamingResponse)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_non_streaming_httpx_response(self):
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {}

        body = {"model": "llama3:latest", "stream": False}

        async def complete_httpx():
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = b'{"done": true}'
                mock_resp.headers = {"content-type": "application/json"}
                all_envs[0].complete(mock_resp)

        task = asyncio.create_task(complete_httpx())

        try:
            result = await server_mod._enqueue_inference(request, body, "/api/chat")
            from fastapi.responses import Response

            assert isinstance(result, Response)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# _enqueue_and_wait
# ---------------------------------------------------------------------------


class TestEnqueueAndWait:
    async def test_returns_json_dict(self):
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {"x-program-id": "test"}

        async def complete_json():
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"message": {"content": "hi"}}
                # Must not have __aiter__ so it falls through to .json()
                del mock_resp.__aiter__
                all_envs[0].complete(mock_resp)

        task = asyncio.create_task(complete_json())

        try:
            result = await server_mod._enqueue_and_wait(
                request,
                "llama3:latest",
                {"model": "llama3:latest", "messages": []},
                "/api/chat",
                stream=False,
            )
            assert isinstance(result, dict)
            assert result["message"]["content"] == "hi"
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_returns_502_on_error(self):
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {}

        async def fail_it():
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:
                all_envs[0].fail(RuntimeError("boom"))

        task = asyncio.create_task(fail_it())

        try:
            result = await server_mod._enqueue_and_wait(
                request,
                "llama3:latest",
                {"model": "llama3:latest"},
                "/api/chat",
                stream=False,
            )
            from fastapi.responses import JSONResponse

            assert isinstance(result, JSONResponse)
            assert result.status_code == 502
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_streaming_response(self):
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {}

        async def complete_stream():
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:

                async def fake_stream():
                    yield b"data: chunk\n"

                all_envs[0].complete(fake_stream())

        task = asyncio.create_task(complete_stream())

        try:
            result = await server_mod._enqueue_and_wait(
                request,
                "llama3:latest",
                {"model": "llama3:latest"},
                "/api/chat",
                stream=True,
            )
            from fastapi.responses import StreamingResponse

            assert isinstance(result, StreamingResponse)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_returns_raw_response_without_json(self):
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {}

        async def complete_raw():
            for _ in range(50):
                pending = await queues.total_pending()
                if pending > 0:
                    break
                await asyncio.sleep(0.01)
            all_envs = await queues.get_all_sorted_by_arrival()
            if all_envs:
                # A plain dict response (no .json, no __aiter__)
                all_envs[0].complete({"raw": "data"})

        task = asyncio.create_task(complete_raw())

        try:
            result = await server_mod._enqueue_and_wait(
                request,
                "llama3:latest",
                {"model": "llama3:latest"},
                "/api/chat",
                stream=False,
            )
            assert result == {"raw": "data"}
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# marshal_status endpoint (via module globals)
# ---------------------------------------------------------------------------


class TestMarshalStatus:
    async def test_status_endpoint_structure(self):
        config = MarshalConfig()
        app = create_app(config)

        # Set module-level globals
        server_mod._queues = _mock_queues()
        server_mod._memory = _mock_memory()
        server_mod._scheduler = _mock_scheduler()
        server_mod._started_at = time.monotonic() - 60

        # Find the status route handler
        from starlette.routing import Route

        status_handler = None
        for route in app.routes:
            if isinstance(route, Route) and route.path == "/api/marshal/status":
                status_handler = route.endpoint
                break

        assert status_handler is not None

        result = await status_handler()

        assert "uptime_seconds" in result
        assert "loaded_models" in result
        assert "memory" in result
        assert "queue" in result
        assert "metrics" in result

        # Verify nested structure
        assert len(result["loaded_models"]) == 1
        assert result["loaded_models"][0]["name"] == "llama3:latest"
        assert result["metrics"]["requests_served"] == 10
        assert result["queue"]["total_pending"] == 2


# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------


def _make_mock_components():
    memory = MagicMock()
    memory.start_polling = AsyncMock()
    memory.stop_polling = AsyncMock()
    memory.get_loaded_models.return_value = {}

    registry = MagicMock()
    registry.initialize = AsyncMock()
    registry.benchmark_unknown = AsyncMock()

    lifecycle = MagicMock()
    lifecycle.unload_all = AsyncMock()

    scheduler = MagicMock()
    scheduler.start = AsyncMock()
    scheduler.stop = AsyncMock()

    queues = MagicMock()
    queues.total_pending = AsyncMock(return_value=0)

    return memory, registry, lifecycle, scheduler, queues


class TestLifespan:
    async def test_lifespan_starts_and_stops_components(self):
        config = MarshalConfig()
        app = create_app(config)

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch(
                "ollama_marshal.server.ModelLifecycle",
                return_value=lifecycle,
            ),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with server_mod.lifespan(app):
                memory.start_polling.assert_called_once()
                registry.initialize.assert_called_once()
                scheduler.start.assert_called_once()

            scheduler.stop.assert_called_once()
            memory.stop_polling.assert_called_once()

    async def test_lifespan_drain_mode(self):
        config = MarshalConfig(
            shutdown=ShutdownConfig(
                mode=ShutdownMode.DRAIN,
                drain_timeout=1,
                unload_models=False,
            )
        )
        app = create_app(config)

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        # Simulate pending requests that drain quickly
        queues.total_pending = AsyncMock(side_effect=[2, 1, 0])

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch(
                "ollama_marshal.server.ModelLifecycle",
                return_value=lifecycle,
            ),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with server_mod.lifespan(app):
                pass

            # Should have been called during drain
            assert queues.total_pending.call_count >= 1

    async def test_lifespan_unload_models_on_shutdown(self):
        config = MarshalConfig(
            shutdown=ShutdownConfig(
                mode=ShutdownMode.IMMEDIATE,
                unload_models=True,
            )
        )
        app = create_app(config)

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        memory.get_loaded_models.return_value = {
            "llama3:latest": LoadedModel(name="llama3:latest", size_vram=4_000_000_000),
        }

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch(
                "ollama_marshal.server.ModelLifecycle",
                return_value=lifecycle,
            ),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with server_mod.lifespan(app):
                pass

            lifecycle.unload_all.assert_called_once_with(["llama3:latest"])

    async def test_lifespan_no_unload_when_disabled(self):
        config = MarshalConfig(shutdown=ShutdownConfig(unload_models=False))
        app = create_app(config)

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch(
                "ollama_marshal.server.ModelLifecycle",
                return_value=lifecycle,
            ),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with server_mod.lifespan(app):
                pass

            lifecycle.unload_all.assert_not_called()


# ---------------------------------------------------------------------------
# Route handler bodies via direct invocation
# ---------------------------------------------------------------------------


def _find_route_handler(app, path):
    for route in app.routes:
        if isinstance(route, Route) and route.path == path:
            return route.endpoint
    return None


class TestRouteHandlers:
    def _setup_globals(self):
        server_mod._queues = ModelQueues()
        server_mod._config = MarshalConfig()
        server_mod._memory = _mock_memory()
        server_mod._scheduler = _mock_scheduler()
        server_mod._started_at = time.monotonic()

    async def test_api_chat_handler(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/api/chat")
        assert handler is not None

        queues = server_mod._queues

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"model": "llama3:latest", "stream": False}
        )
        request.headers = {}

        async def complete():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                envs[0].complete({"done": True})

        task = asyncio.create_task(complete())
        try:
            result = await handler(request)
            assert result is not None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_api_generate_handler(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/api/generate")
        assert handler is not None

        queues = server_mod._queues

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"model": "llama3:latest", "prompt": "hi"}
        )
        request.headers = {}

        async def complete():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                envs[0].complete({"response": "hello"})

        task = asyncio.create_task(complete())
        try:
            result = await handler(request)
            assert result is not None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_api_embeddings_handler(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/api/embeddings")
        assert handler is not None

        queues = server_mod._queues

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"model": "llama3:latest", "prompt": "test"}
        )
        request.headers = {}

        async def complete():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                envs[0].complete({"embedding": [0.1, 0.2]})

        task = asyncio.create_task(complete())
        try:
            result = await handler(request)
            assert result is not None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_openai_chat_handler(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/v1/chat/completions")
        assert handler is not None

        queues = server_mod._queues

        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "model": "llama3:latest",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        request.headers = {}

        async def complete():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "message": {"role": "assistant", "content": "hello"},
                }
                del mock_resp.__aiter__
                envs[0].complete(mock_resp)

        task = asyncio.create_task(complete())
        try:
            result = await handler(request)
            assert result is not None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_openai_completions_handler(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/v1/completions")
        assert handler is not None

        queues = server_mod._queues

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"model": "llama3:latest", "prompt": "Once upon"}
        )
        request.headers = {}

        async def complete():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"response": "a time"}
                del mock_resp.__aiter__
                envs[0].complete(mock_resp)

        task = asyncio.create_task(complete())
        try:
            result = await handler(request)
            assert result is not None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_openai_embeddings_handler(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/v1/embeddings")
        assert handler is not None

        queues = server_mod._queues

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"model": "llama3:latest", "input": "test"}
        )
        request.headers = {}

        async def complete():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"embedding": [0.1, 0.2]}
                del mock_resp.__aiter__
                envs[0].complete(mock_resp)

        task = asyncio.create_task(complete())
        try:
            result = await handler(request)
            assert result is not None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_passthrough_handler(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/api/{path:path}")
        assert handler is not None

        request = MagicMock()
        request.method = "GET"
        request.body = AsyncMock(return_value=None)

        mock_resp = MagicMock()
        mock_resp.content = b'{"models": []}'
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}

        with patch(
            "ollama_marshal.server.forward_passthrough",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await handler(request, "tags")
            assert result.status_code == 200

    async def test_passthrough_handler_post(self):
        self._setup_globals()
        app = create_app(MarshalConfig())
        handler = _find_route_handler(app, "/api/{path:path}")
        assert handler is not None

        request = MagicMock()
        request.method = "POST"
        request.body = AsyncMock(return_value=b'{"name": "llama3"}')

        mock_resp = MagicMock()
        mock_resp.content = b'{"status": "success"}'
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}

        with patch(
            "ollama_marshal.server.forward_passthrough",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await handler(request, "pull")
            assert result.status_code == 200
