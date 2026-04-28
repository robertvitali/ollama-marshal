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
    q.pending_programs_by_model = AsyncMock(
        return_value={"llama3:latest": ["program-alpha"]}
    )
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
    sched.active_programs_by_model = MagicMock(
        return_value={"llama3:latest": ["program-beta"]}
    )
    return sched


def _mock_registry():
    """Stub registry whose probe_metadata is an awaitable that returns None.

    Prevents the num_ctx injection helper from blowing up in tests that
    set module globals directly without going through lifespan. Returning
    None means injection is a no-op (current Ollama default behavior).
    """
    from ollama_marshal.registry import ModelRegistry

    reg = MagicMock(spec=ModelRegistry)
    reg.probe_metadata = AsyncMock(return_value=None)
    reg.get_metadata = MagicMock(return_value=None)
    reg.get_max_context = MagicMock(return_value=None)
    return reg


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_returns_fastapi_instance(self):
        config = MarshalConfig()
        app = create_app(config)
        assert isinstance(app, FastAPI)

    def test_app_title_and_version(self):
        from ollama_marshal import __version__

        config = MarshalConfig()
        app = create_app(config)
        assert app.title == "ollama-marshal"
        assert app.version == __version__

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

    def test_status_alias_route_exists(self):
        # Short alias so `curl localhost:11435/status` works.
        assert "/status" in self._get_routes()

    def test_passthrough_route_exists(self):
        assert "/api/{path:path}" in self._get_routes()


# ---------------------------------------------------------------------------
# _resolve_timeout (per-request timeout override via X-Request-Timeout header)
# ---------------------------------------------------------------------------


class TestResolveTimeout:
    """Cover the per-request X-Request-Timeout header + config fallback."""

    def _request(self, headers: dict, *, with_config: bool = True) -> MagicMock:
        req = MagicMock()
        req.headers = headers
        if with_config:
            cfg = MarshalConfig()
            cfg.proxy.request_timeout_s = 1234
            req.app.state.config = cfg
        else:
            req.app.state = MagicMock(spec=[])
        return req

    def test_no_header_uses_config(self):
        from ollama_marshal.server import _resolve_timeout

        assert _resolve_timeout(self._request({})) == 1234

    def test_header_overrides_config(self):
        from ollama_marshal.server import _resolve_timeout

        assert _resolve_timeout(self._request({"x-request-timeout": "60"})) == 60

    def test_header_zero_falls_back_to_config(self):
        from ollama_marshal.server import _resolve_timeout

        # 0 is invalid; we ignore it and use config.
        assert _resolve_timeout(self._request({"x-request-timeout": "0"})) == 1234

    def test_header_negative_falls_back_to_config(self):
        from ollama_marshal.server import _resolve_timeout

        assert _resolve_timeout(self._request({"x-request-timeout": "-5"})) == 1234

    def test_header_non_int_falls_back_to_config(self):
        from ollama_marshal.server import _resolve_timeout

        assert _resolve_timeout(self._request({"x-request-timeout": "abc"})) == 1234

    def test_no_config_uses_3600s_default(self):
        from ollama_marshal.server import _resolve_timeout

        assert _resolve_timeout(self._request({}, with_config=False)) == 3600


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
        # Programs are the union of pending-queue programs and recently-active
        # programs from the scheduler, sorted and deduped.
        assert result["loaded_models"][0]["programs"] == [
            "program-alpha",
            "program-beta",
        ]
        assert result["metrics"]["requests_served"] == 10
        assert result["queue"]["total_pending"] == 2


# ---------------------------------------------------------------------------
# _record_burst_hint — X-Burst-Size header extraction
# ---------------------------------------------------------------------------


class TestRecordBurstHint:
    def _request(self, headers: dict) -> MagicMock:
        req = MagicMock()
        req.headers = headers
        return req

    def test_records_when_header_present(self):
        sched = MagicMock()
        sched.burst_hints = MagicMock()
        sched.burst_hints.record = MagicMock(return_value=10)
        cfg = MagicMock()
        cfg.scheduler.max_skips = 5
        original_sched = getattr(server_mod, "_scheduler", None)
        original_cfg = getattr(server_mod, "_config", None)
        server_mod._scheduler = sched
        server_mod._config = cfg
        try:
            server_mod._record_burst_hint(
                self._request({"x-burst-size": "10"}), "ai-portfolio", "qwen3"
            )
        finally:
            if original_sched is not None:
                server_mod._scheduler = original_sched
            if original_cfg is not None:
                server_mod._config = original_cfg
        sched.burst_hints.record.assert_called_once_with("ai-portfolio", "qwen3", 10, 5)

    def test_skips_when_header_absent(self):
        sched = MagicMock()
        sched.burst_hints = MagicMock()
        original_sched = getattr(server_mod, "_scheduler", None)
        server_mod._scheduler = sched
        try:
            server_mod._record_burst_hint(self._request({}), "ai-portfolio", "qwen3")
        finally:
            if original_sched is not None:
                server_mod._scheduler = original_sched
        sched.burst_hints.record.assert_not_called()

    def test_skips_non_numeric_header(self):
        sched = MagicMock()
        sched.burst_hints = MagicMock()
        original_sched = getattr(server_mod, "_scheduler", None)
        server_mod._scheduler = sched
        try:
            server_mod._record_burst_hint(
                self._request({"x-burst-size": "not-a-number"}), "p", "m"
            )
        finally:
            if original_sched is not None:
                server_mod._scheduler = original_sched
        sched.burst_hints.record.assert_not_called()

    def test_skips_zero_or_negative(self):
        sched = MagicMock()
        sched.burst_hints = MagicMock()
        original_sched = getattr(server_mod, "_scheduler", None)
        server_mod._scheduler = sched
        try:
            server_mod._record_burst_hint(
                self._request({"x-burst-size": "0"}), "p", "m"
            )
            server_mod._record_burst_hint(
                self._request({"x-burst-size": "-5"}), "p", "m"
            )
        finally:
            if original_sched is not None:
                server_mod._scheduler = original_sched
        sched.burst_hints.record.assert_not_called()

    def test_skips_when_scheduler_unset(self):
        # No scheduler in module globals (e.g., tests bypassing lifespan).
        original_sched = getattr(server_mod, "_scheduler", None)
        if hasattr(server_mod, "_scheduler"):
            del server_mod._scheduler
        try:
            # Should not raise.
            server_mod._record_burst_hint(
                self._request({"x-burst-size": "10"}), "p", "m"
            )
        finally:
            if original_sched is not None:
                server_mod._scheduler = original_sched


# ---------------------------------------------------------------------------
# _inject_num_ctx
# ---------------------------------------------------------------------------


class TestInjectNumCtx:
    """Verify the num_ctx-injection helper.

    Stops Ollama from silently truncating context when KV cache slots
    don't fit at the model's full architectural context length.
    """

    async def test_injects_when_options_missing(self):
        """No options block at all → registry max gets injected."""
        from ollama_marshal.registry import ModelMetadata

        registry = MagicMock()
        registry.probe_metadata = AsyncMock(
            return_value=ModelMetadata(
                name="qwen3.5:9b-bf16",
                architecture="qwen3",
                max_context_length=32768,
                num_layers=28,
                embedding_length=3584,
                head_count=28,
                head_count_kv=4,
            )
        )
        original_registry = getattr(server_mod, "_registry", None)
        server_mod._registry = registry

        body: dict = {"model": "qwen3.5:9b-bf16"}
        try:
            await server_mod._inject_num_ctx("qwen3.5:9b-bf16", body)
        finally:
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry

        assert body["options"]["num_ctx"] == 32768

    async def test_preserves_existing_num_ctx(self):
        """Client-set num_ctx wins."""
        from ollama_marshal.registry import ModelMetadata

        registry = MagicMock()
        registry.probe_metadata = AsyncMock(
            return_value=ModelMetadata(
                name="m",
                architecture="x",
                max_context_length=99999,
                num_layers=1,
                embedding_length=1,
                head_count=1,
                head_count_kv=1,
            )
        )
        original_registry = getattr(server_mod, "_registry", None)
        server_mod._registry = registry
        body: dict = {"model": "m", "options": {"num_ctx": 4096}}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry
        # Client value preserved.
        assert body["options"]["num_ctx"] == 4096

    async def test_skips_when_metadata_missing(self):
        """No metadata for the model → no injection (don't break unknown models)."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=None)
        original_registry = getattr(server_mod, "_registry", None)
        server_mod._registry = registry
        body: dict = {"model": "unknown:xyz"}
        try:
            await server_mod._inject_num_ctx("unknown:xyz", body)
        finally:
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry
        # No options block created when there's nothing to set.
        assert body.get("options", {}).get("num_ctx") is None

    async def test_skips_when_model_empty(self):
        body: dict = {}
        await server_mod._inject_num_ctx("", body)
        assert "options" not in body or "num_ctx" not in body["options"]

    async def test_skips_when_options_not_dict(self):
        """Defensive: client sent something weird as options. Don't touch it."""
        body: dict = {"model": "m", "options": "not a dict"}
        await server_mod._inject_num_ctx("m", body)
        assert body["options"] == "not a dict"

    async def test_skips_when_registry_unset(self):
        """Tests that bypass lifespan don't crash the helper."""
        # Capture and remove _registry entirely.
        original_registry = getattr(server_mod, "_registry", None)
        if hasattr(server_mod, "_registry"):
            del server_mod._registry
        body: dict = {"model": "m"}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            if original_registry is not None:
                server_mod._registry = original_registry
        # No injection happened.
        assert "num_ctx" not in body.get("options", {})


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
        # _registry must be set explicitly — TestLifespan patches the
        # ModelRegistry class and leaves a MagicMock as the module global,
        # which would crash the num_ctx-injection helper if not reset.
        server_mod._registry = _mock_registry()
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
            result = await handler(request, "show")
            assert result.status_code == 200
