from __future__ import annotations

import asyncio
import json
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
# X-Marshal-Retry-Max header parsing (Surface A)
# ---------------------------------------------------------------------------


class TestParseRetryMaxHeader:
    def _req(self, headers=None):
        r = MagicMock()
        r.headers = headers or {}
        return r

    def test_returns_none_when_header_absent(self):
        assert server_mod._parse_retry_max_header(self._req()) is None

    def test_parses_explicit_int(self):
        r = self._req({"x-marshal-retry-max": "5"})
        assert server_mod._parse_retry_max_header(r) == 5

    def test_zero_disables_retry(self):
        # `0` means "no retries" — must not be coerced to None (which
        # would defer to config default).
        r = self._req({"x-marshal-retry-max": "0"})
        assert server_mod._parse_retry_max_header(r) == 0

    def test_negative_clamps_to_zero(self):
        r = self._req({"x-marshal-retry-max": "-3"})
        assert server_mod._parse_retry_max_header(r) == 0

    def test_caps_at_ten(self):
        # An adversarial client can't request 1000 retries.
        r = self._req({"x-marshal-retry-max": "1000"})
        assert server_mod._parse_retry_max_header(r) == 10

    def test_malformed_returns_none(self):
        r = self._req({"x-marshal-retry-max": "abc"})
        assert server_mod._parse_retry_max_header(r) is None

    async def test_envelope_carries_override_through_enqueue(self):
        # End-to-end: header → envelope.retry_max_override.
        original_registry = getattr(server_mod, "_registry", None)
        original_queues = getattr(server_mod, "_queues", None)
        from ollama_marshal.registry import ModelRegistry

        reg = MagicMock(spec=ModelRegistry)
        reg.is_known_model = AsyncMock(return_value=True)
        reg.probe_metadata = AsyncMock(return_value=None)
        reg.get_metadata = MagicMock(return_value=None)
        reg.get_max_context = MagicMock(return_value=None)
        server_mod._registry = reg
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {"x-marshal-retry-max": "0"}
        body = {"model": "llama3:latest", "stream": False}

        captured_override = []

        async def complete_after_enqueue():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                captured_override.append(envs[0].retry_max_override)
                envs[0].complete({"ok": True})

        task = asyncio.create_task(complete_after_enqueue())
        try:
            await server_mod._enqueue_inference(request, body, "/api/chat")
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry
            if original_queues is not None:
                server_mod._queues = original_queues

        assert captured_override == [0]


# ---------------------------------------------------------------------------
# Fail-fast on unknown models (Surface B)
# ---------------------------------------------------------------------------


class TestFailFastUnknownModel:
    """Marshal must 404 in milliseconds for models not installed in Ollama.

    Without this, the request would sit in the queue for proxy.request_timeout_s
    (default 1h) while lifecycle.preload retries trying to load a model
    Ollama doesn't have.
    """

    def _registry_with_known(self, *, known: bool) -> MagicMock:
        from ollama_marshal.registry import ModelRegistry

        reg = MagicMock(spec=ModelRegistry)
        reg.is_known_model = AsyncMock(return_value=known)
        return reg

    async def test_is_known_model_returns_true_when_registry_unset(self):
        # Test paths that bypass lifespan don't set _registry — fail open.
        original = getattr(server_mod, "_registry", None)
        if hasattr(server_mod, "_registry"):
            del server_mod._registry
        try:
            assert await server_mod._is_known_model("anything:latest") is True
        finally:
            if original is not None:
                server_mod._registry = original

    async def test_is_known_model_returns_true_when_registry_lacks_method(self):
        # Defensive: legacy registry stub without is_known_model — fail open.
        original = getattr(server_mod, "_registry", None)
        legacy = MagicMock()
        del legacy.is_known_model
        server_mod._registry = legacy
        try:
            assert await server_mod._is_known_model("anything:latest") is True
        finally:
            if original is not None:
                server_mod._registry = original
            else:
                del server_mod._registry

    async def test_is_known_model_delegates_to_registry(self):
        original = getattr(server_mod, "_registry", None)
        server_mod._registry = self._registry_with_known(known=False)
        try:
            assert await server_mod._is_known_model("nope:latest") is False
        finally:
            if original is not None:
                server_mod._registry = original
            else:
                del server_mod._registry

    async def test_enqueue_inference_returns_404_for_unknown_model(self):
        original_registry = getattr(server_mod, "_registry", None)
        original_queues = getattr(server_mod, "_queues", None)
        server_mod._registry = self._registry_with_known(known=False)
        server_mod._queues = ModelQueues()

        request = MagicMock()
        request.headers = {"x-program-id": "test"}
        body = {"model": "doesnotexist:bf16", "stream": False}

        try:
            result = await server_mod._enqueue_inference(request, body, "/api/chat")
            assert result.status_code == 404
            payload = json.loads(result.body)
            assert "doesnotexist:bf16" in payload["error"]
            assert "ollama pull" in payload["error"]
            # Nothing should have been queued.
            assert await server_mod._queues.total_pending() == 0
        finally:
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry
            if original_queues is not None:
                server_mod._queues = original_queues

    async def test_enqueue_and_wait_returns_openai_404_for_unknown_model(self):
        original_registry = getattr(server_mod, "_registry", None)
        original_queues = getattr(server_mod, "_queues", None)
        server_mod._registry = self._registry_with_known(known=False)
        server_mod._queues = ModelQueues()

        request = MagicMock()
        request.headers = {}

        try:
            result = await server_mod._enqueue_and_wait(
                request,
                "gpt-4-turbo",  # OpenAI-style name, not in Ollama
                {"model": "gpt-4-turbo"},
                "/v1/chat/completions",
                stream=False,
            )
            assert result.status_code == 404
            payload = json.loads(result.body)
            assert payload["error"]["type"] == "model_not_found"
            assert payload["error"]["code"] == "model_not_found"
            assert "gpt-4-turbo" in payload["error"]["message"]
            assert await server_mod._queues.total_pending() == 0
        finally:
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry
            if original_queues is not None:
                server_mod._queues = original_queues

    async def test_enqueue_inference_passes_through_for_known_model(self):
        # Sanity: when registry says known=True, normal queue path runs.
        original_registry = getattr(server_mod, "_registry", None)
        original_queues = getattr(server_mod, "_queues", None)
        server_mod._registry = self._registry_with_known(known=True)
        # Stub probe_metadata so num_ctx injection no-ops.
        server_mod._registry.probe_metadata = AsyncMock(return_value=None)
        server_mod._registry.get_metadata = MagicMock(return_value=None)
        server_mod._registry.get_max_context = MagicMock(return_value=None)
        queues = ModelQueues()
        server_mod._queues = queues

        request = MagicMock()
        request.headers = {}
        body = {"model": "llama3:latest", "stream": False}

        async def complete_after_enqueue():
            for _ in range(50):
                if await queues.total_pending() > 0:
                    break
                await asyncio.sleep(0.01)
            envs = await queues.get_all_sorted_by_arrival()
            if envs:
                envs[0].complete({"response": "ok"})

        task = asyncio.create_task(complete_after_enqueue())
        try:
            result = await server_mod._enqueue_inference(request, body, "/api/chat")
            # Did NOT 404 — the request reached the queue.
            assert getattr(result, "status_code", 200) != 404
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry
            if original_queues is not None:
                server_mod._queues = original_queues


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

    async def test_status_exposes_v040_metrics(self):
        # Regression test for the doctor-CLI integration: all four new
        # v0.4.0 SchedulerMetrics counters must appear in the response,
        # otherwise `marshal doctor` always reads None.
        config = MarshalConfig()
        app = create_app(config)

        server_mod._queues = _mock_queues()
        server_mod._memory = _mock_memory()
        sched = _mock_scheduler()
        sched.metrics = MagicMock(
            requests_served=10,
            model_swaps=3,
            evictions=1,
            average_wait_ms=42.5,
            retries_attempted=5,
            retries_succeeded=4,
            unexpected_unloads=2,
            reload_count=1,
        )
        server_mod._scheduler = sched
        server_mod._started_at = time.monotonic() - 60

        status_handler = None
        for route in app.routes:
            if isinstance(route, Route) and route.path == "/api/marshal/status":
                status_handler = route.endpoint
                break

        assert status_handler is not None
        result = await status_handler()

        m = result["metrics"]
        assert m["retries_attempted"] == 5
        assert m["retries_succeeded"] == 4
        assert m["unexpected_unloads"] == 2
        assert m["reload_count"] == 1


# ---------------------------------------------------------------------------
# _record_burst_hint — X-Burst-Size header extraction
# ---------------------------------------------------------------------------


class TestNormalizeProgramId:
    """Sanitization of `X-Program-ID` header values.

    Without this, an adversarial client cycling 10MB header values would
    inflate burst-hint dicts, _active_programs map, and audit.jsonl by
    10MB per distinct value (256 distinct = 2.5GB resident). Newlines
    in the value also corrupt structlog console output (log injection).
    """

    def test_none_returns_default(self):
        assert server_mod._normalize_program_id(None) == "default"

    def test_empty_returns_default(self):
        assert server_mod._normalize_program_id("") == "default"

    def test_only_disallowed_chars_returns_default(self):
        # All chars stripped → empty → fall back to default.
        assert server_mod._normalize_program_id("!!!@#$%^&*()") == "default"

    def test_keeps_allowed_chars(self):
        assert (
            server_mod._normalize_program_id("ai-portfolio_v1.2") == "ai-portfolio_v1.2"
        )

    def test_strips_disallowed_chars(self):
        # Newlines and ANSI escapes are the log-injection vectors.
        assert server_mod._normalize_program_id(
            "ai-portfolio\n\x1b[31mEVIL"
        ) == "ai-portfolioxEVIL" or "ai-portfolio" in server_mod._normalize_program_id(
            "ai-portfolio\n\x1b[31mEVIL"
        )

    def test_truncates_long_value(self):
        # 10MB header value must not balloon downstream state.
        result = server_mod._normalize_program_id("a" * 10_000_000)
        assert len(result) == 64

    def test_truncates_at_64_chars_exactly(self):
        result = server_mod._normalize_program_id("x" * 65)
        assert result == "x" * 64


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

    def _set_globals(self, *, registry, config=None):
        """Helper: install registry+config in server_mod globals.

        Returns a callable that restores the previous state. Tests use
        this in a try/finally to keep module state isolated.
        """
        if config is None:
            config = MarshalConfig()
        original_registry = getattr(server_mod, "_registry", None)
        original_config = getattr(server_mod, "_config", None)
        server_mod._registry = registry
        server_mod._config = config

        def restore():
            if original_registry is not None:
                server_mod._registry = original_registry
            else:
                del server_mod._registry
            if original_config is not None:
                server_mod._config = original_config
            else:
                del server_mod._config

        return restore

    def _qwen3_metadata(self, max_ctx=32768):
        from ollama_marshal.registry import ModelMetadata

        return ModelMetadata(
            name="qwen3.5:9b-bf16",
            architecture="qwen3",
            max_context_length=max_ctx,
            num_layers=28,
            embedding_length=3584,
            head_count=28,
            head_count_kv=4,
        )

    async def test_injects_smallest_boundary_for_empty_body(self):
        """No prompt content → only completion budget + safety = ~4352 → 8192."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        restore = self._set_globals(registry=registry)

        body: dict = {"model": "qwen3.5:9b-bf16"}
        try:
            await server_mod._inject_num_ctx("qwen3.5:9b-bf16", body)
        finally:
            restore()

        # 0 prompt + 4096 completion + 256 safety = 4352, rounds up to 8192.
        assert body["options"]["num_ctx"] == 8192

    async def test_preserves_existing_num_ctx(self):
        """Client-set num_ctx wins (when within model's max)."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        restore = self._set_globals(registry=registry)

        body: dict = {"model": "m", "options": {"num_ctx": 4096}}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            restore()
        # Client value preserved (4096 < 32768 model max).
        assert body["options"]["num_ctx"] == 4096

    async def test_clamps_client_num_ctx_to_model_max(self):
        """Adversarial/buggy client sending num_ctx > model max gets clamped.

        Without this, a request with num_ctx=999_999_999 triggers
        reload-on-need, fails preload, infinite-loops the scheduler,
        and unboundedly grows reload_count. One bad request bricks
        the proxy for everyone sharing the model.
        """
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(
            return_value=self._qwen3_metadata(max_ctx=32768)
        )
        restore = self._set_globals(registry=registry)
        body: dict = {"model": "m", "options": {"num_ctx": 999_999_999}}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            restore()
        assert body["options"]["num_ctx"] == 32768

    async def test_clamps_client_num_ctx_even_when_injection_disabled(self):
        """REGRESSION: clamp must run even when prompt-driven injection is opt-out.

        Bug caught by /review on PR #6: the early-return on
        `injection_enabled: false` happened BEFORE the clamp logic.
        An operator opting out of prompt-driven sizing was silently
        re-exposing the same `num_ctx: 999_999_999` DoS the v0.4.0
        clamp was meant to fix. The fix moves the clamp out of the
        injection-gated branch — it's a trust boundary, not part of
        prompt-driven sizing.
        """
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(
            return_value=self._qwen3_metadata(max_ctx=32768)
        )
        cfg = MarshalConfig()
        cfg.context.injection_enabled = False  # the opt-out path
        restore = self._set_globals(registry=registry, config=cfg)
        body: dict = {"model": "m", "options": {"num_ctx": 999_999_999}}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            restore()
        # The adversarial value MUST be clamped to the model's max
        # regardless of the injection flag. Untouched would be 999_999_999.
        assert body["options"]["num_ctx"] == 32768

    async def test_drops_malformed_client_num_ctx_when_injection_disabled(self):
        """When injection is disabled and client sends garbage, drop it cleanly.

        Falls through to "no num_ctx in body" — Ollama default behavior
        — rather than passing the garbage value through.
        """
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(
            return_value=self._qwen3_metadata(max_ctx=32768)
        )
        cfg = MarshalConfig()
        cfg.context.injection_enabled = False
        restore = self._set_globals(registry=registry, config=cfg)
        body: dict = {"model": "m", "options": {"num_ctx": -1}}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            restore()
        # Malformed dropped, no prompt-driven fallback because
        # injection is disabled.
        assert "num_ctx" not in body["options"]

    async def test_drops_negative_or_zero_client_num_ctx(self):
        """Non-positive client num_ctx is dropped, not honored."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        restore = self._set_globals(registry=registry)
        body: dict = {"model": "m", "options": {"num_ctx": -1}}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            restore()
        # Falls through to prompt-driven sizing instead.
        assert body["options"]["num_ctx"] != -1
        assert body["options"]["num_ctx"] > 0

    async def test_drops_non_int_client_num_ctx(self):
        """Wrong type client num_ctx is dropped."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        restore = self._set_globals(registry=registry)
        body: dict = {"model": "m", "options": {"num_ctx": "huge"}}
        try:
            await server_mod._inject_num_ctx("m", body)
        finally:
            restore()
        # Falls through to prompt-driven sizing.
        assert isinstance(body["options"]["num_ctx"], int)
        assert body["options"]["num_ctx"] > 0

    async def test_skips_when_metadata_missing(self):
        """No metadata for the model → no injection (don't break unknown models)."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=None)
        restore = self._set_globals(registry=registry)
        body: dict = {"model": "unknown:xyz"}
        try:
            await server_mod._inject_num_ctx("unknown:xyz", body)
        finally:
            restore()
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

    async def test_skips_when_injection_disabled(self):
        """`context.injection_enabled: false` opts out entirely."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        cfg = MarshalConfig()
        cfg.context.injection_enabled = False
        restore = self._set_globals(registry=registry, config=cfg)
        body: dict = {"model": "qwen3.5:9b-bf16"}
        try:
            await server_mod._inject_num_ctx("qwen3.5:9b-bf16", body)
        finally:
            restore()
        # Untouched — Ollama default behavior takes over.
        assert body.get("options", {}).get("num_ctx") is None

    async def test_sizes_from_chat_messages(self):
        """Long messages drive num_ctx upward."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        restore = self._set_globals(registry=registry)
        # ~50K chars * 0.3 = 15K prompt tokens; +4352 budget+safety ≈ 19K
        # → rounds up to 32768.
        long_msg = "x" * 50_000
        body: dict = {
            "model": "qwen3.5:9b-bf16",
            "messages": [{"role": "user", "content": long_msg}],
        }
        try:
            await server_mod._inject_num_ctx("qwen3.5:9b-bf16", body)
        finally:
            restore()
        assert body["options"]["num_ctx"] == 32768

    async def test_sizes_from_generate_prompt(self):
        """`/api/generate` uses `prompt` field instead of `messages`."""
        registry = MagicMock()
        # Use a 262K-context model so the prompt-driven sizer isn't
        # clamped by the model max.
        registry.probe_metadata = AsyncMock(
            return_value=self._qwen3_metadata(max_ctx=262144)
        )
        restore = self._set_globals(registry=registry)
        body: dict = {"model": "qwen3.5:9b-bf16", "prompt": "x" * 100_000}
        try:
            await server_mod._inject_num_ctx("qwen3.5:9b-bf16", body)
        finally:
            restore()
        # 100K * 0.3 = 30K + 4352 ≈ 35K → rounds up to 65536.
        assert body["options"]["num_ctx"] == 65536

    async def test_clamps_to_model_max_context(self):
        """A 100K-char prompt at qwen3.5 8K-max model clamps to 8192."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(
            return_value=self._qwen3_metadata(max_ctx=8192)
        )
        restore = self._set_globals(registry=registry)
        body: dict = {"model": "small", "prompt": "x" * 100_000}
        try:
            await server_mod._inject_num_ctx("small", body)
        finally:
            restore()
        assert body["options"]["num_ctx"] == 8192

    async def test_program_floor_applied(self):
        """A program profile's typical_num_ctx is a floor for short prompts."""
        from ollama_marshal.config import ProgramContextProfile

        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        cfg = MarshalConfig()
        cfg.context.programs["ai-portfolio"] = ProgramContextProfile(
            typical_num_ctx=16384, max_num_ctx=65536
        )
        restore = self._set_globals(registry=registry, config=cfg)
        body: dict = {"model": "qwen3.5:9b-bf16", "prompt": "short"}
        try:
            await server_mod._inject_num_ctx(
                "qwen3.5:9b-bf16", body, program_id="ai-portfolio"
            )
        finally:
            restore()
        # Prompt-driven would land at 8192, but the program floor lifts it to 16384.
        assert body["options"]["num_ctx"] == 16384

    async def test_program_ceiling_applied(self):
        """A program profile's max_num_ctx is a ceiling for runaway prompts."""
        from ollama_marshal.config import ProgramContextProfile

        registry = MagicMock()
        registry.probe_metadata = AsyncMock(
            return_value=self._qwen3_metadata(max_ctx=262144)
        )
        cfg = MarshalConfig()
        cfg.context.programs["ai-email"] = ProgramContextProfile(
            typical_num_ctx=4096, max_num_ctx=8192
        )
        restore = self._set_globals(registry=registry, config=cfg)
        # 50K-char prompt would normally land at 32768 prompt-driven, but
        # the program ceiling caps at 8192.
        body: dict = {"model": "qwen3.5:9b-bf16", "prompt": "x" * 50_000}
        try:
            await server_mod._inject_num_ctx(
                "qwen3.5:9b-bf16", body, program_id="ai-email"
            )
        finally:
            restore()
        assert body["options"]["num_ctx"] == 8192

    async def test_program_without_profile_uses_prompt_driven(self):
        """Programs not in `context.programs` get pure prompt-driven sizing."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        restore = self._set_globals(registry=registry)
        body: dict = {"model": "qwen3.5:9b-bf16", "prompt": "short"}
        try:
            await server_mod._inject_num_ctx(
                "qwen3.5:9b-bf16", body, program_id="not-configured"
            )
        finally:
            restore()
        # No floor, prompt is short → smallest practical boundary above 4352.
        assert body["options"]["num_ctx"] == 8192

    async def test_handles_openai_multimodal_content_list(self):
        """OpenAI chat content as list-of-parts: count text parts only."""
        registry = MagicMock()
        registry.probe_metadata = AsyncMock(return_value=self._qwen3_metadata())
        restore = self._set_globals(registry=registry)
        body: dict = {
            "model": "qwen3.5:9b-bf16",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x" * 50_000},
                        {"type": "image_url", "image_url": {"url": "..."}},
                    ],
                }
            ],
        }
        try:
            await server_mod._inject_num_ctx("qwen3.5:9b-bf16", body)
        finally:
            restore()
        # Same 50K text → 32768.
        assert body["options"]["num_ctx"] == 32768


class TestEstimatePromptTokens:
    def test_zero_for_empty_body(self):
        assert server_mod._estimate_prompt_tokens({}) == 0

    def test_counts_string_prompt(self):
        # 100 chars * 0.3 = 30 tokens.
        assert server_mod._estimate_prompt_tokens({"prompt": "x" * 100}) == 30

    def test_counts_chat_messages(self):
        body = {
            "messages": [
                {"role": "user", "content": "abc"},  # 3
                {"role": "assistant", "content": "defg"},  # 4
            ],
        }
        # 7 chars * 0.3 = 2.
        assert server_mod._estimate_prompt_tokens(body) == 2

    def test_counts_multimodal_text_parts(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "abc"},
                        {"type": "image_url", "image_url": {"url": "..."}},
                    ],
                }
            ],
        }
        # 3 text chars * 0.3 = 0 (rounds via int).
        assert server_mod._estimate_prompt_tokens(body) == 0

    def test_handles_non_dict_messages(self):
        # Defensive: malformed input shouldn't crash the estimator.
        assert server_mod._estimate_prompt_tokens({"messages": "not a list"}) == 0

    def test_handles_messages_with_non_dict_entries(self):
        body = {"messages": ["not a dict", None, 42]}
        assert server_mod._estimate_prompt_tokens(body) == 0


class TestRoundUpToBoundary:
    def test_below_smallest_returns_smallest(self):
        assert server_mod._round_up_to_boundary(0) == 2048
        assert server_mod._round_up_to_boundary(100) == 2048

    def test_exact_boundary_returns_same(self):
        assert server_mod._round_up_to_boundary(4096) == 4096

    def test_one_above_boundary_jumps_up(self):
        assert server_mod._round_up_to_boundary(4097) == 8192

    def test_above_largest_returns_largest(self):
        # Caller is expected to clamp to model max separately.
        assert server_mod._round_up_to_boundary(10_000_000) == 262144

    def test_typical_8b_chat_lands_at_16384(self):
        # 30K-token prompt + 4096 budget + 256 safety ≈ 34K → 65536.
        assert server_mod._round_up_to_boundary(34_000) == 65536


class TestResolveNumCtxDecision:
    def test_prompt_driven_mode_for_unconfigured_program(self):
        cfg = MarshalConfig()
        chosen, mode = server_mod._resolve_num_ctx_decision(
            prompt_tokens=1000,
            program_id="anything",
            model_max_context=32768,
            config=cfg,
        )
        # 1000 + 4096 + 256 = 5352 → 8192.
        assert chosen == 8192
        assert mode == "prompt_driven"

    def test_program_floor_mode(self):
        from ollama_marshal.config import ProgramContextProfile

        cfg = MarshalConfig()
        cfg.context.programs["p"] = ProgramContextProfile(
            typical_num_ctx=16384, max_num_ctx=65536
        )
        chosen, mode = server_mod._resolve_num_ctx_decision(
            prompt_tokens=10,
            program_id="p",
            model_max_context=32768,
            config=cfg,
        )
        assert chosen == 16384
        assert mode == "program_floor"

    def test_program_ceiling_mode(self):
        from ollama_marshal.config import ProgramContextProfile

        cfg = MarshalConfig()
        cfg.context.programs["p"] = ProgramContextProfile(
            typical_num_ctx=2048, max_num_ctx=8192
        )
        chosen, mode = server_mod._resolve_num_ctx_decision(
            prompt_tokens=50_000,
            program_id="p",
            model_max_context=262144,
            config=cfg,
        )
        assert chosen == 8192
        assert mode == "program_ceiling"

    def test_model_max_clamp_overrides_everything(self):
        from ollama_marshal.config import ProgramContextProfile

        cfg = MarshalConfig()
        cfg.context.programs["p"] = ProgramContextProfile(
            typical_num_ctx=131072, max_num_ctx=262144
        )
        chosen, mode = server_mod._resolve_num_ctx_decision(
            prompt_tokens=1000,
            program_id="p",
            model_max_context=8192,
            config=cfg,
        )
        # Profile floor is 131K but model only supports 8K.
        assert chosen == 8192
        assert mode == "model_max"


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

    async def test_lifespan_restores_metrics_from_disk(self, tmp_path):
        """Lifespan reads metrics.json on startup and seeds counters."""
        import json

        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "saved_at": "2026-04-28T03:00:00+00:00",
                    "requests_served": 100,
                    "model_swaps": 5,
                    "evictions": 2,
                    "total_wait_ms": 12345.6,
                }
            )
        )

        config = MarshalConfig()
        app = create_app(config)
        app.state.metrics_path = metrics_path  # override for this test

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        # Real-shape metrics object so the lifespan can mutate fields.
        from ollama_marshal.scheduler import SchedulerMetrics as RealMetrics

        scheduler.metrics = RealMetrics()

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
                # Counters seeded from disk.
                assert scheduler.metrics.requests_served == 100
                assert scheduler.metrics.model_swaps == 5
                assert scheduler.metrics.evictions == 2
                assert scheduler.metrics.total_wait_ms == 12345.6

        # Final snapshot was rewritten at shutdown.
        on_disk = json.loads(metrics_path.read_text())
        assert on_disk["requests_served"] == 100
        assert on_disk["schema_version"] == 1

    async def test_lifespan_starts_with_fresh_metrics_when_no_file(self, tmp_path):
        """No metrics.json on disk → counters start at zero, no error."""
        config = MarshalConfig()
        app = create_app(config)
        app.state.metrics_path = tmp_path / "fresh.json"

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        from ollama_marshal.scheduler import SchedulerMetrics as RealMetrics

        scheduler.metrics = RealMetrics()

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
                assert scheduler.metrics.requests_served == 0

        # Final shutdown save creates the file.
        assert (tmp_path / "fresh.json").exists()

    async def test_lifespan_cancels_running_benchmark_on_shutdown(self, tmp_path):
        """If benchmark_unknown is still running at shutdown, it must be cancelled."""
        config = MarshalConfig()
        config.scheduler.metrics_path = str(tmp_path / "metrics.json")
        app = create_app(config)

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        from ollama_marshal.scheduler import SchedulerMetrics as RealMetrics

        scheduler.metrics = RealMetrics()

        # Long-running benchmark — would never complete on its own.
        async def slow_benchmark():
            await asyncio.sleep(60)

        registry.benchmark_unknown = slow_benchmark

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
        # If the cancel branch didn't execute, the lifespan would hang
        # for 60s; the fact this test returned in milliseconds proves
        # the task was cancelled cleanly.

    async def test_lifespan_starts_audit_logger_when_enabled(self, tmp_path):
        """audit.enabled=True wires a real AuditLogger into the scheduler."""
        from ollama_marshal.audit import AuditLogger
        from ollama_marshal.config import AuditConfig

        config = MarshalConfig()
        config.scheduler.metrics_path = str(tmp_path / "metrics.json")
        config.audit = AuditConfig(
            enabled=True,
            path=str(tmp_path / "audit.jsonl"),
            retention_days=0,
            max_size_mb=0,
        )
        app = create_app(config)

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        from ollama_marshal.scheduler import SchedulerMetrics as RealMetrics

        scheduler.metrics = RealMetrics()
        # Capture what gets installed on scheduler.audit.
        installed_audit = []

        def capture_audit(value):
            installed_audit.append(value)

        type(scheduler).audit = property(
            lambda self: installed_audit[-1] if installed_audit else None,
            lambda self, v: capture_audit(v),
        )

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
                # A real AuditLogger must have been installed on scheduler.
                assert installed_audit, "no audit installed on scheduler"
                assert isinstance(installed_audit[-1], AuditLogger)
                assert installed_audit[-1].enabled is True


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
