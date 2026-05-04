"""Tests for the admin pause/resume endpoints + bypass token middleware.

Coverage targets (v0.6.0 Stage 1):
- POST /api/marshal/admin/pause — gated, auth, drain wait, response shape
- POST /api/marshal/admin/resume — gated, auth, queue depth on response
- _check_admin_token — header missing, header wrong, header right
- _is_bypass_pause — header missing, header wrong, header right, no
  config token (None), no app.state config (test bypass)
- Auto-resume timer fires after delay (verified separately in
  test_scheduler.py — here we just verify the admin pause endpoint
  schedules it via the scheduler call)

Tests use httpx.ASGITransport with the in-process FastAPI app and
asgi-lifespan so the lifespan wires up the scheduler before tests fire
HTTP at the admin endpoints. Lifespan setup mocks Ollama-side
components so tests don't need real Ollama running.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from asgi_lifespan import LifespanManager

from ollama_marshal.config import (
    AdminConfig,
    MarshalConfig,
    SchedulerConfig,
    ShutdownConfig,
    ShutdownMode,
)
from ollama_marshal.scheduler import SchedulerMetrics
from ollama_marshal.server import _check_admin_token, _is_bypass_pause, create_app

ADMIN_TOK = "admin-tok-test-only"
BYPASS_TOK = "bypass-tok-test-only"


def _make_mock_components():
    """Match the mock pattern used in tests/test_server.py::TestLifespan."""
    memory = MagicMock()
    memory.start_polling = AsyncMock()
    memory.stop_polling = AsyncMock()
    memory.get_loaded_models.return_value = {}

    registry = MagicMock()
    registry.initialize = AsyncMock()
    registry.benchmark_unknown = AsyncMock()
    registry.start_polling = AsyncMock()
    registry.stop_polling = AsyncMock()

    lifecycle = MagicMock()
    lifecycle.unload_all = AsyncMock()

    scheduler = MagicMock()
    scheduler.start = AsyncMock()
    scheduler.stop = AsyncMock()
    scheduler.metrics = SchedulerMetrics()
    scheduler.is_paused.return_value = False
    scheduler.in_flight_count.return_value = 0
    scheduler.pause = AsyncMock(return_value=True)
    scheduler.resume = MagicMock()

    queues = MagicMock()
    queues.total_pending = AsyncMock(return_value=0)

    return memory, registry, lifecycle, scheduler, queues


def _admin_enabled_config(
    *, debug_enabled: bool = False, **scheduler_kw
) -> MarshalConfig:
    return MarshalConfig(
        admin=AdminConfig(
            pause_endpoints_enabled=True,
            admin_token=ADMIN_TOK,
            test_bypass_token=BYPASS_TOK,
        ),
        scheduler=SchedulerConfig(benchmark_on_startup=False, **scheduler_kw),
        shutdown=ShutdownConfig(
            mode=ShutdownMode.IMMEDIATE,
            drain_timeout=1,
            unload_models=False,
        ),
    )


# ---------------------------------------------------------------------------
# _check_admin_token (unit, no HTTP)
# ---------------------------------------------------------------------------


class TestCheckAdminToken:
    def test_returns_false_when_no_admin_token_configured(self):
        request = MagicMock()
        request.headers.get.return_value = "anything"
        cfg = MarshalConfig()  # default admin_token=None
        assert _check_admin_token(request, cfg) is False

    def test_returns_false_when_header_missing(self):
        request = MagicMock()
        request.headers.get.return_value = None
        cfg = MarshalConfig(
            admin=AdminConfig(pause_endpoints_enabled=True, admin_token=ADMIN_TOK)
        )
        assert _check_admin_token(request, cfg) is False

    def test_returns_false_when_header_wrong(self):
        request = MagicMock()
        request.headers.get.return_value = "wrong-token"
        cfg = MarshalConfig(
            admin=AdminConfig(pause_endpoints_enabled=True, admin_token=ADMIN_TOK)
        )
        assert _check_admin_token(request, cfg) is False

    def test_returns_true_when_header_matches(self):
        request = MagicMock()
        request.headers.get.return_value = ADMIN_TOK
        cfg = MarshalConfig(
            admin=AdminConfig(pause_endpoints_enabled=True, admin_token=ADMIN_TOK)
        )
        assert _check_admin_token(request, cfg) is True


# ---------------------------------------------------------------------------
# _is_bypass_pause (unit, no HTTP)
# ---------------------------------------------------------------------------


class TestIsBypassPause:
    def test_returns_false_when_app_state_has_no_config(self):
        """Tests that bypass lifespan have no app.state.config — graceful no."""
        request = MagicMock()
        request.app.state.config = None
        # Make getattr return None for the test's missing config.
        type(request.app.state).config = property(lambda self: None)
        assert _is_bypass_pause(request) is False

    def test_returns_false_when_no_test_bypass_token_configured(self):
        request = MagicMock()
        request.app.state.config = MarshalConfig()  # default test_bypass_token=None
        request.headers.get.return_value = "anything"
        assert _is_bypass_pause(request) is False

    def test_returns_false_when_header_missing(self):
        request = MagicMock()
        request.app.state.config = MarshalConfig(
            admin=AdminConfig(test_bypass_token=BYPASS_TOK)
        )
        request.headers.get.return_value = None
        assert _is_bypass_pause(request) is False

    def test_returns_false_when_header_wrong(self):
        request = MagicMock()
        request.app.state.config = MarshalConfig(
            admin=AdminConfig(test_bypass_token=BYPASS_TOK)
        )
        request.headers.get.return_value = "wrong-tok"
        assert _is_bypass_pause(request) is False

    def test_returns_true_when_header_matches(self):
        request = MagicMock()
        request.app.state.config = MarshalConfig(
            admin=AdminConfig(test_bypass_token=BYPASS_TOK)
        )
        request.headers.get.return_value = BYPASS_TOK
        assert _is_bypass_pause(request) is True


# ---------------------------------------------------------------------------
# Admin endpoints — full HTTP path through lifespan
# ---------------------------------------------------------------------------


@pytest.fixture
async def admin_client():
    """Start a marshal app with admin endpoints enabled, mocked Ollama-side."""
    memory, registry, lifecycle, scheduler, queues = _make_mock_components()
    config = _admin_enabled_config()
    app = create_app(config)

    with (
        patch("ollama_marshal.server.MemoryManager", return_value=memory),
        patch("ollama_marshal.server.ModelRegistry", return_value=registry),
        patch("ollama_marshal.server.ModelLifecycle", return_value=lifecycle),
        patch("ollama_marshal.server.Scheduler", return_value=scheduler),
        patch("ollama_marshal.server.ModelQueues", return_value=queues),
    ):
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client,
        ):
            yield client, scheduler, queues


class TestAdminPauseEndpoint:
    async def test_returns_404_when_endpoint_disabled(self):
        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        config = MarshalConfig(
            scheduler=SchedulerConfig(benchmark_on_startup=False),
        )  # admin disabled by default
        app = create_app(config)

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch("ollama_marshal.server.ModelLifecycle", return_value=lifecycle),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with (
                LifespanManager(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client,
            ):
                resp = await client.post("/api/marshal/admin/pause")

        assert resp.status_code == 404

    async def test_returns_401_without_admin_token(self, admin_client):
        client, _scheduler, _queues = admin_client
        resp = await client.post("/api/marshal/admin/pause")
        assert resp.status_code == 401

    async def test_returns_401_with_wrong_admin_token(self, admin_client):
        client, _scheduler, _queues = admin_client
        resp = await client.post(
            "/api/marshal/admin/pause",
            headers={"X-Marshal-Admin-Token": "wrong-tok"},
        )
        assert resp.status_code == 401

    async def test_returns_200_with_correct_token(self, admin_client):
        client, scheduler, queues = admin_client
        scheduler.pause.return_value = True
        queues.total_pending.return_value = 0

        resp = await client.post(
            "/api/marshal/admin/pause",
            headers={"X-Marshal-Admin-Token": ADMIN_TOK},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["drained_in_flight"] == 0
        assert body["queued_at_pause"] == 0
        assert "auto_resume_at" in body
        scheduler.pause.assert_awaited_once()

    async def test_passes_drain_timeout_and_auto_resume_to_scheduler(
        self, admin_client
    ):
        client, scheduler, _queues = admin_client
        scheduler.pause.return_value = True

        await client.post(
            "/api/marshal/admin/pause",
            headers={"X-Marshal-Admin-Token": ADMIN_TOK},
            json={"drain_timeout_s": 30, "auto_resume_after_seconds": 600},
        )
        scheduler.pause.assert_awaited_with(
            drain_timeout_s=30.0,
            auto_resume_after_seconds=600.0,
        )

    async def test_returns_409_on_drain_timeout(self, admin_client):
        client, scheduler, _queues = admin_client
        scheduler.pause.return_value = False  # drain timed out
        scheduler.in_flight_count.return_value = 2

        resp = await client.post(
            "/api/marshal/admin/pause",
            headers={"X-Marshal-Admin-Token": ADMIN_TOK},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body["in_flight"] == 2
        assert "auto_resume_at" in body

    async def test_returns_400_on_invalid_json_body(self, admin_client):
        client, _scheduler, _queues = admin_client
        resp = await client.post(
            "/api/marshal/admin/pause",
            headers={"X-Marshal-Admin-Token": ADMIN_TOK},
            content=b"not-json{",
        )
        assert resp.status_code == 400


class TestAdminResumeEndpoint:
    async def test_returns_404_when_disabled(self):
        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        config = MarshalConfig(
            scheduler=SchedulerConfig(benchmark_on_startup=False),
        )
        app = create_app(config)

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch("ollama_marshal.server.ModelLifecycle", return_value=lifecycle),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with (
                LifespanManager(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client,
            ):
                resp = await client.post("/api/marshal/admin/resume")

        assert resp.status_code == 404

    async def test_returns_401_without_admin_token(self, admin_client):
        client, _scheduler, _queues = admin_client
        resp = await client.post("/api/marshal/admin/resume")
        assert resp.status_code == 401

    async def test_calls_scheduler_resume_and_returns_queue_depth(self, admin_client):
        client, scheduler, queues = admin_client
        queues.total_pending.return_value = 7

        resp = await client.post(
            "/api/marshal/admin/resume",
            headers={"X-Marshal-Admin-Token": ADMIN_TOK},
        )
        assert resp.status_code == 200
        assert resp.json() == {"queue_depth": 7}
        scheduler.resume.assert_called_once()


# ---------------------------------------------------------------------------
# Debug endpoint
# ---------------------------------------------------------------------------


class TestDebugEndpoint:
    async def test_returns_404_when_disabled(self):
        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        config = MarshalConfig(
            scheduler=SchedulerConfig(benchmark_on_startup=False),
        )  # debug disabled by default
        app = create_app(config)

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch("ollama_marshal.server.ModelLifecycle", return_value=lifecycle),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with (
                LifespanManager(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client,
            ):
                resp = await client.get("/api/marshal/debug")

        assert resp.status_code == 404

    async def test_returns_payload_when_enabled(self):
        from ollama_marshal.config import DebugConfig

        memory, registry, lifecycle, scheduler, queues = _make_mock_components()
        scheduler.is_paused.return_value = False
        scheduler.in_flight_count.return_value = 3

        config = MarshalConfig(
            scheduler=SchedulerConfig(benchmark_on_startup=False),
            debug=DebugConfig(endpoint_enabled=True),
        )
        app = create_app(config)

        with (
            patch("ollama_marshal.server.MemoryManager", return_value=memory),
            patch("ollama_marshal.server.ModelRegistry", return_value=registry),
            patch("ollama_marshal.server.ModelLifecycle", return_value=lifecycle),
            patch("ollama_marshal.server.Scheduler", return_value=scheduler),
            patch("ollama_marshal.server.ModelQueues", return_value=queues),
        ):
            async with (
                LifespanManager(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client,
            ):
                # Lifespan resets metrics from disk on startup; mutate
                # AFTER lifespan to capture the test's intended values
                # in the running scheduler.
                scheduler.metrics.requests_served = 100
                scheduler.metrics.evictions = 5
                scheduler.metrics.reload_count = 2
                scheduler.metrics.unexpected_unloads = 1
                resp = await client.get("/api/marshal/debug")

        assert resp.status_code == 200
        body = resp.json()
        assert body["metrics"]["requests_served"] == 100
        assert body["metrics"]["evictions"] == 5
        assert body["metrics"]["reload_count"] == 2
        assert body["metrics"]["unexpected_unloads"] == 1
        assert body["scheduler"]["is_paused"] is False
        assert body["scheduler"]["in_flight_count"] == 3
