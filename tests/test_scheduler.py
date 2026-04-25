from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

from ollama_marshal.config import MarshalConfig, Priority, ProgramConfig
from ollama_marshal.queue import ModelQueues, RequestEnvelope
from ollama_marshal.scheduler import Scheduler, SchedulerMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    cfg = MarshalConfig()
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], v)
    return cfg


def _make_envelope(model="llama3:latest", program_id="default", **kw):
    return RequestEnvelope(
        model=model,
        program_id=program_id,
        request_body={"model": model},
        endpoint=kw.pop("endpoint", "/api/chat"),
        stream=kw.pop("stream", False),
        **kw,
    )


def _make_scheduler(
    queues=None,
    memory=None,
    registry=None,
    lifecycle=None,
    config=None,
):
    if queues is None:
        queues = ModelQueues()
    if memory is None:
        memory = MagicMock()
        memory.get_loaded_models.return_value = {}
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = True
        memory.available_vram.return_value = 50 * 1024**3
        memory.get_eviction_candidates.return_value = []
        memory.refresh = AsyncMock()
    if registry is None:
        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)
    if lifecycle is None:
        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)
        lifecycle.unload = AsyncMock(return_value=True)
    if config is None:
        config = MarshalConfig()
    return Scheduler(
        queues=queues,
        memory=memory,
        registry=registry,
        lifecycle=lifecycle,
        config=config,
    )


# ---------------------------------------------------------------------------
# SchedulerMetrics
# ---------------------------------------------------------------------------


class TestSchedulerMetrics:
    def test_default_values(self):
        m = SchedulerMetrics()
        assert m.requests_served == 0
        assert m.model_swaps == 0
        assert m.evictions == 0
        assert m.total_wait_ms == 0.0
        assert isinstance(m.started_at, float)

    def test_average_wait_ms_zero_requests(self):
        m = SchedulerMetrics()
        assert m.average_wait_ms == 0.0

    def test_average_wait_ms_nonzero(self):
        m = SchedulerMetrics(requests_served=4, total_wait_ms=200.0)
        assert m.average_wait_ms == 50.0

    def test_started_at_is_monotonic(self):
        before = time.monotonic()
        m = SchedulerMetrics()
        after = time.monotonic()
        assert before <= m.started_at <= after


# ---------------------------------------------------------------------------
# _forward_loaded_model_requests
# ---------------------------------------------------------------------------


class TestForwardLoadedModelRequests:
    async def test_forwards_when_model_loaded(self):
        queues = ModelQueues()
        envelope = _make_envelope()
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.get_loaded_models.return_value = {"llama3:latest": MagicMock()}

        sched = _make_scheduler(queues=queues, memory=memory)

        with patch.object(sched, "_process_batch", new_callable=AsyncMock) as mock_pb:
            await sched._forward_loaded_model_requests()
            mock_pb.assert_called_once()
            batch = mock_pb.call_args[0][0]
            assert len(batch) == 1
            assert batch[0] is envelope

    async def test_skips_when_no_pending(self):
        queues = ModelQueues()
        memory = MagicMock()
        memory.get_loaded_models.return_value = {"llama3:latest": MagicMock()}

        sched = _make_scheduler(queues=queues, memory=memory)

        with patch.object(sched, "_process_batch", new_callable=AsyncMock) as mock_pb:
            await sched._forward_loaded_model_requests()
            mock_pb.assert_not_called()

    async def test_skips_unloaded_models(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="unloaded:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.get_loaded_models.return_value = {"llama3:latest": MagicMock()}

        sched = _make_scheduler(queues=queues, memory=memory)

        with patch.object(sched, "_process_batch", new_callable=AsyncMock) as mock_pb:
            await sched._forward_loaded_model_requests()
            mock_pb.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_critical_preemption
# ---------------------------------------------------------------------------


class TestHandleCriticalPreemption:
    async def test_preempts_for_critical_program(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="critical-model", program_id="urgent")
        await queues.enqueue(envelope)

        config = MarshalConfig(
            programs={
                "default": ProgramConfig(),
                "urgent": ProgramConfig(priority=Priority.CRITICAL),
            }
        )

        memory = MagicMock()
        memory.is_loaded.return_value = False

        sched = _make_scheduler(queues=queues, memory=memory, config=config)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            mock_eml.return_value = True
            await sched._handle_critical_preemption()
            mock_eml.assert_called_once_with("critical-model")

    async def test_skips_normal_priority(self):
        queues = ModelQueues()
        envelope = _make_envelope(program_id="normal_prog")
        await queues.enqueue(envelope)

        config = MarshalConfig()
        memory = MagicMock()
        memory.is_loaded.return_value = False

        sched = _make_scheduler(queues=queues, memory=memory, config=config)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            await sched._handle_critical_preemption()
            mock_eml.assert_not_called()

    async def test_skips_already_loaded_critical(self):
        queues = ModelQueues()
        envelope = _make_envelope(program_id="urgent")
        await queues.enqueue(envelope)

        config = MarshalConfig(
            programs={
                "default": ProgramConfig(),
                "urgent": ProgramConfig(priority=Priority.CRITICAL),
            }
        )

        memory = MagicMock()
        memory.is_loaded.return_value = True

        sched = _make_scheduler(queues=queues, memory=memory, config=config)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            await sched._handle_critical_preemption()
            mock_eml.assert_not_called()

    async def test_only_handles_one_per_tick(self):
        queues = ModelQueues()
        e1 = _make_envelope(model="m1", program_id="urgent")
        e2 = _make_envelope(model="m2", program_id="urgent")
        await queues.enqueue(e1)
        await queues.enqueue(e2)

        config = MarshalConfig(
            programs={
                "default": ProgramConfig(),
                "urgent": ProgramConfig(priority=Priority.CRITICAL),
            }
        )

        memory = MagicMock()
        memory.is_loaded.return_value = False

        sched = _make_scheduler(queues=queues, memory=memory, config=config)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            mock_eml.return_value = True
            await sched._handle_critical_preemption()
            assert mock_eml.call_count == 1


# ---------------------------------------------------------------------------
# _handle_unskippable_requests
# ---------------------------------------------------------------------------


class TestHandleUnskippableRequests:
    async def test_force_loads_after_skip_limit(self):
        queues = ModelQueues()
        envelope = _make_envelope()
        # Exceed skip limit
        for _ in range(5):
            envelope.increment_skip()
        await queues.enqueue(envelope)

        config = _make_config()
        memory = MagicMock()
        memory.is_loaded.return_value = False

        sched = _make_scheduler(queues=queues, memory=memory, config=config)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            mock_eml.return_value = True
            await sched._handle_unskippable_requests()
            mock_eml.assert_called_once_with("llama3:latest")

    async def test_skips_if_below_limit(self):
        queues = ModelQueues()
        envelope = _make_envelope()
        await queues.enqueue(envelope)

        config = _make_config()
        memory = MagicMock()
        memory.is_loaded.return_value = False

        sched = _make_scheduler(queues=queues, memory=memory, config=config)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            await sched._handle_unskippable_requests()
            mock_eml.assert_not_called()

    async def test_skips_already_loaded(self):
        queues = ModelQueues()
        envelope = _make_envelope()
        for _ in range(5):
            envelope.increment_skip()
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = True

        sched = _make_scheduler(queues=queues, memory=memory)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            await sched._handle_unskippable_requests()
            mock_eml.assert_not_called()

    async def test_only_loads_one_per_tick(self):
        queues = ModelQueues()
        e1 = _make_envelope(model="m1")
        e2 = _make_envelope(model="m2")
        for e in (e1, e2):
            for _ in range(5):
                e.increment_skip()
        await queues.enqueue(e1)
        await queues.enqueue(e2)

        memory = MagicMock()
        memory.is_loaded.return_value = False

        sched = _make_scheduler(queues=queues, memory=memory)

        with patch.object(
            sched, "_ensure_model_loaded", new_callable=AsyncMock
        ) as mock_eml:
            mock_eml.return_value = True
            await sched._handle_unskippable_requests()
            assert mock_eml.call_count == 1


# ---------------------------------------------------------------------------
# _bin_pack_models
# ---------------------------------------------------------------------------


class TestBinPackModels:
    async def test_loads_model_that_fits(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="small:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = True
        memory.available_vram.return_value = 50 * 1024**3
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=2 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)

        sched = _make_scheduler(
            queues=queues,
            memory=memory,
            registry=registry,
            lifecycle=lifecycle,
        )

        await sched._bin_pack_models()

        lifecycle.preload.assert_called_once_with("small:latest")
        assert sched.metrics.model_swaps == 1
        memory.refresh.assert_called_once()

    async def test_skips_model_that_does_not_fit(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="huge:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = False
        memory.available_vram.return_value = 1 * 1024**3

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=100 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock()

        sched = _make_scheduler(
            queues=queues,
            memory=memory,
            registry=registry,
            lifecycle=lifecycle,
        )

        await sched._bin_pack_models()

        lifecycle.preload.assert_not_called()
        assert sched.metrics.model_swaps == 0

    async def test_skips_already_loaded(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="loaded:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = True

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock()

        sched = _make_scheduler(
            queues=queues,
            memory=memory,
            registry=registry,
            lifecycle=lifecycle,
        )

        await sched._bin_pack_models()

        registry.get_or_estimate_size.assert_not_called()
        lifecycle.preload.assert_not_called()

    async def test_preload_failure_no_swap_increment(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="fail:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = True
        memory.available_vram.return_value = 50 * 1024**3
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=2 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=False)

        sched = _make_scheduler(
            queues=queues,
            memory=memory,
            registry=registry,
            lifecycle=lifecycle,
        )

        await sched._bin_pack_models()

        assert sched.metrics.model_swaps == 0
        memory.refresh.assert_not_called()

    async def test_no_pending_requests(self):
        queues = ModelQueues()
        memory = MagicMock()
        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock()
        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock()

        sched = _make_scheduler(
            queues=queues,
            memory=memory,
            registry=registry,
            lifecycle=lifecycle,
        )

        await sched._bin_pack_models()

        lifecycle.preload.assert_not_called()


# ---------------------------------------------------------------------------
# _bin_pack_models skip counter behavior
# ---------------------------------------------------------------------------


class TestBinPackSkipCounters:
    async def test_increments_skips_for_models_that_dont_fit(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="big:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = False
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=50 * 1024**3)

        sched = _make_scheduler(queues=queues, memory=memory, registry=registry)

        assert envelope.skip_count == 0
        await sched._bin_pack_models()
        assert envelope.skip_count == 1

    async def test_does_not_increment_for_loaded_models(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="loaded:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = True

        sched = _make_scheduler(queues=queues, memory=memory)

        await sched._bin_pack_models()
        assert envelope.skip_count == 0

    async def test_does_not_increment_for_models_that_fit(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="small:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = True
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=1 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)

        sched = _make_scheduler(
            queues=queues,
            memory=memory,
            registry=registry,
            lifecycle=lifecycle,
        )

        await sched._bin_pack_models()
        assert envelope.skip_count == 0


# ---------------------------------------------------------------------------
# _ensure_model_loaded
# ---------------------------------------------------------------------------


class TestEnsureModelLoaded:
    async def test_already_loaded_returns_true(self):
        memory = MagicMock()
        memory.is_loaded.return_value = True

        sched = _make_scheduler(memory=memory)

        result = await sched._ensure_model_loaded("llama3:latest")
        assert result is True

    async def test_loads_when_fits(self):
        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = True
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)

        sched = _make_scheduler(memory=memory, registry=registry, lifecycle=lifecycle)

        result = await sched._ensure_model_loaded("llama3:latest")
        assert result is True
        lifecycle.preload.assert_called_once_with("llama3:latest")
        assert sched.metrics.model_swaps == 1

    async def test_evicts_when_needed(self):
        call_count = 0

        def can_fit_side_effect(size):
            nonlocal call_count
            call_count += 1
            # First call returns False (need eviction), second returns True
            return call_count > 1

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.side_effect = can_fit_side_effect
        memory.refresh = AsyncMock()
        memory.available_vram.return_value = 2 * 1024**3

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)

        sched = _make_scheduler(memory=memory, registry=registry, lifecycle=lifecycle)

        with patch.object(sched, "_evict_one", new_callable=AsyncMock) as mock_evict:
            mock_evict.return_value = True
            result = await sched._ensure_model_loaded("big:latest")
            assert result is True
            mock_evict.assert_called_once_with("big:latest")

    async def test_returns_false_when_cannot_evict(self):
        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = False
        memory.available_vram.return_value = 0

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)

        sched = _make_scheduler(memory=memory, registry=registry)

        with patch.object(sched, "_evict_one", new_callable=AsyncMock) as mock_evict:
            mock_evict.return_value = False
            result = await sched._ensure_model_loaded("big:latest")
            assert result is False

    async def test_preload_failure_returns_false(self):
        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = True
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=False)

        sched = _make_scheduler(memory=memory, registry=registry, lifecycle=lifecycle)

        result = await sched._ensure_model_loaded("fail:latest")
        assert result is False
        assert sched.metrics.model_swaps == 0


# ---------------------------------------------------------------------------
# _evict_one
# ---------------------------------------------------------------------------


class TestEvictOne:
    async def test_picks_best_candidate(self):
        queues = ModelQueues()

        memory = MagicMock()
        memory.get_loaded_models.return_value = {
            "victim:latest": MagicMock(),
            "keeper:latest": MagicMock(),
        }
        memory.get_eviction_candidates.return_value = [
            "victim:latest",
            "keeper:latest",
        ]
        memory.refresh = AsyncMock()

        lifecycle = MagicMock()
        lifecycle.unload = AsyncMock(return_value=True)

        sched = _make_scheduler(queues=queues, memory=memory, lifecycle=lifecycle)

        result = await sched._evict_one("new:latest")
        assert result is True
        lifecycle.unload.assert_called_once_with("victim:latest")
        assert sched.metrics.evictions == 1

    async def test_excludes_needed_model(self):
        queues = ModelQueues()

        memory = MagicMock()
        memory.get_loaded_models.return_value = {
            "new:latest": MagicMock(),
        }
        memory.get_eviction_candidates.return_value = ["new:latest"]
        memory.refresh = AsyncMock()

        lifecycle = MagicMock()
        lifecycle.unload = AsyncMock()

        sched = _make_scheduler(queues=queues, memory=memory, lifecycle=lifecycle)

        result = await sched._evict_one("new:latest")
        assert result is False
        lifecycle.unload.assert_not_called()

    async def test_no_candidates_returns_false(self):
        queues = ModelQueues()

        memory = MagicMock()
        memory.get_loaded_models.return_value = {}
        memory.get_eviction_candidates.return_value = []

        sched = _make_scheduler(queues=queues, memory=memory)

        result = await sched._evict_one("new:latest")
        assert result is False

    async def test_drains_before_evict(self):
        queues = ModelQueues()
        envelope = _make_envelope(model="victim:latest")
        await queues.enqueue(envelope)

        memory = MagicMock()
        memory.get_loaded_models.return_value = {
            "victim:latest": MagicMock(),
        }
        memory.get_eviction_candidates.return_value = ["victim:latest"]
        memory.refresh = AsyncMock()

        lifecycle = MagicMock()
        lifecycle.unload = AsyncMock(return_value=True)

        sched = _make_scheduler(queues=queues, memory=memory, lifecycle=lifecycle)

        with patch.object(sched, "_process_batch", new_callable=AsyncMock) as mock_pb:
            result = await sched._evict_one("new:latest")
            assert result is True
            mock_pb.assert_called_once()
            batch = mock_pb.call_args[0][0]
            assert len(batch) == 1
            assert batch[0] is envelope

    async def test_unload_failure(self):
        queues = ModelQueues()

        memory = MagicMock()
        memory.get_loaded_models.return_value = {
            "victim:latest": MagicMock(),
        }
        memory.get_eviction_candidates.return_value = ["victim:latest"]
        memory.refresh = AsyncMock()

        lifecycle = MagicMock()
        lifecycle.unload = AsyncMock(return_value=False)

        sched = _make_scheduler(queues=queues, memory=memory, lifecycle=lifecycle)

        result = await sched._evict_one("new:latest")
        assert result is False
        assert sched.metrics.evictions == 0


# ---------------------------------------------------------------------------
# _process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    async def test_embeddings_concurrently(self):
        e1 = _make_envelope(endpoint="/api/embeddings")
        e2 = _make_envelope(endpoint="/v1/embeddings")

        sched = _make_scheduler()
        call_order = []

        async def mock_forward(env):
            call_order.append(env.endpoint)

        with patch.object(sched, "_forward_single", side_effect=mock_forward):
            await sched._process_batch([e1, e2])

        assert "/api/embeddings" in call_order
        assert "/v1/embeddings" in call_order

    async def test_non_embeddings_sequentially(self):
        e1 = _make_envelope(endpoint="/api/chat")
        e2 = _make_envelope(endpoint="/api/generate")

        sched = _make_scheduler()
        call_order = []

        async def mock_forward(env):
            call_order.append(env.endpoint)

        with patch.object(sched, "_forward_single", side_effect=mock_forward):
            await sched._process_batch([e1, e2])

        assert call_order == ["/api/chat", "/api/generate"]

    async def test_mixed_batch(self):
        chat = _make_envelope(endpoint="/api/chat")
        embed = _make_envelope(endpoint="/api/embeddings")

        sched = _make_scheduler()
        forwarded = []

        async def mock_forward(env):
            forwarded.append(env.endpoint)

        with patch.object(sched, "_forward_single", side_effect=mock_forward):
            await sched._process_batch([chat, embed])

        # Chat processed first (sequential), then embeddings
        assert forwarded[0] == "/api/chat"
        assert "/api/embeddings" in forwarded

    async def test_empty_batch(self):
        sched = _make_scheduler()
        with patch.object(sched, "_forward_single", new_callable=AsyncMock) as mock_fw:
            await sched._process_batch([])
            mock_fw.assert_not_called()


# ---------------------------------------------------------------------------
# _forward_single
# ---------------------------------------------------------------------------


class TestForwardSingle:
    @patch("ollama_marshal.scheduler.forward_request", new_callable=AsyncMock)
    async def test_completes_envelope_on_success(self, mock_forward):
        mock_response = MagicMock()
        mock_forward.return_value = mock_response

        envelope = _make_envelope()
        sched = _make_scheduler()

        await sched._forward_single(envelope)

        assert envelope.response is mock_response
        assert envelope.done_event.is_set()
        assert sched.metrics.requests_served == 1
        assert sched.metrics.total_wait_ms >= 0

    @patch("ollama_marshal.scheduler.forward_request", new_callable=AsyncMock)
    async def test_fails_envelope_on_error(self, mock_forward):
        mock_forward.side_effect = RuntimeError("connection lost")

        envelope = _make_envelope()
        sched = _make_scheduler()

        await sched._forward_single(envelope)

        assert envelope.error is not None
        assert "connection lost" in str(envelope.error)
        assert envelope.done_event.is_set()
        assert sched.metrics.requests_served == 0

    @patch("ollama_marshal.scheduler.forward_request", new_callable=AsyncMock)
    async def test_tracks_wait_time(self, mock_forward):
        mock_forward.return_value = MagicMock()

        envelope = _make_envelope()
        sched = _make_scheduler()

        await sched._forward_single(envelope)

        assert sched.metrics.total_wait_ms >= 0


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    async def test_start_creates_task(self):
        sched = _make_scheduler()
        await sched.start()
        assert sched._task is not None
        assert sched._running is True
        # Clean up
        await sched.stop()

    async def test_stop_cancels_task(self):
        sched = _make_scheduler()
        await sched.start()
        task = sched._task
        await sched.stop()
        assert sched._task is None
        assert sched._running is False
        assert task.cancelled() or task.done()

    async def test_stop_without_start(self):
        sched = _make_scheduler()
        # Should not raise
        await sched.stop()
        assert sched._task is None
        assert sched._running is False


# ---------------------------------------------------------------------------
# _run and _tick
# ---------------------------------------------------------------------------


class TestRunAndTick:
    async def test_tick_calls_all_steps(self):
        sched = _make_scheduler()

        with (
            patch.object(
                sched,
                "_forward_loaded_model_requests",
                new_callable=AsyncMock,
            ) as mock_flm,
            patch.object(
                sched,
                "_handle_critical_preemption",
                new_callable=AsyncMock,
            ) as mock_hcp,
            patch.object(
                sched,
                "_handle_unskippable_requests",
                new_callable=AsyncMock,
            ) as mock_hur,
            patch.object(sched, "_bin_pack_models", new_callable=AsyncMock) as mock_bp,
        ):
            await sched._tick()
            mock_flm.assert_called_once()
            mock_hcp.assert_called_once()
            mock_hur.assert_called_once()
            mock_bp.assert_called_once()

    async def test_run_calls_tick(self):
        sched = _make_scheduler()
        tick_count = 0

        async def mock_tick():
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 2:
                sched._running = False

        with patch.object(sched, "_tick", side_effect=mock_tick):
            sched._running = True
            await sched._run()

        assert tick_count >= 2

    async def test_run_handles_tick_error(self):
        sched = _make_scheduler()
        call_count = 0

        async def mock_tick():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("boom")
            sched._running = False

        with patch.object(sched, "_tick", side_effect=mock_tick):
            sched._running = True
            await sched._run()

        # Should have continued after the error
        assert call_count >= 2
