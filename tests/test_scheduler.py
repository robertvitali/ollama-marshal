from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ollama_marshal.config import MarshalConfig, Priority, ProgramConfig
from ollama_marshal.queue import ModelQueues, RequestEnvelope
from ollama_marshal.scheduler import (
    BurstHints,
    InflightTracker,
    Scheduler,
    SchedulerMetrics,
)

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
# SchedulerMetrics persistence
# ---------------------------------------------------------------------------


class TestSchedulerMetricsPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "metrics.json"
        m = SchedulerMetrics(
            requests_served=42, model_swaps=7, evictions=3, total_wait_ms=12345.6
        )
        m.save_to(path)
        loaded = SchedulerMetrics.load_from(path)
        assert loaded.requests_served == 42
        assert loaded.model_swaps == 7
        assert loaded.evictions == 3
        assert loaded.total_wait_ms == 12345.6

    def test_load_missing_file_returns_fresh(self, tmp_path):
        loaded = SchedulerMetrics.load_from(tmp_path / "nonexistent.json")
        assert loaded.requests_served == 0
        assert loaded.model_swaps == 0

    def test_load_corrupt_json_returns_fresh(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text("not valid json {{{")
        loaded = SchedulerMetrics.load_from(path)
        assert loaded.requests_served == 0

    def test_load_wrong_shape_returns_fresh(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps([1, 2, 3]))  # list, not dict
        loaded = SchedulerMetrics.load_from(path)
        assert loaded.requests_served == 0

    def test_load_schema_version_mismatch_returns_fresh(self, tmp_path):
        # Old or future schema version — refuse to load and start fresh.
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"schema_version": 999, "requests_served": 99}))
        loaded = SchedulerMetrics.load_from(path)
        assert loaded.requests_served == 0  # fresh, didn't trust the data

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "metrics.json"
        SchedulerMetrics(requests_served=5).save_to(path)
        assert path.exists()
        assert json.loads(path.read_text())["requests_served"] == 5

    def test_save_io_error_logs_not_raises(self, tmp_path, monkeypatch):
        # Simulate a disk-full or permission-denied scenario.
        path = tmp_path / "metrics.json"

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        # Must not raise — best-effort persistence.
        SchedulerMetrics(requests_served=1).save_to(path)

    def test_to_json_dict_omits_started_at(self):
        # started_at is process-local, never persisted.
        d = SchedulerMetrics(started_at=999.0).to_json_dict()
        assert "started_at" not in d
        assert "schema_version" in d


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

    @patch("ollama_marshal.scheduler.forward_request", new_callable=AsyncMock)
    async def test_records_program_in_active_programs(self, mock_forward):
        # Successful dispatch should stamp the model -> program_id in
        # _active_programs so the dashboard/status payload can show
        # which programs are using each loaded model.
        mock_forward.return_value = MagicMock()
        sched = _make_scheduler()

        e1 = _make_envelope(program_id="program-alpha")
        e1.model = "llama3:latest"
        e2 = _make_envelope(program_id="program-beta")
        e2.model = "llama3:latest"

        await sched._forward_single(e1)
        await sched._forward_single(e2)

        assert sched.active_programs_by_model() == {
            "llama3:latest": ["program-alpha", "program-beta"],
        }

    @patch("ollama_marshal.scheduler.forward_request", new_callable=AsyncMock)
    async def test_active_programs_not_recorded_on_failure(self, mock_forward):
        # A failed dispatch must not pollute the active-programs map.
        mock_forward.side_effect = RuntimeError("boom")
        sched = _make_scheduler()

        envelope = _make_envelope(program_id="program-alpha")
        await sched._forward_single(envelope)

        assert sched.active_programs_by_model() == {}


class TestActiveProgramsAccessor:
    async def test_returns_empty_when_nothing_loaded(self):
        sched = _make_scheduler()
        assert sched.active_programs_by_model() == {}

    async def test_skips_models_with_no_programs(self):
        # Defensive: if a model entry exists but is empty, it shouldn't
        # leak through as `model: []` — those models are filtered out.
        sched = _make_scheduler()
        sched._active_programs["empty-model"] = {}
        sched._active_programs["used-model"] = {"prog-a": 1.0, "prog-b": 2.0}
        assert sched.active_programs_by_model() == {
            "used-model": ["prog-a", "prog-b"],
        }


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


# ---------------------------------------------------------------------------
# _idle_evict_unused_models  (time-based eviction added in v0.2.0)
# ---------------------------------------------------------------------------


class TestIdleEvictUnusedModels:
    async def test_disabled_when_threshold_zero(self):
        cfg = _make_config(**{"scheduler.idle_eviction_minutes": 0})
        sched = _make_scheduler(config=cfg)
        sched.memory.get_loaded_models.return_value = {"foo": object()}
        # Even with old activity, threshold=0 disables time-eviction.
        sched._last_activity["foo"] = time.monotonic() - 99999
        await sched._idle_evict_unused_models()
        sched.lifecycle.unload.assert_not_called()

    async def test_evicts_after_threshold(self):
        cfg = _make_config(**{"scheduler.idle_eviction_minutes": 15})
        sched = _make_scheduler(config=cfg)
        sched.memory.get_loaded_models.return_value = {"foo": object()}
        # Mark "foo" as idle for 16 minutes (threshold is 15).
        sched._last_activity["foo"] = time.monotonic() - (16 * 60)
        # Real empty ModelQueues — pending_count returns 0 naturally,
        # so the scheduler's "no pending requests" guard is satisfied.

        await sched._idle_evict_unused_models()

        sched.lifecycle.unload.assert_awaited_once_with("foo")
        # Eviction counter incremented + activity entry removed on success.
        assert sched.metrics.evictions == 1
        assert "foo" not in sched._last_activity

    async def test_skips_models_with_pending_requests(self):
        cfg = _make_config(**{"scheduler.idle_eviction_minutes": 15})
        sched = _make_scheduler(config=cfg)
        sched.memory.get_loaded_models.return_value = {"foo": object()}
        sched._last_activity["foo"] = time.monotonic() - (16 * 60)
        # Real ModelQueues with 2 envelopes for "foo" — pending_count
        # naturally returns 2, exercising the "skip if pending > 0" path
        # without mocking an internal method.
        await sched.queues.enqueue(_make_envelope(model="foo"))
        await sched.queues.enqueue(_make_envelope(model="foo"))

        await sched._idle_evict_unused_models()

        sched.lifecycle.unload.assert_not_called()
        assert sched.metrics.evictions == 0

    async def test_first_seen_models_get_full_idle_window(self):
        cfg = _make_config(**{"scheduler.idle_eviction_minutes": 15})
        sched = _make_scheduler(config=cfg)
        sched.memory.get_loaded_models.return_value = {"new-model": object()}
        # No prior _last_activity entry — should be initialized, not evicted.
        await sched._idle_evict_unused_models()
        sched.lifecycle.unload.assert_not_called()
        assert "new-model" in sched._last_activity

    async def test_below_threshold_keeps_loaded(self):
        cfg = _make_config(**{"scheduler.idle_eviction_minutes": 15})
        sched = _make_scheduler(config=cfg)
        sched.memory.get_loaded_models.return_value = {"foo": object()}
        # Idle for 5 minutes — well under 15.
        sched._last_activity["foo"] = time.monotonic() - (5 * 60)
        await sched._idle_evict_unused_models()
        sched.lifecycle.unload.assert_not_called()

    async def test_only_evicts_one_per_tick(self):
        cfg = _make_config(**{"scheduler.idle_eviction_minutes": 15})
        sched = _make_scheduler(config=cfg)
        sched.memory.get_loaded_models.return_value = {
            "foo": object(),
            "bar": object(),
        }
        old_ts = time.monotonic() - (20 * 60)
        sched._last_activity["foo"] = old_ts
        sched._last_activity["bar"] = old_ts
        # Real empty ModelQueues — pending_count returns 0 for both models.

        await sched._idle_evict_unused_models()

        # Loop breaks after first eviction. Next tick handles the rest.
        assert sched.lifecycle.unload.await_count == 1


# ---------------------------------------------------------------------------
# BurstHints — X-Burst-Size header storage with TTL-based expiry
# ---------------------------------------------------------------------------


class TestBurstHintsRecord:
    def test_records_hint_under_cap(self):
        hints = BurstHints()
        result = hints.record("ai-portfolio", "qwen3.5:9b", 10, max_skips=3, now=100.0)
        # Cap = 3 * 4 = 12, so 10 fits and is stored verbatim.
        assert result == 10

    def test_caps_at_max_skips_times_multiplier(self):
        hints = BurstHints()
        # max_skips=3 → cap = 3*4 = 12. Asking for 999 gets clamped.
        result = hints.record("attacker", "model", 999, max_skips=3)
        assert result == 12

    def test_zero_or_negative_ignored(self):
        hints = BurstHints()
        assert hints.record("p", "m", 0, max_skips=3) == 0
        assert hints.record("p", "m", -5, max_skips=3) == 0
        # Nothing stored.
        assert hints.boost_for_model("m") == 0

    def test_empty_program_or_model_ignored(self):
        hints = BurstHints()
        assert hints.record("", "m", 5, max_skips=3) == 0
        assert hints.record("p", "", 5, max_skips=3) == 0
        assert hints.boost_for_model("m") == 0

    def test_refresh_replaces_previous(self):
        hints = BurstHints()
        hints.record("p", "m", 5, max_skips=3, now=100.0)
        # Recording again with a different value replaces, not adds.
        hints.record("p", "m", 8, max_skips=3, now=110.0)
        assert hints.boost_for_model("m", now=110.0) == 8


class TestBurstHintsBoostQueries:
    def test_boost_for_model_sums_across_programs(self):
        hints = BurstHints()
        hints.record("ai-email", "qwen3.5:9b", 5, max_skips=3, now=100.0)
        hints.record("ai-portfolio", "qwen3.5:9b", 7, max_skips=3, now=100.0)
        # Two programs targeting the same model — boosts add.
        assert hints.boost_for_model("qwen3.5:9b", now=101.0) == 12

    def test_boost_for_model_ignores_other_models(self):
        hints = BurstHints()
        hints.record("p", "model-a", 5, max_skips=3, now=100.0)
        hints.record("p", "model-b", 9, max_skips=3, now=100.0)
        assert hints.boost_for_model("model-a", now=101.0) == 5
        assert hints.boost_for_model("model-b", now=101.0) == 9

    def test_boost_returns_zero_after_expiry(self):
        # TTL is 30s by default. now=200 is well past 100+30=130.
        hints = BurstHints()
        hints.record("p", "m", 5, max_skips=3, now=100.0)
        assert hints.boost_for_model("m", now=200.0) == 0

    def test_all_boosts_by_model(self):
        hints = BurstHints()
        hints.record("p1", "model-a", 3, max_skips=3, now=100.0)
        hints.record("p2", "model-a", 4, max_skips=3, now=100.0)
        hints.record("p1", "model-b", 5, max_skips=3, now=100.0)
        result = hints.all_boosts_by_model(now=101.0)
        assert result == {"model-a": 7, "model-b": 5}


class TestBurstHintsExpiry:
    def test_prune_removes_expired_keeps_live(self):
        hints = BurstHints()
        hints.record("p", "old", 5, max_skips=3, now=10.0)  # expires at 40
        hints.record("p", "fresh", 7, max_skips=3, now=100.0)  # expires at 130
        removed = hints.prune_expired(now=110.0)
        assert removed == 1
        assert hints.boost_for_model("old", now=110.0) == 0
        assert hints.boost_for_model("fresh", now=110.0) == 7

    def test_prune_no_op_when_all_live(self):
        hints = BurstHints()
        hints.record("p", "m", 5, max_skips=3, now=100.0)
        assert hints.prune_expired(now=120.0) == 0  # within TTL

    def test_prune_empty_safe(self):
        hints = BurstHints()
        assert hints.prune_expired(now=100.0) == 0

    def test_custom_ttl(self):
        hints = BurstHints(ttl_s=5.0)
        hints.record("p", "m", 5, max_skips=3, now=100.0)  # expires at 105
        # 4s later: still live.
        assert hints.boost_for_model("m", now=104.0) == 5
        # 6s later: expired.
        assert hints.boost_for_model("m", now=106.0) == 0


class TestBurstHintsIntegration:
    """End-to-end: scheduler eviction scoring picks up burst boosts."""

    async def test_burst_protects_from_eviction(self):
        sched = _make_scheduler()
        # No actual envelopes for "burst-protected", so pending=0 normally.
        # But a burst hint of 50 should make it the LEAST evictable model.
        # max_skips=15 → cap is 60, so the requested 50 stays uncapped.
        sched.burst_hints.record("ai-portfolio", "burst-protected", 50, max_skips=15)

        memory = sched.memory
        memory.get_loaded_models.return_value = {
            "burst-protected": object(),
            "low-priority": object(),
        }

        # The pending_by_model + burst_boosts dict that _evict_one builds
        # should make low-priority the eviction target. Mock this by
        # capturing what gets passed to memory.get_eviction_candidates.
        captured: dict = {}

        def capture(pending_counts, program_priorities):
            captured["pending"] = dict(pending_counts)
            return ["low-priority", "burst-protected"]

        memory.get_eviction_candidates.side_effect = capture
        sched.queues.pending_by_model = AsyncMock(return_value={})

        await sched._evict_one(needed_for="other-model")

        # Burst boost must have been added before scoring.
        assert captured["pending"]["burst-protected"] == 50
        # And the chosen target was low-priority (not the burst one).
        sched.lifecycle.unload.assert_awaited_with("low-priority")


# ---------------------------------------------------------------------------
# InflightTracker — per-model dispatch concurrency gate
# ---------------------------------------------------------------------------


class TestInflightTracker:
    def test_default_at_least_one(self):
        # 0 or negative would deadlock all dispatch — clamp to 1.
        assert InflightTracker(0).parallel_per_model == 1
        assert InflightTracker(-5).parallel_per_model == 1

    def test_creates_semaphore_lazily(self):
        tracker = InflightTracker(4)
        sem_a = tracker.semaphore_for("model-a")
        sem_b = tracker.semaphore_for("model-b")
        # Same model returns the same semaphore — that is the whole point.
        assert tracker.semaphore_for("model-a") is sem_a
        assert sem_a is not sem_b

    async def test_semaphore_caps_concurrency(self):
        tracker = InflightTracker(2)
        sem = tracker.semaphore_for("model")

        in_flight = 0
        peak = 0

        async def worker():
            nonlocal in_flight, peak
            async with sem:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        # 5 workers but cap is 2 — peak should never exceed 2.
        await asyncio.gather(*(worker() for _ in range(5)))
        assert peak == 2


# ---------------------------------------------------------------------------
# _process_batch with parallel_per_model
# ---------------------------------------------------------------------------


class TestProcessBatchParallelism:
    """Verify dispatch concurrency matches scheduler.parallel_per_model."""

    async def test_default_serializes(self):
        # parallel_per_model=1 (default) → at most 1 in flight at a time.
        sched = _make_scheduler()
        # Five envelopes all for the same model.
        batch = [_make_envelope(model="m") for _ in range(5)]

        in_flight = 0
        peak = 0

        async def fake_forward(env):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1

        with patch.object(sched, "_forward_single", side_effect=fake_forward):
            await sched._process_batch(batch)

        assert peak == 1  # current behavior: serial

    async def test_parallel_4_runs_4_concurrent(self):
        cfg = _make_config(**{"scheduler.parallel_per_model": 4})
        sched = _make_scheduler(config=cfg)
        batch = [_make_envelope(model="m") for _ in range(10)]

        in_flight = 0
        peak = 0

        async def fake_forward(env):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1

        with patch.object(sched, "_forward_single", side_effect=fake_forward):
            await sched._process_batch(batch)

        # Cap at parallel_per_model. (Could be < 4 if scheduling jitter
        # never lined up 4-at-a-time, but with 10 envelopes and a real
        # event loop they should overlap.)
        assert peak <= 4
        assert peak >= 2  # at least some parallelism observed

    async def test_embeddings_always_concurrent(self):
        # Embeddings ignore the gate — they should fan out in parallel
        # regardless of parallel_per_model setting.
        sched = _make_scheduler()  # default parallel_per_model=1
        batch = [
            _make_envelope(model="m", endpoint="/api/embeddings") for _ in range(5)
        ]

        in_flight = 0
        peak = 0

        async def fake_forward(env):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1

        with patch.object(sched, "_forward_single", side_effect=fake_forward):
            await sched._process_batch(batch)

        assert peak == 5  # all 5 embeddings ran concurrently

    async def test_semaphore_releases_on_exception(self):
        # If _forward_single raises, the semaphore must still release;
        # otherwise a single failure permanently shrinks the cap.
        cfg = _make_config(**{"scheduler.parallel_per_model": 2})
        sched = _make_scheduler(config=cfg)
        batch = [_make_envelope(model="m") for _ in range(4)]

        async def bad_forward(env):
            raise RuntimeError("boom")

        with patch.object(sched, "_forward_single", side_effect=bad_forward):
            # asyncio.gather(return_exceptions=True) swallows them.
            await sched._process_batch(batch)

        # All 4 should have been attempted (semaphore released after each).
        # Verify the semaphore is back to fully available (2/2 free).
        sem = sched.inflight.semaphore_for("m")
        # Both slots free → can acquire twice without blocking.
        async with sem, sem:
            pass
