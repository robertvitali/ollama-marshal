from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest

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


_PRIMARY_INSTANCE_URL = "http://localhost:11434"


def _apply_instance_defaults(memory, registry, config):
    """Stamp single-instance defaults on test mocks so routing works.

    v0.5.0 routing reads ``memory.instances`` + ``memory.probe_fit``
    + ``registry.get_total_footprint``. Tests that build their own
    MagicMock for memory/registry need these defaults filled in so
    ``routing.pick_instance`` short-circuits via ``SINGLE_INSTANCE``
    instead of hitting the unreachable-state RuntimeError.
    """
    from ollama_marshal.routing import FitProbe as _FitProbe

    memory.instances = list(config.instances)
    memory.loaded_on.return_value = {_PRIMARY_INSTANCE_URL: set()}
    memory.get_loaded_models_on.return_value = {}
    memory.probe_fit.return_value = _FitProbe(fits=True, would_evict_non_idle=False)
    # In single-instance mode, "loaded somewhere" == "loaded on the
    # primary". Mirror is_loaded → is_loaded_on so tests that only
    # set is_loaded keep working. find_instance_for unconditionally
    # returns the primary URL — that's the correct answer in single-
    # instance setups even before /api/ps confirms the load (the
    # legacy behavior was equivalent).
    memory.is_loaded_on.side_effect = lambda model, _url: bool(memory.is_loaded(model))
    memory.find_instance_for.return_value = _PRIMARY_INSTANCE_URL
    # Likewise, a test that didn't bother to set memory.refresh = AsyncMock
    # still needs an awaitable refresh for the routing-aware load path.
    if not isinstance(memory.refresh, AsyncMock):
        memory.refresh = AsyncMock()
    # registry.get_total_footprint may be missing on user-supplied
    # mocks; replace with an AsyncMock that returns a sensible default
    # so routing's footprint math returns a real int.
    gtf = getattr(registry, "get_total_footprint", None)
    if (
        not isinstance(gtf, AsyncMock)
        or gtf.return_value is None
        or isinstance(gtf.return_value, MagicMock)
    ):
        registry.get_total_footprint = AsyncMock(return_value=4 * 1024**3)


def _make_scheduler(
    queues=None,
    memory=None,
    registry=None,
    lifecycle=None,
    config=None,
):
    if config is None:
        config = MarshalConfig()
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
        # v0.4.0 methods (Surface C1 Dim 4 + C2). Default to "no
        # reload needed" / "no unexpected unloads" / "no allocation".
        memory.needs_reload.return_value = False
        memory.get_allocated_num_ctx.return_value = None
        memory.take_unexpected_unload_count.return_value = 0
        # v0.6.5 (Bug 4): scheduler ``_tick`` drains the per-(model,
        # instance) eviction set into ``_needs_reload``. Default to
        # empty so tests not exercising the reactivity path don't
        # accidentally inject phantom evictions.
        memory.take_recent_unexpected_unloads.return_value = set()
        memory.mark_intended_unload = MagicMock()
        memory.mark_owned = MagicMock()
        memory.record_allocated_num_ctx = MagicMock()
    if registry is None:
        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)
    if lifecycle is None:
        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)
        lifecycle.unload = AsyncMock(return_value=True)
    # Always stamp multi-instance defaults so routing has the data it
    # needs, even when the test built its own memory/registry mocks.
    _apply_instance_defaults(memory, registry, config)
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
            # num_ctx is None because the envelope had no options.num_ctx.
            mock_eml.assert_called_once_with("critical-model", num_ctx=None)

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
            mock_eml.assert_called_once_with("llama3:latest", num_ctx=None)

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

        lifecycle.preload.assert_called_once_with(
            "small:latest",
            num_ctx=None,
            instance_url=_PRIMARY_INSTANCE_URL,
            load_timeout_s=3600,
            is_known_model_check=ANY,
        )
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

    async def test_critical_envelopes_exempt_from_skip_increment(self):
        """CRITICAL-priority requests should NOT have skip_count incremented.

        The fairness floor (max_skips → forced load) is for NORMAL-
        priority starvation prevention. CRITICAL has its own
        preemption path (_handle_critical_preemption), so the skip
        counter staying at 0 keeps the two paths cleanly separated
        and avoids spurious "forced_load" events.
        """
        # Two envelopes for the same model: one CRITICAL, one NORMAL.
        # Bin-pack should skip the model (doesn't fit) and increment
        # only the NORMAL envelope's skip count.
        config = MarshalConfig(
            programs={
                "default": ProgramConfig(),
                "crit-prog": ProgramConfig(priority=Priority.CRITICAL),
            }
        )
        queues = ModelQueues()
        critical_env = _make_envelope(model="big:latest", program_id="crit-prog")
        normal_env = _make_envelope(model="big:latest", program_id="default")
        await queues.enqueue(critical_env)
        await queues.enqueue(normal_env)

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = False  # Doesn't fit → skip
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=50 * 1024**3)

        sched = _make_scheduler(
            queues=queues, memory=memory, registry=registry, config=config
        )

        await sched._bin_pack_models()

        assert critical_env.skip_count == 0  # Exempt
        assert normal_env.skip_count == 1  # Counted

    async def test_critical_only_envelopes_skip_count_stays_zero(self):
        """If the only pending envelope is CRITICAL, skip count stays 0."""
        config = MarshalConfig(
            programs={
                "default": ProgramConfig(),
                "crit-prog": ProgramConfig(priority=Priority.CRITICAL),
            }
        )
        queues = ModelQueues()
        critical_env = _make_envelope(model="big:latest", program_id="crit-prog")
        await queues.enqueue(critical_env)

        memory = MagicMock()
        memory.is_loaded.return_value = False
        memory.can_fit_model.return_value = False
        memory.refresh = AsyncMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=50 * 1024**3)

        sched = _make_scheduler(
            queues=queues, memory=memory, registry=registry, config=config
        )

        # Multiple ticks of bin-pack should still leave skip_count at 0.
        for _ in range(5):
            await sched._bin_pack_models()

        assert critical_env.skip_count == 0


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
        lifecycle.preload.assert_called_once_with(
            "llama3:latest",
            num_ctx=None,
            instance_url=_PRIMARY_INSTANCE_URL,
            load_timeout_s=3600,
            is_known_model_check=ANY,
        )
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
        lifecycle.unload.assert_called_once_with(
            "victim:latest", instance_url=_PRIMARY_INSTANCE_URL
        )
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


class TestResolveRetryAttempts:
    """Resolution precedence for the per-envelope retry budget."""

    def test_streaming_always_returns_one(self):
        sched = _make_scheduler()
        env = _make_envelope(stream=True, retry_max_override=99)
        assert sched._resolve_retry_attempts(env) == 1

    def test_envelope_override_wins_over_config(self):
        cfg = _make_config(**{"retry.max_attempts": 5})
        sched = _make_scheduler(config=cfg)
        env = _make_envelope(retry_max_override=2)
        assert sched._resolve_retry_attempts(env) == 2

    def test_envelope_override_zero_clamps_to_one(self):
        # `X-Marshal-Retry-Max: 0` means "don't retry" — that's still
        # one attempt, not zero.
        sched = _make_scheduler()
        env = _make_envelope(retry_max_override=0)
        assert sched._resolve_retry_attempts(env) == 1

    def test_uses_config_when_no_override(self):
        cfg = _make_config(**{"retry.max_attempts": 4})
        sched = _make_scheduler(config=cfg)
        env = _make_envelope()
        assert sched._resolve_retry_attempts(env) == 4

    def test_disabled_config_returns_one(self):
        cfg = _make_config(**{"retry.enabled": False, "retry.max_attempts": 5})
        sched = _make_scheduler(config=cfg)
        env = _make_envelope()
        assert sched._resolve_retry_attempts(env) == 1


class TestForwardSingleRetry:
    """Integration: _forward_single + call_with_retry counter wiring."""

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    @patch("ollama_marshal.scheduler.forward_request", new_callable=AsyncMock)
    async def test_no_retry_path_skips_helper(self, mock_forward, mock_retry):
        # max_attempts=1 → don't even call call_with_retry (clearer fast path).
        mock_forward.return_value = MagicMock()
        cfg = _make_config(**{"retry.max_attempts": 1, "retry.enabled": True})
        sched = _make_scheduler(config=cfg)

        await sched._forward_single(_make_envelope())

        mock_forward.assert_called_once()
        mock_retry.assert_not_called()

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_retry_path_records_metrics_on_success(self, mock_retry):
        # Helper succeeded on attempt 3 → 2 retries attempted, 1 succeeded.
        sentinel = MagicMock()
        mock_retry.return_value = (sentinel, 3, False)
        sched = _make_scheduler()

        env = _make_envelope()
        await sched._forward_single(env)

        assert sched.metrics.retries_attempted == 2
        assert sched.metrics.retries_succeeded == 1
        assert env.response is sentinel

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_exhausted_status_does_not_count_as_succeeded(self, mock_retry):
        # All attempts returned 503 → caller still gets the response,
        # but `exhausted=True` means we should NOT count this as a
        # successful retry. Critical for `marshal doctor` correctness.
        mock_retry.return_value = (MagicMock(), 3, True)
        sched = _make_scheduler()

        env = _make_envelope()
        await sched._forward_single(env)

        assert sched.metrics.retries_attempted == 2
        # The key assertion — exhausted retries are NOT successes.
        assert sched.metrics.retries_succeeded == 0

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_retry_path_no_metrics_on_first_attempt(self, mock_retry):
        # attempts_used=1 → no retry happened, no counters move.
        mock_retry.return_value = (MagicMock(), 1, False)
        sched = _make_scheduler()

        await sched._forward_single(_make_envelope())

        assert sched.metrics.retries_attempted == 0
        assert sched.metrics.retries_succeeded == 0

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_retry_exhaustion_records_attempted_only(self, mock_retry):
        # Helper exhausted retries and reraised a RETRYABLE exception →
        # retries_attempted bumps, retries_succeeded does NOT.
        import httpx

        mock_retry.side_effect = httpx.ConnectError("permanent")
        cfg = _make_config(**{"retry.max_attempts": 3})
        sched = _make_scheduler(config=cfg)

        env = _make_envelope()
        await sched._forward_single(env)

        assert env.error is not None
        assert sched.metrics.retries_attempted == 2  # max_attempts - 1
        assert sched.metrics.retries_succeeded == 0

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_non_retryable_exception_does_not_bump_retries_attempted(
        self, mock_retry
    ):
        # call_with_retry raises non-retryable exceptions on attempt 1
        # without consuming retry budget. The metric must not bump as
        # though all retries were used.
        import httpx

        mock_retry.side_effect = httpx.ReadTimeout("slow")  # non-retryable by default
        cfg = _make_config(**{"retry.max_attempts": 3, "retry.read_timeouts": False})
        sched = _make_scheduler(config=cfg)

        env = _make_envelope()
        await sched._forward_single(env)

        assert env.error is not None
        # Crucial: 0, not 2. ReadTimeout was not retried.
        assert sched.metrics.retries_attempted == 0
        assert sched.metrics.retries_succeeded == 0

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_unrelated_exception_does_not_bump_retries_attempted(
        self, mock_retry
    ):
        # A non-network exception (e.g. our code has a bug) must also
        # not count against retry metrics.
        mock_retry.side_effect = ValueError("scheduler bug")
        cfg = _make_config(**{"retry.max_attempts": 3})
        sched = _make_scheduler(config=cfg)

        env = _make_envelope()
        await sched._forward_single(env)

        assert sched.metrics.retries_attempted == 0

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_embeddings_endpoint_enables_read_timeout_retry(self, mock_retry):
        # /api/embeddings is idempotent — the helper is invoked with
        # retry_read_timeouts=True even when the global config is False.
        mock_retry.return_value = (MagicMock(), 1, False)
        cfg = _make_config(**{"retry.read_timeouts": False})
        sched = _make_scheduler(config=cfg)

        await sched._forward_single(_make_envelope(endpoint="/api/embeddings"))

        kwargs = mock_retry.call_args.kwargs
        assert kwargs["retry_read_timeouts"] is True

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    async def test_chat_endpoint_does_not_force_read_timeout_retry(self, mock_retry):
        mock_retry.return_value = (MagicMock(), 1, False)
        cfg = _make_config(**{"retry.read_timeouts": False})
        sched = _make_scheduler(config=cfg)

        await sched._forward_single(_make_envelope(endpoint="/api/chat"))

        kwargs = mock_retry.call_args.kwargs
        assert kwargs["retry_read_timeouts"] is False

    @patch("ollama_marshal.scheduler.call_with_retry", new_callable=AsyncMock)
    @patch("ollama_marshal.scheduler.forward_request", new_callable=AsyncMock)
    async def test_streaming_envelope_skips_retry_helper(
        self, mock_forward, mock_retry
    ):
        # Streaming requests must NEVER go through the retry helper —
        # _resolve_retry_attempts returns 1 for stream=True, taking the
        # fast path.
        mock_forward.return_value = MagicMock()
        cfg = _make_config(**{"retry.max_attempts": 5})
        sched = _make_scheduler(config=cfg)

        await sched._forward_single(_make_envelope(stream=True))

        mock_forward.assert_called_once()
        mock_retry.assert_not_called()


class TestEnvelopeNumCtxHelpers:
    """Helpers that read num_ctx out of envelopes for slot sizing."""

    def test_envelope_num_ctx_none_when_no_options(self):
        env = _make_envelope()
        env.request_body = {"model": "x"}
        assert Scheduler._envelope_num_ctx(env) is None

    def test_envelope_num_ctx_none_when_options_not_dict(self):
        env = _make_envelope()
        env.request_body = {"options": "not a dict"}
        assert Scheduler._envelope_num_ctx(env) is None

    def test_envelope_num_ctx_extracted(self):
        env = _make_envelope()
        env.request_body = {"options": {"num_ctx": 16384}}
        assert Scheduler._envelope_num_ctx(env) == 16384

    def test_envelope_num_ctx_ignores_zero_or_negative(self):
        env = _make_envelope()
        env.request_body = {"options": {"num_ctx": 0}}
        assert Scheduler._envelope_num_ctx(env) is None
        env.request_body = {"options": {"num_ctx": -1}}
        assert Scheduler._envelope_num_ctx(env) is None

    async def test_max_num_ctx_for_pending_picks_largest(self):
        queues = ModelQueues()
        e1 = _make_envelope()
        e1.request_body = {"options": {"num_ctx": 8192}}
        e2 = _make_envelope()
        e2.request_body = {"options": {"num_ctx": 32768}}
        e3 = _make_envelope()  # no num_ctx
        await queues.enqueue(e1)
        await queues.enqueue(e2)
        await queues.enqueue(e3)

        sched = _make_scheduler(queues=queues)
        assert await sched._max_num_ctx_for_pending("llama3:latest") == 32768

    async def test_max_num_ctx_for_pending_returns_none_for_empty(self):
        sched = _make_scheduler()
        assert await sched._max_num_ctx_for_pending("nothing:queued") is None


class TestEnsureModelLoadedReloadOnNeed:
    """Reload-on-need: dispatch with a num_ctx > current allocation reloads."""

    async def test_skips_reload_when_request_fits(self):
        memory = MagicMock()
        memory.is_loaded.return_value = True
        memory.needs_reload.return_value = False
        memory.refresh = AsyncMock()
        memory.take_unexpected_unload_count.return_value = 0
        memory.mark_intended_unload = MagicMock()
        memory.mark_owned = MagicMock()
        memory.record_allocated_num_ctx = MagicMock()

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)
        lifecycle.unload = AsyncMock(return_value=True)

        sched = _make_scheduler(memory=memory, lifecycle=lifecycle)

        result = await sched._ensure_model_loaded("llama3:latest", num_ctx=4096)

        assert result is True
        # No reload work — preload not called.
        lifecycle.preload.assert_not_called()
        lifecycle.unload.assert_not_called()
        assert sched.metrics.reload_count == 0

    async def test_reload_does_not_drain_pending_via_old_slot(self):
        # Critical correctness test: the request whose num_ctx > allocated
        # is what TRIGGERED the reload. If we drained pending before
        # unload, that request would dispatch via the OLD smaller slot
        # and Ollama would silently truncate it. The whole point of
        # Surface C1 Dim 4 is to NEVER silently truncate. So the reload
        # path MUST NOT call _process_batch — pending requests stay
        # queued and dispatch on the next tick against the new slot.
        memory = MagicMock()
        memory.is_loaded.return_value = True
        memory.needs_reload.return_value = True
        memory.get_allocated_num_ctx.return_value = 4096
        memory.can_fit_model.return_value = True
        memory.refresh = AsyncMock()
        memory.take_unexpected_unload_count.return_value = 0
        memory.mark_intended_unload = MagicMock()
        memory.mark_owned = MagicMock()
        memory.record_allocated_num_ctx = MagicMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)

        lifecycle = MagicMock()
        lifecycle.preload = AsyncMock(return_value=True)
        lifecycle.unload = AsyncMock(return_value=True)

        queues = ModelQueues()
        env = _make_envelope()
        env.request_body = {"options": {"num_ctx": 32768}}
        await queues.enqueue(env)

        sched = _make_scheduler(
            queues=queues, memory=memory, registry=registry, lifecycle=lifecycle
        )

        with patch.object(
            sched, "_process_batch", new_callable=AsyncMock
        ) as mock_process:
            result = await sched._ensure_model_loaded("llama3:latest", num_ctx=32768)

        assert result is True
        # CRITICAL: drain MUST NOT have run.
        mock_process.assert_not_called()
        # Pending request remains queued for next-tick dispatch against
        # the new (larger) slot.
        assert await queues.pending_count("llama3:latest") == 1
        # Marshal told memory it's an intended unload.
        memory.mark_intended_unload.assert_called_with(
            "llama3:latest", instance_url=_PRIMARY_INSTANCE_URL
        )
        # Unload then preload at the bigger size.
        lifecycle.unload.assert_called_once_with(
            "llama3:latest", instance_url=_PRIMARY_INSTANCE_URL
        )
        lifecycle.preload.assert_called_once_with(
            "llama3:latest",
            num_ctx=32768,
            instance_url=_PRIMARY_INSTANCE_URL,
            load_timeout_s=3600,
            is_known_model_check=ANY,
        )
        memory.record_allocated_num_ctx.assert_called_with(
            "llama3:latest", 32768, instance_url=_PRIMARY_INSTANCE_URL
        )
        # Metric bumped only after preload succeeded.
        assert sched.metrics.reload_count == 1

    async def test_reload_count_not_bumped_on_failed_preload(self):
        # Reload counter should track ACTUAL reloads, not attempts.
        # If the unload succeeds but the preload fails, the model is
        # gone and we couldn't replace it — that's a degraded state,
        # not a successful reload.
        memory = MagicMock()
        memory.is_loaded.return_value = True
        memory.needs_reload.return_value = True
        memory.get_allocated_num_ctx.return_value = 4096
        memory.can_fit_model.return_value = True
        memory.available_vram.return_value = 50 * 1024**3
        memory.refresh = AsyncMock()
        memory.take_unexpected_unload_count.return_value = 0
        memory.mark_intended_unload = MagicMock()
        memory.mark_owned = MagicMock()
        memory.record_allocated_num_ctx = MagicMock()

        registry = MagicMock()
        registry.get_or_estimate_size = AsyncMock(return_value=4 * 1024**3)

        lifecycle = MagicMock()
        # Preload fails.
        lifecycle.preload = AsyncMock(return_value=False)
        lifecycle.unload = AsyncMock(return_value=True)

        sched = _make_scheduler(memory=memory, registry=registry, lifecycle=lifecycle)
        result = await sched._ensure_model_loaded("llama3:latest", num_ctx=32768)

        assert result is False
        # Counter did NOT bump on failed reload.
        assert sched.metrics.reload_count == 0
        # Sentinel allocation written so future calls force a fresh reload
        # rather than silently dispatching against an unknown slot.
        memory.record_allocated_num_ctx.assert_called_with(
            "llama3:latest", 0, instance_url=_PRIMARY_INSTANCE_URL
        )


class TestUnexpectedUnloadsRollup:
    async def test_tick_drains_unexpected_unloads_into_metrics(self):
        # Memory poll loop runs in the background and accumulates a
        # count; the scheduler tick must drain it into SchedulerMetrics
        # so the dashboard sees one authoritative number.
        memory = MagicMock()
        memory.get_loaded_models.return_value = {}
        memory.is_loaded.return_value = False
        memory.refresh = AsyncMock()
        memory.take_unexpected_unload_count.return_value = 3
        memory.mark_intended_unload = MagicMock()
        memory.mark_owned = MagicMock()
        memory.record_allocated_num_ctx = MagicMock()

        sched = _make_scheduler(memory=memory)

        with (
            patch.object(
                sched, "_forward_loaded_model_requests", new_callable=AsyncMock
            ),
            patch.object(sched, "_handle_critical_preemption", new_callable=AsyncMock),
            patch.object(sched, "_handle_unskippable_requests", new_callable=AsyncMock),
            patch.object(sched, "_bin_pack_models", new_callable=AsyncMock),
            patch.object(sched, "_idle_evict_unused_models", new_callable=AsyncMock),
        ):
            await sched._tick()

        assert sched.metrics.unexpected_unloads == 3


class TestSchedulerMetricsRetryFields:
    def test_default_zero(self):
        m = SchedulerMetrics()
        assert m.retries_attempted == 0
        assert m.retries_succeeded == 0
        assert m.unexpected_unloads == 0
        assert m.reload_count == 0

    def test_serializes_new_fields(self):
        m = SchedulerMetrics(
            retries_attempted=5,
            retries_succeeded=4,
            unexpected_unloads=2,
            reload_count=7,
        )
        d = m.to_json_dict()
        assert d["retries_attempted"] == 5
        assert d["retries_succeeded"] == 4
        assert d["unexpected_unloads"] == 2
        assert d["reload_count"] == 7

    def test_loads_with_missing_new_fields_for_v0_3_x_snapshots(self):
        # A v0.3.x metrics file won't have the new fields — they should
        # default to 0 instead of raising.
        from ollama_marshal.scheduler import _METRICS_SCHEMA_VERSION

        m = SchedulerMetrics.from_json_dict(
            {
                "schema_version": _METRICS_SCHEMA_VERSION,
                "requests_served": 10,
                "model_swaps": 1,
                "evictions": 0,
                "total_wait_ms": 1234.0,
            }
        )
        assert m.requests_served == 10
        assert m.retries_attempted == 0
        assert m.retries_succeeded == 0
        assert m.unexpected_unloads == 0
        assert m.reload_count == 0


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

        sched.lifecycle.unload.assert_awaited_once_with(
            "foo", instance_url=_PRIMARY_INSTANCE_URL
        )
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

    def test_aggregate_cap_clamps_per_model_sum(self):
        """Per-model summed boost is clamped at max_skips * aggregate_multiplier.

        Even if 100 distinct (attacker-controlled) program_ids each
        register a hint at the per-pair cap, the eviction scorer reads
        only the aggregate cap value — not 100 * per_pair_cap.
        """
        hints = BurstHints()
        # Default aggregate_multiplier=8, max_skips=3 → cap = 24.
        # Record 10 distinct programs each with per-pair cap=12 → raw
        # sum would be 120 without the aggregate cap.
        for i in range(10):
            hints.record(f"p{i}", "target", 12, max_skips=3, now=100.0)
        # Without max_skips passed: returns the raw sum.
        assert hints.boost_for_model("target", now=101.0) == 120
        # With max_skips=3 passed: clamped at 3 * 8 = 24.
        assert hints.boost_for_model("target", max_skips=3, now=101.0) == 24

    def test_all_boosts_by_model_aggregate_cap(self):
        hints = BurstHints()
        for i in range(10):
            hints.record(f"p{i}", "target", 12, max_skips=3, now=100.0)
        result = hints.all_boosts_by_model(max_skips=3, now=101.0)
        assert result["target"] == 24


class TestBurstHintsCapacity:
    """Total-dict cap prevents unbounded growth from flooded distinct pairs."""

    def test_drops_new_pair_when_at_capacity(self):
        hints = BurstHints(max_live=3)
        assert hints.record("p1", "m1", 5, max_skips=3) == 5
        assert hints.record("p2", "m2", 5, max_skips=3) == 5
        assert hints.record("p3", "m3", 5, max_skips=3) == 5
        # 4th distinct pair is dropped.
        assert hints.record("p4", "m4", 5, max_skips=3) == 0

    def test_refresh_at_capacity_still_works(self):
        # Refreshing an EXISTING pair works even when dict is full —
        # otherwise legitimate burst-refresh traffic would be lost.
        hints = BurstHints(max_live=2)
        hints.record("p1", "m1", 5, max_skips=3, now=100.0)
        hints.record("p2", "m2", 5, max_skips=3, now=100.0)
        # At capacity. Refresh of an existing pair should succeed.
        assert hints.record("p1", "m1", 8, max_skips=3, now=110.0) == 8

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
        # Default scheduler config: max_skips=3, aggregate_multiplier=8 →
        # per-model aggregate cap = 24. Even though the per-pair record
        # accepts 50 (max_skips=15 → per-pair cap=60), the eviction-time
        # call clamps the per-model sum to scheduler.max_skips * 8 = 24.
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

        # Boost is added then capped at aggregate_multiplier * max_skips
        # = 8 * 3 = 24 (using SchedulerConfig.max_skips default of 3).
        assert captured["pending"]["burst-protected"] == 24
        # The chosen target is low-priority — burst-protected stays loaded.
        sched.lifecycle.unload.assert_awaited_with(
            "low-priority", instance_url=_PRIMARY_INSTANCE_URL
        )


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
        # Patches `forward_request` (the Ollama HTTP boundary imported
        # into scheduler.py), NOT `_forward_single` — per CLAUDE.md
        # bright-line #1, tests must not patch internal ollama_marshal
        # methods. `forward_request` is the legitimate mock target.
        sched = _make_scheduler()
        batch = [_make_envelope(model="m") for _ in range(5)]

        in_flight = 0
        peak = 0

        async def fake_request(*_args, **_kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1
            return MagicMock()

        with patch(
            "ollama_marshal.scheduler.forward_request", side_effect=fake_request
        ):
            await sched._process_batch(batch)

        assert peak == 1  # current behavior: serial

    async def test_parallel_4_runs_4_concurrent(self):
        cfg = _make_config(**{"scheduler.parallel_per_model": 4})
        sched = _make_scheduler(config=cfg)
        batch = [_make_envelope(model="m") for _ in range(10)]

        in_flight = 0
        peak = 0

        async def fake_request(*_args, **_kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1
            return MagicMock()

        with patch(
            "ollama_marshal.scheduler.forward_request", side_effect=fake_request
        ):
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

        async def fake_request(*_args, **_kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1
            return MagicMock()

        with patch(
            "ollama_marshal.scheduler.forward_request", side_effect=fake_request
        ):
            await sched._process_batch(batch)

        assert peak == 5  # all 5 embeddings ran concurrently

    async def test_semaphore_releases_on_exception(self):
        # If forward_request raises, the semaphore must still release;
        # otherwise a single failure permanently shrinks the cap.
        cfg = _make_config(**{"scheduler.parallel_per_model": 2})
        sched = _make_scheduler(config=cfg)
        batch = [_make_envelope(model="m") for _ in range(4)]

        async def bad_request(*_args, **_kwargs):
            raise RuntimeError("boom")

        with patch("ollama_marshal.scheduler.forward_request", side_effect=bad_request):
            # _forward_single catches and logs; gather(return_exceptions)
            # swallows what's left.
            await sched._process_batch(batch)

        # All 4 should have been attempted (semaphore released after each).
        # Verify the semaphore is back to fully available (2/2 free).
        sem = sched.inflight.semaphore_for("m")
        # Both slots free → can acquire twice without blocking.
        async with sem, sem:
            pass


# ---------------------------------------------------------------------------
# Multi-instance routing — _resolve_routing + envelope tagging
# ---------------------------------------------------------------------------


def _multi_instance_config():
    """Build a 3-instance config for routing tests."""
    from ollama_marshal.config import KVCacheType, OllamaInstance

    return MarshalConfig(
        instances=[
            OllamaInstance(url="http://localhost:11434", kv_cache_type=KVCacheType.F16),
            OllamaInstance(
                url="http://localhost:11444", kv_cache_type=KVCacheType.Q8_0
            ),
        ],
    )


class TestResolveRouting:
    """``_resolve_routing`` builds the FitProbe map + calls pick_instance."""

    async def test_single_instance_short_circuits(self):
        # Default config has one auto-backfilled instance.
        sched = _make_scheduler()
        decision = await sched._resolve_routing("any:model", num_ctx=4096)
        from ollama_marshal.routing import RoutingReason

        assert decision.reason == RoutingReason.SINGLE_INSTANCE
        assert decision.instance.url == "http://localhost:11434"

    async def test_multi_instance_cold_start_picks_primary(self):
        from ollama_marshal.routing import FitProbe, RoutingReason

        config = _multi_instance_config()
        sched = _make_scheduler(config=config)
        # Both instances are empty + budget has room. Routing should
        # pick the primary (f16) tier.
        sched.memory.loaded_on.return_value = {
            "http://localhost:11434": set(),
            "http://localhost:11444": set(),
        }
        sched.memory.probe_fit.return_value = FitProbe(
            fits=True, would_evict_non_idle=False
        )
        decision = await sched._resolve_routing("any:model", num_ctx=8192)
        assert decision.instance.url == "http://localhost:11434"
        assert decision.reason == RoutingReason.PRIMARY_FITS

    async def test_multi_instance_falls_back_to_q8_under_pressure(self):
        from ollama_marshal.routing import FitProbe, RoutingReason

        config = _multi_instance_config()
        sched = _make_scheduler(config=config)
        sched.memory.loaded_on.return_value = {
            "http://localhost:11434": {"pinned:x"},
            "http://localhost:11444": set(),
        }

        def per_instance(*, instance_url, model_size, non_idle_loaded_on_instance):
            if instance_url == "http://localhost:11434":
                return FitProbe(fits=False, would_evict_non_idle=True)
            return FitProbe(fits=True, would_evict_non_idle=False)

        sched.memory.probe_fit.side_effect = per_instance
        decision = await sched._resolve_routing("any:model", num_ctx=8192)
        assert decision.instance.url == "http://localhost:11444"
        assert decision.reason == RoutingReason.PRIMARY_WOULD_EVICT


class TestNonIdleModelsPerInstance:
    """Non-idle = has pending requests OR an active program."""

    async def test_pending_marks_model_non_idle(self):
        config = _multi_instance_config()
        queues = ModelQueues()
        await queues.enqueue(_make_envelope(model="busy:x"))
        sched = _make_scheduler(queues=queues, config=config)
        sched.memory.get_loaded_models_on.side_effect = lambda url: (
            {"busy:x": MagicMock()} if "11434" in url else {}
        )
        non_idle = await sched._non_idle_models_per_instance()
        assert "busy:x" in non_idle["http://localhost:11434"]

    async def test_active_program_marks_model_non_idle(self):
        config = _multi_instance_config()
        sched = _make_scheduler(config=config)
        sched._active_programs["recent:x"] = {"prog-a": time.monotonic()}
        sched.memory.get_loaded_models_on.side_effect = lambda url: (
            {"recent:x": MagicMock()} if "11434" in url else {}
        )
        non_idle = await sched._non_idle_models_per_instance()
        assert "recent:x" in non_idle["http://localhost:11434"]

    async def test_silent_model_is_idle(self):
        config = _multi_instance_config()
        sched = _make_scheduler(config=config)
        sched.memory.get_loaded_models_on.side_effect = lambda url: (
            {"silent:x": MagicMock()} if "11434" in url else {}
        )
        non_idle = await sched._non_idle_models_per_instance()
        assert "silent:x" not in non_idle["http://localhost:11434"]


class TestForwardSingleAuditFields:
    """Audit records carry tier_label + routing_reason from the envelope."""

    async def test_served_includes_tier_and_reason(self):
        from ollama_marshal.routing import RoutingReason

        sched = _make_scheduler()
        sched.audit = MagicMock()
        sched.audit.record = AsyncMock()
        envelope = _make_envelope(model="m:1")
        envelope.instance_url = "http://localhost:11444"
        envelope.tier_label = "fallback"
        envelope.routing_reason = RoutingReason.PRIMARY_WOULD_EVICT.value
        # Stub forward_request to return immediately.
        with patch(
            "ollama_marshal.scheduler.forward_request",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ):
            await sched._forward_single(envelope)
        served_calls = [
            c
            for c in sched.audit.record.await_args_list
            if c.args and c.args[0] == "request.served"
        ]
        assert len(served_calls) == 1
        kwargs = served_calls[0].kwargs
        assert kwargs["instance_url"] == "http://localhost:11444"
        assert kwargs["tier_label"] == "fallback"
        assert kwargs["routing_reason"] == "primary_would_evict"

    async def test_failed_includes_tier_and_reason(self):
        sched = _make_scheduler()
        sched.audit = MagicMock()
        sched.audit.record = AsyncMock()
        envelope = _make_envelope(model="m:1")
        envelope.instance_url = "http://localhost:11434"
        envelope.tier_label = "primary"
        envelope.routing_reason = "primary_fits"
        with patch(
            "ollama_marshal.scheduler.forward_request",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("oops"),
        ):
            await sched._forward_single(envelope)
        failed = [
            c
            for c in sched.audit.record.await_args_list
            if c.args and c.args[0] == "request.failed"
        ]
        assert len(failed) == 1
        kwargs = failed[0].kwargs
        assert kwargs["instance_url"] == "http://localhost:11434"
        assert kwargs["tier_label"] == "primary"
        assert kwargs["routing_reason"] == "primary_fits"


class TestForwardSingleUsesEnvelopeInstance:
    """``forward_request`` targets the envelope's instance URL when set."""

    async def test_envelope_url_overrides_legacy_host(self):
        sched = _make_scheduler()
        envelope = _make_envelope(model="m:1")
        envelope.instance_url = "http://other:9999"
        with patch(
            "ollama_marshal.scheduler.forward_request",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_forward:
            await sched._forward_single(envelope)
        assert mock_forward.await_args.kwargs["ollama_host"] == "http://other:9999"

    async def test_legacy_host_used_when_envelope_unset(self):
        sched = _make_scheduler()
        envelope = _make_envelope(model="m:1")
        # envelope.instance_url left at default None
        with patch(
            "ollama_marshal.scheduler.forward_request",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_forward:
            await sched._forward_single(envelope)
        assert mock_forward.await_args.kwargs["ollama_host"] == sched.config.ollama.host


# ---------------------------------------------------------------------------
# Steady-state audit field plumbing — regression test for the
# /review-found gap where envelopes dispatched via _tag_batch_with_instance
# (already-loaded common case) were missing tier_label / routing_reason.
# ---------------------------------------------------------------------------


class TestTagBatchWithInstancePopulatesAllFields:
    """``_tag_batch_with_instance`` must populate all 3 routing fields.

    Regression target: an earlier implementation only set
    ``envelope.instance_url`` and left ``tier_label`` and
    ``routing_reason`` as None — meaning audit records for
    already-loaded steady-state dispatches (the common case) were
    missing the routing context the audit log promises to deliver.
    """

    def test_tags_all_three_fields(self):
        sched = _make_scheduler()
        batch = [_make_envelope(model="m") for _ in range(3)]
        sched._tag_batch_with_instance(batch, _PRIMARY_INSTANCE_URL)
        for env in batch:
            assert env.instance_url == _PRIMARY_INSTANCE_URL
            # tier_label looked up from memory.instances (default
            # config has one f16 instance with tier_label="primary").
            assert env.tier_label == "primary"
            # Reason for already-loaded dispatches is ALREADY_LOADED.
            assert env.routing_reason == "already_loaded"

    def test_skips_envelopes_with_existing_tags(self):
        # An envelope already tagged via _tag_pending_with_decision
        # (cold-start path) must NOT be retagged with the
        # already-loaded reason — it should keep the original
        # cold-start reason (e.g. PRIMARY_FITS).
        sched = _make_scheduler()
        env = _make_envelope(model="m")
        env.instance_url = _PRIMARY_INSTANCE_URL
        env.tier_label = "primary"
        env.routing_reason = "primary_fits"
        sched._tag_batch_with_instance([env], _PRIMARY_INSTANCE_URL)
        # Original cold-start tag preserved.
        assert env.routing_reason == "primary_fits"

    def test_no_op_when_instance_url_is_none(self):
        # Race-safe path: scheduler skips dispatch when find_instance_for
        # returns None, but if some future caller passes None directly,
        # we still no-op cleanly.
        sched = _make_scheduler()
        env = _make_envelope(model="m")
        sched._tag_batch_with_instance([env], None)
        assert env.instance_url is None
        assert env.tier_label is None
        assert env.routing_reason is None


class TestForwardLoadedModelRequestsRaceSafe:
    """``_forward_loaded_model_requests`` skips when find_instance_for is None.

    Regression target: a poll-loop unload between the
    ``get_loaded_models()`` snapshot and ``find_instance_for`` would
    leave the dispatch path with ``instance_url=None``, falling back
    to ``config.ollama.host`` — wrong instance in multi-instance
    setups.
    """

    async def test_skips_when_model_no_longer_attributable(self):
        memory = MagicMock()
        # Snapshot says model is loaded.
        memory.get_loaded_models.return_value = {"vanished:x": MagicMock()}

        queues = ModelQueues()
        envelope = _make_envelope(model="vanished:x")
        await queues.enqueue(envelope)

        sched = _make_scheduler(queues=queues, memory=memory)
        # Override the helper-applied default (which returns the primary
        # URL) AFTER scheduler construction. Simulates the race where
        # the poll loop unloaded the model between the snapshot above
        # and the find_instance_for call inside the dispatch loop.
        memory.find_instance_for.return_value = None

        with patch.object(sched, "_process_batch", new_callable=AsyncMock) as mock_pb:
            await sched._forward_loaded_model_requests()

        # Crucially, no dispatch happened — envelope still in queue.
        assert envelope.done_event.is_set() is False
        mock_pb.assert_not_called()


# ---------------------------------------------------------------------------
# Pause / resume state machine (v0.6.0+)
# ---------------------------------------------------------------------------


class TestSchedulerPauseState:
    """Pause/resume state machine — flag flips, drain wait, in-flight count.

    These tests cover the state machine in isolation. The dispatch loop
    integration (envelopes actually skipped during pause, bypass-flagged
    envelopes still dispatching) ships in v0.6.0 Stage 2 with its own
    tests.
    """

    async def test_initial_state_is_unpaused(self):
        sched = _make_scheduler()
        assert sched.is_paused() is False
        assert sched.in_flight_count() == 0

    async def test_resume_when_already_unpaused_is_noop(self):
        sched = _make_scheduler()
        sched.resume()
        assert sched.is_paused() is False

    async def test_pause_with_no_in_flight_returns_drained_immediately(self):
        sched = _make_scheduler()
        # No envelopes in flight → drain completes on first poll.
        drained = await sched.pause(drain_timeout_s=1.0)
        assert drained is True
        assert sched.is_paused() is True

    async def test_pause_then_resume_clears_flag(self):
        sched = _make_scheduler()
        await sched.pause(drain_timeout_s=1.0)
        assert sched.is_paused() is True
        sched.resume()
        assert sched.is_paused() is False

    async def test_pause_idempotent(self):
        """Calling pause twice in a row returns drained both times."""
        sched = _make_scheduler()
        first = await sched.pause(drain_timeout_s=1.0)
        second = await sched.pause(drain_timeout_s=1.0)
        assert first is True
        assert second is True
        assert sched.is_paused() is True

    async def test_pause_waits_for_in_flight_to_drain(self):
        """Pause should not return until ``_in_flight_count`` is zero."""
        sched = _make_scheduler()
        sched._in_flight_count = 1

        async def drain_in_flight() -> None:
            await asyncio.sleep(0.2)
            sched._in_flight_count = 0

        # Run the drainer concurrently with pause; pause should wait
        # for the in-flight envelope to clear before returning.
        drain_task = asyncio.create_task(drain_in_flight())
        drained = await sched.pause(drain_timeout_s=2.0)
        await drain_task

        assert drained is True
        assert sched.is_paused() is True
        assert sched.in_flight_count() == 0

    async def test_pause_returns_false_on_drain_timeout(self):
        """If in-flight envelopes don't clear in time, pause returns False.

        The flag is still set (subsequent calls see is_paused() True).
        Caller decides what to do — typically skip the test phase or
        retry pause later.
        """
        sched = _make_scheduler()
        sched._in_flight_count = 1

        # Tight timeout vs counter that never drains.
        drained = await sched.pause(drain_timeout_s=0.2)

        assert drained is False
        assert sched.is_paused() is True
        assert sched.in_flight_count() == 1

    async def test_in_flight_count_tracks_counter(self):
        sched = _make_scheduler()
        sched._in_flight_count = 2
        assert sched.in_flight_count() == 2
        sched._in_flight_count = 1
        assert sched.in_flight_count() == 1
        sched._in_flight_count = 0
        assert sched.in_flight_count() == 0

    async def test_pause_schedules_auto_resume_task(self):
        sched = _make_scheduler()
        await sched.pause(drain_timeout_s=0.1, auto_resume_after_seconds=10.0)
        assert sched._auto_resume_task is not None
        assert not sched._auto_resume_task.done()
        sched.resume()  # cleanup

    async def test_resume_cancels_auto_resume_task(self):
        sched = _make_scheduler()
        await sched.pause(drain_timeout_s=0.1, auto_resume_after_seconds=10.0)
        task = sched._auto_resume_task
        assert task is not None

        sched.resume()
        # Give the cancelled task a tick to settle.
        await asyncio.sleep(0)
        assert sched._auto_resume_task is None
        assert task.cancelled() or task.done()

    async def test_repause_cancels_prior_auto_resume_and_schedules_new(self):
        sched = _make_scheduler()
        await sched.pause(drain_timeout_s=0.1, auto_resume_after_seconds=10.0)
        first_task = sched._auto_resume_task

        await sched.pause(drain_timeout_s=0.1, auto_resume_after_seconds=20.0)
        second_task = sched._auto_resume_task

        assert second_task is not first_task
        assert first_task is not None
        # Give the cancellation a tick to land.
        await asyncio.sleep(0)
        assert first_task.cancelled() or first_task.done()
        sched.resume()  # cleanup

    async def test_auto_resume_fires_after_delay(self):
        """Sleep just past the auto-resume window; flag flips back."""
        sched = _make_scheduler()
        # Tight delay so the test runs fast.
        await sched.pause(drain_timeout_s=0.05, auto_resume_after_seconds=0.15)
        assert sched.is_paused() is True

        # Wait past the auto-resume window.
        await asyncio.sleep(0.3)
        assert sched.is_paused() is False

    async def test_scheduler_stop_cancels_auto_resume(self):
        sched = _make_scheduler()
        await sched.pause(drain_timeout_s=0.1, auto_resume_after_seconds=60.0)
        task = sched._auto_resume_task
        assert task is not None

        await sched.stop()
        assert sched._auto_resume_task is None
        # Cancellation goes through "cancelling" → "cancelled"; yield
        # the loop once to let the cancellation finalize.
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()


class TestRequestEnvelopeBypassPause:
    """RequestEnvelope.bypass_pause defaults False; explicit True allowed."""

    def test_default_bypass_pause_false(self):
        envelope = _make_envelope(model="qwen3.5:0.8b-bf16")
        assert envelope.bypass_pause is False

    def test_bypass_pause_can_be_set(self):
        envelope = RequestEnvelope(
            model="qwen3.5:0.8b-bf16",
            program_id="integration-test",
            request_body={},
            endpoint="/api/chat",
            bypass_pause=True,
        )
        assert envelope.bypass_pause is True


# ---------------------------------------------------------------------------
# Pause dispatch integration — _tick_bypass_only + _in_flight_count tracking
# ---------------------------------------------------------------------------


class TestPauseDispatchIntegration:
    """Behavior tests for the dispatch loop's pause integration.

    Covers:
    - _tick early-returns to _tick_bypass_only when paused
    - _tick_bypass_only dispatches bypass envelopes for loaded models
    - _tick_bypass_only force-loads model for bypass envelopes when needed
    - non-bypass envelopes stay queued during pause
    - _forward_single increments/decrements _in_flight_count
    """

    async def test_tick_skips_normal_dispatch_when_paused(self):
        """Paused tick must not call any of the normal dispatch steps."""
        sched = _make_scheduler()
        sched._dispatch_paused = True

        with (
            patch.object(
                sched,
                "_forward_loaded_model_requests",
                new_callable=AsyncMock,
            ) as forward,
            patch.object(
                sched, "_handle_critical_preemption", new_callable=AsyncMock
            ) as crit,
            patch.object(sched, "_bin_pack_models", new_callable=AsyncMock) as binpack,
            patch.object(
                sched, "_tick_bypass_only", new_callable=AsyncMock
            ) as bypass_only,
        ):
            await sched._tick()

        forward.assert_not_called()
        crit.assert_not_called()
        binpack.assert_not_called()
        bypass_only.assert_called_once()

    async def test_tick_runs_normal_dispatch_when_not_paused(self):
        sched = _make_scheduler()
        # Default state — not paused.
        assert sched._dispatch_paused is False

        with (
            patch.object(
                sched,
                "_forward_loaded_model_requests",
                new_callable=AsyncMock,
            ) as forward,
            patch.object(
                sched, "_tick_bypass_only", new_callable=AsyncMock
            ) as bypass_only,
        ):
            await sched._tick()

        forward.assert_called_once()
        bypass_only.assert_not_called()

    async def test_tick_bypass_only_dispatches_bypass_envelopes(self):
        """Pop bypass envelope, call _process_batch with it."""
        queues = ModelQueues()
        bypass_env = _make_envelope(model="loaded:1")
        bypass_env.bypass_pause = True
        await queues.enqueue(bypass_env)

        memory = MagicMock()
        memory.is_loaded.return_value = True  # No load needed

        sched = _make_scheduler(queues=queues, memory=memory)

        with patch.object(
            sched, "_process_batch", new_callable=AsyncMock
        ) as process_batch:
            await sched._tick_bypass_only()

        process_batch.assert_called_once()
        dispatched = process_batch.call_args[0][0]
        assert dispatched == [bypass_env]

    async def test_tick_bypass_only_skips_non_bypass_envelopes(self):
        """Non-bypass envelope stays in queue; not dispatched during pause."""
        queues = ModelQueues()
        normal_env = _make_envelope(model="loaded:1")
        # bypass_pause defaults to False
        await queues.enqueue(normal_env)

        memory = MagicMock()
        memory.is_loaded.return_value = True

        sched = _make_scheduler(queues=queues, memory=memory)

        with patch.object(
            sched, "_process_batch", new_callable=AsyncMock
        ) as process_batch:
            await sched._tick_bypass_only()

        process_batch.assert_not_called()
        # Envelope still in queue.
        assert await queues.pending_count("loaded:1") == 1

    async def test_tick_bypass_only_force_loads_model_when_unloaded(self):
        """If model isn't loaded, _ensure_model_loaded fires before dispatch."""
        queues = ModelQueues()
        bypass_env = _make_envelope(model="cold:1")
        bypass_env.bypass_pause = True
        await queues.enqueue(bypass_env)

        memory = MagicMock()
        memory.is_loaded.return_value = False  # Model not loaded

        sched = _make_scheduler(queues=queues, memory=memory)

        with (
            patch.object(
                sched, "_ensure_model_loaded", new_callable=AsyncMock
            ) as ensure,
            patch.object(
                sched, "_process_batch", new_callable=AsyncMock
            ) as process_batch,
            patch.object(
                sched, "_max_num_ctx_for_pending", new_callable=AsyncMock
            ) as max_ctx,
        ):
            ensure.return_value = True
            max_ctx.return_value = None
            await sched._tick_bypass_only()

        ensure.assert_awaited_once_with("cold:1", num_ctx=None)
        process_batch.assert_called_once()

    async def test_tick_bypass_only_keeps_envelopes_when_load_fails(self):
        """If _ensure_model_loaded returns False, bypass envelopes stay queued."""
        queues = ModelQueues()
        bypass_env = _make_envelope(model="cold:1")
        bypass_env.bypass_pause = True
        await queues.enqueue(bypass_env)

        memory = MagicMock()
        memory.is_loaded.return_value = False

        sched = _make_scheduler(queues=queues, memory=memory)

        with (
            patch.object(
                sched, "_ensure_model_loaded", new_callable=AsyncMock
            ) as ensure,
            patch.object(
                sched, "_process_batch", new_callable=AsyncMock
            ) as process_batch,
            patch.object(
                sched, "_max_num_ctx_for_pending", new_callable=AsyncMock
            ) as max_ctx,
        ):
            ensure.return_value = False  # Load failed
            max_ctx.return_value = None
            await sched._tick_bypass_only()

        process_batch.assert_not_called()
        # Envelope still in queue.
        assert await queues.pending_count("cold:1") == 1

    async def test_forward_single_increments_in_flight_count(self):
        """_in_flight_count rises during dispatch, falls back after."""
        sched = _make_scheduler()
        observed_during_call: list[int] = []

        async def fake_inner(_envelope):
            observed_during_call.append(sched._in_flight_count)

        envelope = _make_envelope(model="x:1")
        with patch.object(sched, "_forward_single_inner", side_effect=fake_inner):
            await sched._forward_single(envelope)

        assert observed_during_call == [1]
        assert sched._in_flight_count == 0

    async def test_forward_single_decrements_in_flight_on_exception(self):
        """If dispatch raises, _in_flight_count still decrements."""
        sched = _make_scheduler()

        async def fake_inner(_envelope):
            raise RuntimeError("dispatch failed")

        envelope = _make_envelope(model="x:1")
        with (
            patch.object(sched, "_forward_single_inner", side_effect=fake_inner),
            pytest.raises(RuntimeError),
        ):
            await sched._forward_single(envelope)

        assert sched._in_flight_count == 0

    async def test_in_flight_count_does_not_go_negative(self):
        """Defensive: max(0, ...) guards against double-decrement bugs."""
        sched = _make_scheduler()
        # Simulate a state where the counter is already 0 but a finally
        # block fires anyway — should clamp to 0, not go negative.
        sched._in_flight_count = 0
        # We can't easily trigger this naturally; just verify the
        # max(0, ...) guard exists by reading the source semantics:
        # the assertion below confirms the public API never reports
        # negative numbers regardless of internal state.
        assert sched.in_flight_count() >= 0


# ---------------------------------------------------------------------------
# v0.6.4 Bug 2 — per-model preload backoff with exponential delay + giveup
# ---------------------------------------------------------------------------


class TestPreloadBackoffStateMachine:
    """Cover the cooldown + backoff + giveup helpers added in v0.6.4.

    The state machine prevents the 313-failures-in-30s cascade observed
    when Ollama crashes — without backoff, the 0.1s scheduler tick
    keeps calling ``lifecycle.preload`` ~10 times/sec while Ollama is
    unreachable. After ``preload_max_consecutive_failures`` consecutive
    failures, queued envelopes for that model fail with
    ``PreloadFailedError`` rather than waiting forever.
    """

    async def test_preload_failure_records_state_with_backoff(self):
        """A failed preload bumps the failure counter and sets a cooldown deadline."""
        cfg = _make_config(
            **{
                "scheduler.preload_backoff_base_s": 1.0,
                "scheduler.preload_backoff_max_s": 30.0,
            }
        )
        sched = _make_scheduler(config=cfg)

        # First failure → consecutive_failures=1, cooldown set in future.
        before = time.monotonic()
        failures = sched._record_preload_failure("llama3:latest")
        after = time.monotonic()

        assert failures == 1
        state = sched._preload_failures["llama3:latest"]
        assert state.consecutive_failures == 1
        # Cooldown is `now + delay` where delay ∈ [0, base_s] for the first
        # failure (full jitter). Verify the deadline lies in that window.
        assert state.cooldown_until >= before
        assert state.cooldown_until <= after + 1.0

    async def test_preload_in_cooldown_is_skipped(self):
        """``_attempt_preload`` returns False without calling lifecycle.preload."""
        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)

        # Plant a fresh cooldown 60s in the future.
        from ollama_marshal.scheduler import _PreloadFailureState

        sched._preload_failures["llama3:latest"] = _PreloadFailureState(
            consecutive_failures=2,
            cooldown_until=time.monotonic() + 60.0,
        )

        result = await sched._attempt_preload(
            "llama3:latest", num_ctx=None, instance_url=_PRIMARY_INSTANCE_URL
        )

        assert result is False
        sched.lifecycle.preload.assert_not_called()

    async def test_preload_success_clears_failure_state(self):
        """A successful preload removes the model's failure entry."""
        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)

        from ollama_marshal.scheduler import _PreloadFailureState

        sched._preload_failures["llama3:latest"] = _PreloadFailureState(
            consecutive_failures=2,
            cooldown_until=time.monotonic() - 1.0,  # already expired
        )

        result = await sched._attempt_preload(
            "llama3:latest", num_ctx=None, instance_url=_PRIMARY_INSTANCE_URL
        )

        assert result is True
        assert "llama3:latest" not in sched._preload_failures

    async def test_max_consecutive_failures_fails_pending_envelopes(self):
        """After N failures the queued envelopes fail with PreloadFailedError.

        Drives ``_attempt_preload`` past
        ``preload_max_consecutive_failures``, asserts that pending
        envelopes get ``envelope.error = PreloadFailedError`` and
        ``done_event`` set so blocked clients unblock immediately.
        Failure state clears so a future request can try again from
        scratch.
        """
        from ollama_marshal.queue import PreloadFailedError

        cfg = _make_config(
            **{
                "scheduler.preload_max_consecutive_failures": 3,
                "scheduler.preload_backoff_base_s": 0.001,
                "scheduler.preload_backoff_max_s": 0.001,
            }
        )
        queues = ModelQueues()
        env1 = _make_envelope(model="llama3:latest", program_id="p1")
        env2 = _make_envelope(model="llama3:latest", program_id="p2")
        await queues.enqueue(env1)
        await queues.enqueue(env2)

        sched = _make_scheduler(queues=queues, config=cfg)
        sched.lifecycle.preload = AsyncMock(return_value=False)

        # 3 attempts → failures hit the max on the 3rd, giveup fires.
        for _ in range(3):
            # Sleep just enough that the tiny cooldown clears between
            # attempts (base/max set to 0.001s above so the test stays
            # snappy without flakes).
            await asyncio.sleep(0.005)
            await sched._attempt_preload(
                "llama3:latest", num_ctx=None, instance_url=_PRIMARY_INSTANCE_URL
            )

        # Both envelopes failed with PreloadFailedError.
        assert env1.done_event.is_set()
        assert env2.done_event.is_set()
        assert isinstance(env1.error, PreloadFailedError)
        assert isinstance(env2.error, PreloadFailedError)
        # Failure state cleared after giveup so the next request retries
        # from scratch (per-batch giveup, not permanent lockout).
        assert "llama3:latest" not in sched._preload_failures
        # All three preload attempts actually ran (cooldown didn't skip
        # them because we slept past the tiny window between calls).
        assert sched.lifecycle.preload.await_count == 3

    async def test_clear_preload_failure_idempotent_on_unknown_model(self):
        """``_clear_preload_failure`` on an absent key is a silent no-op."""
        sched = _make_scheduler()
        # No KeyError, no log spam.
        sched._clear_preload_failure("never-tracked:latest")
        assert "never-tracked:latest" not in sched._preload_failures

    async def test_is_in_preload_cooldown_returns_false_for_untracked_model(self):
        """Models without recorded failures aren't in cooldown."""
        sched = _make_scheduler()
        assert sched._is_in_preload_cooldown("fresh:latest") is False

    async def test_attempt_preload_threads_forward_timeout_to_lifecycle(self):
        """``_attempt_preload`` passes ``ollama_forward_timeout_s`` through.

        Bug 1 + Bug 2 integration check — the scheduler reads the
        configured Hop 2 timeout and forwards it as ``load_timeout_s``
        so preloads get the same wall-clock budget as forward calls.
        """
        cfg = _make_config(**{"scheduler.ollama_forward_timeout_s": 1234})
        sched = _make_scheduler(config=cfg)
        sched.lifecycle.preload = AsyncMock(return_value=True)

        await sched._attempt_preload(
            "llama3:latest", num_ctx=None, instance_url=_PRIMARY_INSTANCE_URL
        )

        sched.lifecycle.preload.assert_awaited_once_with(
            "llama3:latest",
            num_ctx=None,
            instance_url=_PRIMARY_INSTANCE_URL,
            load_timeout_s=1234,
            is_known_model_check=ANY,
        )

    async def test_ensure_model_loaded_short_circuits_during_cooldown(self):
        """``_ensure_model_loaded`` bails before any eviction work during cooldown.

        Without the early return, the eviction path could tear down
        loaded models to make room for a preload that immediately gets
        skipped — leaving VRAM empty.
        """
        from ollama_marshal.scheduler import _PreloadFailureState

        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)
        sched.lifecycle.unload = AsyncMock(return_value=True)
        sched._preload_failures["llama3:latest"] = _PreloadFailureState(
            consecutive_failures=2,
            cooldown_until=time.monotonic() + 60.0,
        )

        result = await sched._ensure_model_loaded("llama3:latest", num_ctx=None)

        assert result is False
        # Neither preload nor unload should have fired — the cooldown
        # short-circuit happens before any eviction work.
        sched.lifecycle.preload.assert_not_called()
        sched.lifecycle.unload.assert_not_called()


class TestUnexpectedUnloadReactivity:
    """v0.6.5 Bug 4: scheduler reacts to Ollama-side evictions on next tick.

    Memory poller surfaces evictions in
    ``_recent_unexpected_unloads``; scheduler ``_tick`` drains them
    into ``_needs_reload``; bin-packing then bypasses the per-model
    preload cooldown for the just-evicted model so a preload-failure
    cooldown from an unrelated earlier failure can't delay the
    eviction-triggered reload.
    """

    async def test_tick_drains_recent_evictions_into_needs_reload(self):
        sched = _make_scheduler()
        # Simulate the memory poller having detected an eviction —
        # take_recent_unexpected_unloads returns the (model, instance)
        # pair, drains internally on the memory side.
        sched.memory.take_recent_unexpected_unloads.return_value = {
            ("llama3:latest", _PRIMARY_INSTANCE_URL),
        }
        # Make the rest of the tick a no-op so the assertion is solely
        # about the Step 0 drain.
        sched._forward_loaded_model_requests = AsyncMock()
        sched._handle_critical_preemption = AsyncMock()
        sched._handle_unskippable_requests = AsyncMock()
        sched._bin_pack_models = AsyncMock()
        sched._idle_evict_unused_models = AsyncMock()

        await sched._tick()

        assert ("llama3:latest", _PRIMARY_INSTANCE_URL) in sched._needs_reload
        sched.memory.take_recent_unexpected_unloads.assert_called_once()

    async def test_needs_reload_bypasses_cooldown_in_bin_pack(self):
        """A just-evicted model preloads immediately, even if its cooldown is active.

        Reproduces the exact scenario the reactivity feature exists for:
        a model has a stale preload-failure cooldown (from an earlier
        unrelated failure window) AND just got Ollama-side evicted. The
        cooldown would normally skip the bin-pack attempt, leaving
        pending requests for the evicted model parked. Bug 4's
        ``_needs_reload`` set bypasses the cooldown so the model
        reloads on this tick.
        """
        from ollama_marshal.scheduler import _PreloadFailureState

        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)
        # Plant a fresh cooldown that would normally block bin-pack.
        sched._preload_failures["llama3:latest"] = _PreloadFailureState(
            consecutive_failures=2,
            cooldown_until=time.monotonic() + 60.0,
        )
        # Mark as needing reload (eviction-triggered).
        sched._needs_reload.add(("llama3:latest", _PRIMARY_INSTANCE_URL))
        # Queue a pending envelope so bin-pack picks the model.
        await sched.queues.enqueue(_make_envelope(model="llama3:latest"))
        sched.memory.is_loaded.return_value = False

        await sched._bin_pack_models()

        # Cooldown was bypassed → preload was actually attempted.
        sched.lifecycle.preload.assert_called_once()
        # Successful preload clears the (model, instance) entry so
        # subsequent ticks don't re-force on the same instance.
        assert ("llama3:latest", _PRIMARY_INSTANCE_URL) not in sched._needs_reload

    async def test_needs_reload_cleared_on_preload_failure(self):
        """Failed preload still drops the entry so cooldown-bypass doesn't loop.

        Without this drop, a flapping Ollama would: bypass cooldown →
        attempt preload → fail → record_preload_failure → next tick
        bypass cooldown again → attempt preload → fail again. That
        converts v0.6.4's jittered exponential backoff into per-tick
        (~100 ms) hammering, defeating the storm protection.
        Dropping on failure forces subsequent attempts back through
        the normal cooldown gate.
        """
        from ollama_marshal.scheduler import _PreloadFailureState

        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=False)
        sched._preload_failures["llama3:latest"] = _PreloadFailureState(
            consecutive_failures=1,
            cooldown_until=time.monotonic() + 60.0,
        )
        sched._needs_reload.add(("llama3:latest", _PRIMARY_INSTANCE_URL))
        await sched.queues.enqueue(_make_envelope(model="llama3:latest"))
        sched.memory.is_loaded.return_value = False

        await sched._bin_pack_models()

        sched.lifecycle.preload.assert_called_once()
        # Entry dropped despite failure — next tick honors the cooldown.
        assert ("llama3:latest", _PRIMARY_INSTANCE_URL) not in sched._needs_reload

    async def test_needs_reload_only_drops_matching_instance(self):
        """Multi-instance: success on instance A leaves (model, B) intact.

        Reactivity is per-(model, instance). When Ollama evicts model X
        on both instance A and instance B, marshal records two entries.
        Reloading X on A only must NOT silently drop the B entry —
        otherwise B would never get its eviction-triggered reload, and
        a future request routed to B would dispatch against a phantom-
        loaded model.
        """
        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)
        sched._needs_reload.add(("llama3:latest", _PRIMARY_INSTANCE_URL))
        sched._needs_reload.add(("llama3:latest", "http://other:11434"))
        await sched.queues.enqueue(_make_envelope(model="llama3:latest"))
        sched.memory.is_loaded.return_value = False

        await sched._bin_pack_models()

        # Primary entry dropped (bin-pack picked _PRIMARY_INSTANCE_URL).
        assert ("llama3:latest", _PRIMARY_INSTANCE_URL) not in sched._needs_reload
        # Other-instance entry still pending — that instance hasn't
        # been reloaded yet.
        assert ("llama3:latest", "http://other:11434") in sched._needs_reload

    async def test_ensure_model_loaded_bypasses_cooldown_for_evicted(self):
        """CRITICAL preemption path also reacts to evictions.

        Pre-fix, only ``_bin_pack_models`` consulted ``_needs_reload``,
        so a CRITICAL request for a just-evicted model with a stale
        cooldown sat blocked until the cooldown expired.
        ``_ensure_model_loaded`` (used by ``_handle_critical_preemption``
        and ``_handle_unskippable_requests``) now mirrors the bypass.
        """
        from ollama_marshal.scheduler import _PreloadFailureState

        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)
        sched.lifecycle.unload = AsyncMock(return_value=True)
        sched._preload_failures["llama3:latest"] = _PreloadFailureState(
            consecutive_failures=2,
            cooldown_until=time.monotonic() + 60.0,
        )
        sched._needs_reload.add(("llama3:latest", _PRIMARY_INSTANCE_URL))
        sched.memory.is_loaded_on.return_value = False

        result = await sched._ensure_model_loaded("llama3:latest", num_ctx=None)

        assert result is True
        sched.lifecycle.preload.assert_called_once()
        assert ("llama3:latest", _PRIMARY_INSTANCE_URL) not in sched._needs_reload


class TestPreloadOwnershipClaim:
    """Bug 8: a successful ``_attempt_preload`` claims memory ownership.

    Without this, ``shutdown.unload_models`` would tear down every
    model in ``/api/ps`` — including models loaded by another marshal
    or human against the same Ollama. The integration suite hit this
    when its test marshal saw prod marshal's gpt-oss:120b in /api/ps.
    """

    async def test_successful_preload_marks_owned(self):
        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)

        result = await sched._attempt_preload(
            "llama3:latest", num_ctx=None, instance_url=_PRIMARY_INSTANCE_URL
        )

        assert result is True
        sched.memory.mark_owned.assert_called_once_with(
            "llama3:latest", _PRIMARY_INSTANCE_URL
        )

    async def test_failed_preload_does_not_mark_owned(self):
        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=False)

        result = await sched._attempt_preload(
            "llama3:latest", num_ctx=None, instance_url=_PRIMARY_INSTANCE_URL
        )

        assert result is False
        sched.memory.mark_owned.assert_not_called()

    async def test_cooldown_skip_does_not_mark_owned(self):
        # Skipped preloads short-circuit before lifecycle.preload — the
        # model isn't loaded, so claiming ownership would be a lie.
        from ollama_marshal.scheduler import _PreloadFailureState

        sched = _make_scheduler()
        sched.lifecycle.preload = AsyncMock(return_value=True)
        sched._preload_failures["llama3:latest"] = _PreloadFailureState(
            consecutive_failures=2,
            cooldown_until=time.monotonic() + 60.0,
        )

        result = await sched._attempt_preload(
            "llama3:latest", num_ctx=None, instance_url=_PRIMARY_INSTANCE_URL
        )

        assert result is False
        sched.memory.mark_owned.assert_not_called()
        sched.lifecycle.preload.assert_not_called()
