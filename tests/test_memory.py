from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from ollama_marshal.config import MarshalConfig, MemoryConfig, OllamaConfig
from ollama_marshal.memory import LoadedModel, MemoryBudget, MemoryManager

PATCH_ASYNC_CLIENT = "ollama_marshal.memory.httpx.AsyncClient"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    total_ram=None,
    os_overhead="4GB",
    safety_margin="2GB",
    poll_interval=5,
    host="http://localhost:11434",
):
    return MarshalConfig(
        ollama=OllamaConfig(host=host),
        memory=MemoryConfig(
            total_ram=total_ram,
            os_overhead=os_overhead,
            safety_margin=safety_margin,
            poll_interval=poll_interval,
        ),
    )


def _ps_response(*models):
    return {
        "models": [
            {
                "name": m[0],
                "model": m[0],
                "size_vram": m[1],
                "expires_at": (m[2] if len(m) > 2 else "2026-04-24T23:59:59Z"),
            }
            for m in models
        ]
    }


def _make_manager(total_ram="64GB"):
    config = _make_config(total_ram=total_ram)
    with patch("ollama_marshal.memory.psutil"):
        return MemoryManager(config)


def _mock_async_client(mock_client):
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# LoadedModel
# ---------------------------------------------------------------------------


class TestLoadedModel:
    def test_creation(self):
        model = LoadedModel(
            name="llama3:latest",
            size_vram=4_661_224_676,
            expires_at="2026-04-24T23:59:59Z",
        )
        assert model.name == "llama3:latest"
        assert model.size_vram == 4_661_224_676
        assert model.expires_at == "2026-04-24T23:59:59Z"

    def test_default_expires_at(self):
        model = LoadedModel(name="llama3:latest", size_vram=1000)
        assert model.expires_at == ""

    def test_is_dataclass(self):
        model = LoadedModel(name="test", size_vram=0)
        assert hasattr(model, "__dataclass_fields__")


# ---------------------------------------------------------------------------
# MemoryBudget
# ---------------------------------------------------------------------------


class TestMemoryBudget:
    def test_available_calculation(self):
        budget = MemoryBudget(
            total_ram=64 * 1024**3,
            os_overhead=4 * 1024**3,
            safety_margin=2 * 1024**3,
        )
        expected = (64 - 4 - 2) * 1024**3
        assert budget.available == expected

    def test_large_overhead_reduces_available(self):
        budget = MemoryBudget(
            total_ram=16 * 1024**3,
            os_overhead=8 * 1024**3,
            safety_margin=4 * 1024**3,
        )
        assert budget.available == 4 * 1024**3

    def test_zero_overhead_and_margin(self):
        budget = MemoryBudget(
            total_ram=32 * 1024**3,
            os_overhead=0,
            safety_margin=0,
        )
        assert budget.available == 32 * 1024**3

    def test_available_can_be_negative(self):
        budget = MemoryBudget(
            total_ram=4 * 1024**3,
            os_overhead=4 * 1024**3,
            safety_margin=2 * 1024**3,
        )
        assert budget.available == -2 * 1024**3

    def test_stores_all_fields(self):
        budget = MemoryBudget(
            total_ram=100,
            os_overhead=20,
            safety_margin=10,
        )
        assert budget.total_ram == 100
        assert budget.os_overhead == 20
        assert budget.safety_margin == 10
        assert budget.available == 70


# ---------------------------------------------------------------------------
# MemoryManager._calculate_budget
# ---------------------------------------------------------------------------


class TestCalculateBudget:
    def test_config_override_total_ram(self):
        config = _make_config(
            total_ram="64GB",
            os_overhead="4GB",
            safety_margin="2GB",
        )
        with patch("ollama_marshal.memory.psutil") as mock_psutil:
            manager = MemoryManager(config)

        mock_psutil.virtual_memory.assert_not_called()
        assert manager.budget.total_ram == 64 * 1024**3
        assert manager.budget.available == (64 - 4 - 2) * 1024**3

    def test_auto_detect_ram(self):
        config = _make_config(
            total_ram=None,
            os_overhead="4GB",
            safety_margin="2GB",
        )
        mock_vmem = MagicMock()
        mock_vmem.total = 32 * 1024**3

        with patch("ollama_marshal.memory.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem
            manager = MemoryManager(config)

        assert manager.budget.total_ram == 32 * 1024**3
        assert manager.budget.available == (32 - 4 - 2) * 1024**3

    def test_custom_overhead_and_margin(self):
        config = _make_config(
            total_ram="128GB",
            os_overhead="8GB",
            safety_margin="4GB",
        )
        with patch("ollama_marshal.memory.psutil"):
            manager = MemoryManager(config)

        assert manager.budget.os_overhead == 8 * 1024**3
        assert manager.budget.safety_margin == 4 * 1024**3
        assert manager.budget.available == (128 - 8 - 4) * 1024**3


# ---------------------------------------------------------------------------
# MemoryManager._update_from_ps
# ---------------------------------------------------------------------------


class TestUpdateFromPs:
    def test_loads_models(self):
        manager = _make_manager()
        ps_data = _ps_response(
            ("llama3:latest", 4_661_224_676),
            ("mistral:latest", 4_109_865_159),
        )
        manager._update_from_ps(ps_data)

        loaded = manager.get_loaded_models()
        assert len(loaded) == 2
        assert loaded["llama3:latest"].size_vram == 4_661_224_676
        assert loaded["mistral:latest"].size_vram == 4_109_865_159

    def test_replaces_previous_state(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("llama3:latest", 1000)),
        )
        manager._update_from_ps(
            _ps_response(("mistral:latest", 2000)),
        )

        loaded = manager.get_loaded_models()
        assert "llama3:latest" not in loaded
        assert "mistral:latest" in loaded

    def test_empty_ps_clears_models(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("llama3:latest", 1000)),
        )
        manager._update_from_ps({"models": []})
        assert manager.get_loaded_models() == {}

    def test_no_models_key(self):
        manager = _make_manager()
        manager._update_from_ps({})
        assert manager.get_loaded_models() == {}

    def test_preserves_expires_at(self):
        manager = _make_manager()
        ps_data = _ps_response(
            ("llama3:latest", 1000, "2026-12-31T00:00:00Z"),
        )
        manager._update_from_ps(ps_data)

        loaded = manager.get_loaded_models()
        assert loaded["llama3:latest"].expires_at == "2026-12-31T00:00:00Z"

    def test_missing_size_vram_defaults_to_zero(self):
        manager = _make_manager()
        ps_data = {"models": [{"name": "llama3:latest"}]}
        manager._update_from_ps(ps_data)

        loaded = manager.get_loaded_models()
        assert loaded["llama3:latest"].size_vram == 0

    def test_missing_expires_at_defaults_to_empty(self):
        manager = _make_manager()
        ps_data = {
            "models": [
                {"name": "llama3:latest", "size_vram": 1000},
            ],
        }
        manager._update_from_ps(ps_data)

        loaded = manager.get_loaded_models()
        assert loaded["llama3:latest"].expires_at == ""


# ---------------------------------------------------------------------------
# is_loaded / used_vram / available_vram
# ---------------------------------------------------------------------------


class TestIsLoaded:
    def test_loaded_model(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("llama3:latest", 1000)),
        )
        assert manager.is_loaded("llama3:latest") is True

    def test_not_loaded(self):
        manager = _make_manager()
        assert manager.is_loaded("llama3:latest") is False

    def test_after_unload(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("llama3:latest", 1000)),
        )
        manager._update_from_ps({"models": []})
        assert manager.is_loaded("llama3:latest") is False


class TestUsedVram:
    def test_no_models(self):
        manager = _make_manager()
        assert manager.used_vram() == 0

    def test_single_model(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("llama3:latest", 4_000_000_000)),
        )
        assert manager.used_vram() == 4_000_000_000

    def test_multiple_models(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(
                ("llama3:latest", 4_000_000_000),
                ("mistral:latest", 3_000_000_000),
            )
        )
        assert manager.used_vram() == 7_000_000_000


class TestAvailableVram:
    def test_no_models_full_available(self):
        manager = _make_manager(total_ram="64GB")
        expected = (64 - 4 - 2) * 1024**3
        assert manager.available_vram() == expected

    def test_with_loaded_models(self):
        manager = _make_manager(total_ram="64GB")
        used = 10 * 1024**3
        manager._update_from_ps(
            _ps_response(("llama3:latest", used)),
        )
        expected = (64 - 4 - 2) * 1024**3 - used
        assert manager.available_vram() == expected


# ---------------------------------------------------------------------------
# can_fit_model
# ---------------------------------------------------------------------------


class TestCanFitModel:
    def test_model_fits(self):
        manager = _make_manager(total_ram="64GB")
        assert manager.can_fit_model(4 * 1024**3) is True

    def test_model_does_not_fit(self):
        manager = _make_manager(total_ram="64GB")
        assert manager.can_fit_model(60 * 1024**3) is False

    def test_model_exactly_fits(self):
        manager = _make_manager(total_ram="64GB")
        available = (64 - 4 - 2) * 1024**3
        assert manager.can_fit_model(available) is True

    def test_model_does_not_fit_after_loading(self):
        manager = _make_manager(total_ram="64GB")
        used = 55 * 1024**3
        manager._update_from_ps(
            _ps_response(("llama3:latest", used)),
        )
        assert manager.can_fit_model(4 * 1024**3) is False

    def test_zero_size_model_fits(self):
        manager = _make_manager(total_ram="64GB")
        assert manager.can_fit_model(0) is True


# ---------------------------------------------------------------------------
# get_eviction_candidates
# ---------------------------------------------------------------------------


class TestGetEvictionCandidates:
    def _setup_manager_with_models(self):
        manager = _make_manager(total_ram="64GB")
        manager._update_from_ps(
            _ps_response(
                ("llama3:latest", 4_000_000_000),
                ("codellama:latest", 3_000_000_000),
                ("mistral:latest", 5_000_000_000),
            )
        )
        return manager

    def test_fewest_pending_first(self):
        manager = self._setup_manager_with_models()
        pending = {
            "llama3:latest": 5,
            "codellama:latest": 0,
            "mistral:latest": 2,
        }
        candidates = manager.get_eviction_candidates(pending, {})
        assert candidates[0] == "codellama:latest"
        assert candidates[-1] == "llama3:latest"

    def test_priority_breaks_tie_on_pending(self):
        manager = self._setup_manager_with_models()
        pending = {
            "llama3:latest": 0,
            "codellama:latest": 0,
            "mistral:latest": 0,
        }
        priorities = {
            "llama3:latest": "critical",
            "codellama:latest": "normal",
            "mistral:latest": "normal",
        }
        candidates = manager.get_eviction_candidates(
            pending,
            priorities,
        )
        # Normal (0) before critical (1) -> llama3 last
        assert candidates[-1] == "llama3:latest"
        # Among same pending + priority, larger size first
        assert candidates[0] == "mistral:latest"
        assert candidates[1] == "codellama:latest"

    def test_largest_size_breaks_final_tie(self):
        manager = self._setup_manager_with_models()
        # All pending=0, all "normal" priority -> sort by -size
        candidates = manager.get_eviction_candidates({}, {})
        assert candidates[0] == "mistral:latest"
        assert candidates[1] == "llama3:latest"
        assert candidates[2] == "codellama:latest"

    def test_empty_loaded_models(self):
        manager = _make_manager()
        candidates = manager.get_eviction_candidates({}, {})
        assert candidates == []

    def test_unknown_priority_treated_as_normal(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(
                ("llama3:latest", 4_000_000_000),
                ("codellama:latest", 3_000_000_000),
            )
        )
        priorities = {"llama3:latest": "unknown_level"}
        candidates = manager.get_eviction_candidates({}, priorities)
        # Both priority 0, sort by -size
        assert candidates[0] == "llama3:latest"
        assert candidates[1] == "codellama:latest"

    def test_missing_pending_defaults_to_zero(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("llama3:latest", 1000)),
        )
        candidates = manager.get_eviction_candidates({}, {})
        assert candidates == ["llama3:latest"]


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


class TestRefresh:
    async def test_refresh_success(self):
        manager = _make_manager()
        ps_data = _ps_response(("llama3:latest", 4_661_224_676))

        mock_resp = MagicMock()
        mock_resp.json.return_value = ps_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        loaded = manager.get_loaded_models()
        assert "llama3:latest" in loaded
        assert loaded["llama3:latest"].size_vram == 4_661_224_676

    async def test_refresh_http_error(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("existing:latest", 1000)),
        )

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPError("connection error"),
        )

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        assert manager.is_loaded("existing:latest") is True

    async def test_refresh_updates_state(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("old:latest", 1000)),
        )

        ps_data = _ps_response(("new:latest", 2000))
        mock_resp = MagicMock()
        mock_resp.json.return_value = ps_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        assert not manager.is_loaded("old:latest")
        assert manager.is_loaded("new:latest")

    async def test_refresh_calls_correct_endpoint(self):
        config = _make_config(
            total_ram="32GB",
            host="http://custom:9999",
        )
        with patch("ollama_marshal.memory.psutil"):
            manager = MemoryManager(config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        mock_client.get.assert_called_once_with(
            "http://custom:9999/api/ps",
            timeout=10,
        )


# ---------------------------------------------------------------------------
# get_loaded_models returns copy
# ---------------------------------------------------------------------------


class TestGetLoadedModels:
    def test_returns_copy(self):
        manager = _make_manager()
        manager._update_from_ps(
            _ps_response(("llama3:latest", 1000)),
        )
        models = manager.get_loaded_models()
        models["injected"] = LoadedModel(name="injected", size_vram=999)
        assert "injected" not in manager.get_loaded_models()

    def test_empty_when_nothing_loaded(self):
        manager = _make_manager()
        assert manager.get_loaded_models() == {}


# ---------------------------------------------------------------------------
# budget property
# ---------------------------------------------------------------------------


class TestBudgetProperty:
    def test_budget_accessible(self):
        manager = _make_manager(total_ram="64GB")
        budget = manager.budget
        assert isinstance(budget, MemoryBudget)
        assert budget.total_ram == 64 * 1024**3


# ---------------------------------------------------------------------------
# start_polling / stop_polling / _poll_loop
# ---------------------------------------------------------------------------


class TestStartPolling:
    async def test_creates_task(self):
        manager = _make_manager()
        assert manager._poll_task is None

        with patch.object(manager, "_poll_loop", new_callable=AsyncMock):
            await manager.start_polling()
            assert manager._poll_task is not None
            # Clean up
            await manager.stop_polling()

    async def test_task_is_running(self):
        manager = _make_manager()

        with patch.object(manager, "_poll_loop", new_callable=AsyncMock):
            await manager.start_polling()
            task = manager._poll_task
            assert task is not None
            assert not task.done()
            # Clean up
            await manager.stop_polling()


class TestStopPolling:
    async def test_cancels_task(self):
        manager = _make_manager()

        with patch.object(manager, "_poll_loop", new_callable=AsyncMock):
            await manager.start_polling()
            task = manager._poll_task
            assert task is not None

            await manager.stop_polling()
            assert manager._poll_task is None
            assert task.cancelled() or task.done()

    async def test_stop_without_start(self):
        manager = _make_manager()
        assert manager._poll_task is None

        # Should not raise
        await manager.stop_polling()
        assert manager._poll_task is None

    async def test_stop_idempotent(self):
        manager = _make_manager()

        with patch.object(manager, "_poll_loop", new_callable=AsyncMock):
            await manager.start_polling()
            await manager.stop_polling()
            # Second stop is safe
            await manager.stop_polling()
            assert manager._poll_task is None


class TestPollLoop:
    async def test_poll_loop_calls_refresh(self):
        manager = _make_manager()

        call_count = 0

        async def mock_refresh():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        with patch.object(manager, "refresh", side_effect=mock_refresh):
            manager._poll_interval = 0.01
            try:
                await manager._poll_loop()
            except asyncio.CancelledError:
                pass

        assert call_count >= 2

    async def test_poll_loop_handles_errors(self):
        manager = _make_manager()

        call_count = 0

        async def mock_refresh():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            if call_count >= 3:
                raise asyncio.CancelledError

        with patch.object(manager, "refresh", side_effect=mock_refresh):
            manager._poll_interval = 0.01
            try:
                await manager._poll_loop()
            except asyncio.CancelledError:
                pass

        # Should have continued past the error
        assert call_count >= 3

    async def test_poll_loop_reraises_cancelled(self):
        manager = _make_manager()

        async def mock_refresh():
            raise asyncio.CancelledError

        with patch.object(manager, "refresh", side_effect=mock_refresh):
            manager._poll_interval = 0.01
            try:
                await manager._poll_loop()
                raised = False
            except asyncio.CancelledError:
                raised = True

        assert raised is True


# ---------------------------------------------------------------------------
# Allocated num_ctx tracking + reload-on-need (Surface C1 Dim 4)
# ---------------------------------------------------------------------------


class TestAllocatedNumCtx:
    def test_initial_get_returns_none(self):
        manager = _make_manager()
        assert manager.get_allocated_num_ctx("any:model") is None

    def test_record_and_get(self):
        manager = _make_manager()
        manager.record_allocated_num_ctx("llama3:latest", 16384)
        assert manager.get_allocated_num_ctx("llama3:latest") == 16384

    def test_record_overwrites(self):
        manager = _make_manager()
        manager.record_allocated_num_ctx("llama3:latest", 4096)
        manager.record_allocated_num_ctx("llama3:latest", 32768)
        assert manager.get_allocated_num_ctx("llama3:latest") == 32768

    def test_unload_clears_allocation(self):
        # An unexpected unload (Ollama-side) should drop the allocation
        # so the next preload doesn't reuse a stale recorded num_ctx.
        manager = _make_manager()
        manager._loaded_models = {
            "llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)
        }
        manager.record_allocated_num_ctx("llama3:latest", 16384)
        # Simulate /api/ps now reporting no models.
        manager._update_from_ps({"models": []})
        assert manager.get_allocated_num_ctx("llama3:latest") is None


class TestNeedsReload:
    def test_false_when_not_loaded(self):
        # If the model isn't loaded, the upcoming preload will use the
        # right size — no reload needed.
        manager = _make_manager()
        assert manager.needs_reload("not-loaded:x", 16384) is False

    def test_false_when_no_allocation_recorded(self):
        # Defensive: we can't tell, so don't trigger spurious reloads.
        manager = _make_manager()
        manager._loaded_models = {
            "llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)
        }
        assert manager.needs_reload("llama3:latest", 16384) is False

    def test_false_when_requested_fits(self):
        manager = _make_manager()
        manager._loaded_models = {
            "llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)
        }
        manager.record_allocated_num_ctx("llama3:latest", 32768)
        assert manager.needs_reload("llama3:latest", 16384) is False
        assert manager.needs_reload("llama3:latest", 32768) is False

    def test_true_when_requested_exceeds(self):
        manager = _make_manager()
        manager._loaded_models = {
            "llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)
        }
        manager.record_allocated_num_ctx("llama3:latest", 8192)
        assert manager.needs_reload("llama3:latest", 16384) is True


# ---------------------------------------------------------------------------
# Unexpected-unload detection (Surface C2)
# ---------------------------------------------------------------------------


class TestUnexpectedUnloadDetection:
    def test_intended_unload_does_not_trigger(self):
        manager = _make_manager()
        manager._loaded_models = {
            "llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)
        }
        manager.mark_intended_unload("llama3:latest")
        manager._update_from_ps({"models": []})
        assert manager.unexpected_unloads_observed == 0

    def test_unexpected_unload_increments_counter(self):
        # Marshal didn't ask for the unload — Ollama did it.
        manager = _make_manager()
        manager._loaded_models = {
            "mistral:latest": LoadedModel(name="mistral:latest", size_vram=1)
        }
        manager._update_from_ps({"models": []})
        assert manager.unexpected_unloads_observed == 1

    def test_take_count_resets(self):
        manager = _make_manager()
        manager._loaded_models = {
            "x:1": LoadedModel(name="x:1", size_vram=1),
            "x:2": LoadedModel(name="x:2", size_vram=1),
        }
        manager._update_from_ps({"models": []})
        assert manager.take_unexpected_unload_count() == 2
        # Subsequent take returns 0 (resets on read).
        assert manager.take_unexpected_unload_count() == 0

    def test_intended_set_consumed_only_once(self):
        # Marking intended-unload should consume the marker so a later
        # unexpected unload of the same name still triggers.
        manager = _make_manager()
        manager._loaded_models = {"x:1": LoadedModel(name="x:1", size_vram=1)}
        manager.mark_intended_unload("x:1")
        manager._update_from_ps({"models": []})  # intended — no count
        assert manager.unexpected_unloads_observed == 0

        # Now load + unexpected-unload again; should count.
        manager._update_from_ps({"models": [{"name": "x:1", "size_vram": 1}]})
        manager._update_from_ps({"models": []})
        assert manager.unexpected_unloads_observed == 1
