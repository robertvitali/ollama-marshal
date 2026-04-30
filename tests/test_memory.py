from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from ollama_marshal.config import MarshalConfig, MemoryConfig, OllamaConfig
from ollama_marshal.memory import LoadedModel, MemoryBudget, MemoryManager

PATCH_ASYNC_CLIENT = "ollama_marshal.memory.httpx.AsyncClient"

# Primary instance URL for v0.5.0 per-instance state. Single-instance
# setups (these tests) all use the legacy ``ollama.host`` value, which
# the config validator backfills as ``instances[0].url``.
_PRIMARY_URL = "http://localhost:11434"


def _apply_ps(manager, ps_data, instance_url=_PRIMARY_URL):
    """Call _update_from_ps with the primary instance URL.

    Tests pre-v0.5.0 called ``_update_from_ps(ps_data)`` directly.
    The per-instance API now requires the URL up front.
    """
    manager._update_from_ps(instance_url, ps_data)


def _set_loaded(manager, models, instance_url=_PRIMARY_URL):
    """Seed manager._loaded_models with a flat ``{name: LoadedModel}`` dict.

    The internal structure is ``{instance_url: {name: LoadedModel}}``;
    this helper handles the wrapping so test assertions stay readable.
    """
    manager._loaded_models = {instance_url: dict(models)}


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
        _apply_ps(manager, ps_data)

        loaded = manager.get_loaded_models()
        assert len(loaded) == 2
        assert loaded["llama3:latest"].size_vram == 4_661_224_676
        assert loaded["mistral:latest"].size_vram == 4_109_865_159

    def test_replaces_previous_state(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            _ps_response(("llama3:latest", 1000)),
        )
        _apply_ps(
            manager,
            _ps_response(("mistral:latest", 2000)),
        )

        loaded = manager.get_loaded_models()
        assert "llama3:latest" not in loaded
        assert "mistral:latest" in loaded

    def test_empty_ps_clears_models(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            _ps_response(("llama3:latest", 1000)),
        )
        _apply_ps(manager, {"models": []})
        assert manager.get_loaded_models() == {}

    def test_no_models_key(self):
        manager = _make_manager()
        _apply_ps(manager, {})
        assert manager.get_loaded_models() == {}

    def test_preserves_expires_at(self):
        manager = _make_manager()
        ps_data = _ps_response(
            ("llama3:latest", 1000, "2026-12-31T00:00:00Z"),
        )
        _apply_ps(manager, ps_data)

        loaded = manager.get_loaded_models()
        assert loaded["llama3:latest"].expires_at == "2026-12-31T00:00:00Z"

    def test_missing_size_vram_defaults_to_zero(self):
        manager = _make_manager()
        ps_data = {"models": [{"name": "llama3:latest"}]}
        _apply_ps(manager, ps_data)

        loaded = manager.get_loaded_models()
        assert loaded["llama3:latest"].size_vram == 0

    def test_missing_expires_at_defaults_to_empty(self):
        manager = _make_manager()
        ps_data = {
            "models": [
                {"name": "llama3:latest", "size_vram": 1000},
            ],
        }
        _apply_ps(manager, ps_data)

        loaded = manager.get_loaded_models()
        assert loaded["llama3:latest"].expires_at == ""


# ---------------------------------------------------------------------------
# is_loaded / used_vram / available_vram
# ---------------------------------------------------------------------------


class TestIsLoaded:
    def test_loaded_model(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            _ps_response(("llama3:latest", 1000)),
        )
        assert manager.is_loaded("llama3:latest") is True

    def test_not_loaded(self):
        manager = _make_manager()
        assert manager.is_loaded("llama3:latest") is False

    def test_after_unload(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            _ps_response(("llama3:latest", 1000)),
        )
        _apply_ps(manager, {"models": []})
        assert manager.is_loaded("llama3:latest") is False


class TestUsedVram:
    def test_no_models(self):
        manager = _make_manager()
        assert manager.used_vram() == 0

    def test_single_model(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            _ps_response(("llama3:latest", 4_000_000_000)),
        )
        assert manager.used_vram() == 4_000_000_000

    def test_multiple_models(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            _ps_response(
                ("llama3:latest", 4_000_000_000),
                ("mistral:latest", 3_000_000_000),
            ),
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
        _apply_ps(
            manager,
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
        _apply_ps(
            manager,
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
        _apply_ps(
            manager,
            _ps_response(
                ("llama3:latest", 4_000_000_000),
                ("codellama:latest", 3_000_000_000),
                ("mistral:latest", 5_000_000_000),
            ),
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
        _apply_ps(
            manager,
            _ps_response(
                ("llama3:latest", 4_000_000_000),
                ("codellama:latest", 3_000_000_000),
            ),
        )
        priorities = {"llama3:latest": "unknown_level"}
        candidates = manager.get_eviction_candidates({}, priorities)
        # Both priority 0, sort by -size
        assert candidates[0] == "llama3:latest"
        assert candidates[1] == "codellama:latest"

    def test_missing_pending_defaults_to_zero(self):
        manager = _make_manager()
        _apply_ps(
            manager,
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
        _apply_ps(
            manager,
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
        _apply_ps(
            manager,
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
        _apply_ps(
            manager,
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
        _set_loaded(
            manager,
            {"llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)},
        )
        manager.record_allocated_num_ctx("llama3:latest", 16384)
        # Simulate /api/ps now reporting no models.
        _apply_ps(manager, {"models": []})
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
        _set_loaded(
            manager,
            {"llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)},
        )
        assert manager.needs_reload("llama3:latest", 16384) is False

    def test_false_when_requested_fits(self):
        manager = _make_manager()
        _set_loaded(
            manager,
            {"llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)},
        )
        manager.record_allocated_num_ctx("llama3:latest", 32768)
        assert manager.needs_reload("llama3:latest", 16384) is False
        assert manager.needs_reload("llama3:latest", 32768) is False

    def test_true_when_requested_exceeds(self):
        manager = _make_manager()
        _set_loaded(
            manager,
            {"llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)},
        )
        manager.record_allocated_num_ctx("llama3:latest", 8192)
        assert manager.needs_reload("llama3:latest", 16384) is True

    def test_zero_allocation_is_sentinel_always_needs_reload(self):
        # Allocation==0 is the post-failed-preload sentinel. The model
        # is loaded but we don't know what slot Ollama actually has.
        # `needs_reload` must return True regardless of requested size
        # so the scheduler tries to reload again instead of silently
        # dispatching against an unknown slot.
        manager = _make_manager()
        _set_loaded(
            manager,
            {"llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)},
        )
        manager.record_allocated_num_ctx("llama3:latest", 0)
        assert manager.needs_reload("llama3:latest", 1024) is True
        assert manager.needs_reload("llama3:latest", 65536) is True


class TestUpdateFromPsShapeRobustness:
    """Defensive parsing of /api/ps response.

    /api/ps comes from a process we don't fully control (Ollama, or a
    proxy in front of it). A malformed entry must NOT crash the poll
    loop. A crash-then-broad-except would leave _loaded_models stale,
    silently turning unexpected_unload detection into a false negative
    on the very signal Surface C2 was meant to catch.
    """

    def test_models_not_a_list_treated_as_empty(self):
        manager = _make_manager()
        _set_loaded(
            manager,
            {"x:1": LoadedModel(name="x:1", size_vram=1)},
        )
        _apply_ps(manager, {"models": "not a list"})
        assert manager.get_loaded_models() == {}

    def test_models_is_none_treated_as_empty(self):
        manager = _make_manager()
        _apply_ps(manager, {"models": None})
        assert manager.get_loaded_models() == {}

    def test_non_dict_entries_skipped(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            {
                "models": [
                    "string-not-dict",
                    None,
                    42,
                    {"name": "valid:1", "size_vram": 1000},
                ]
            },
        )
        loaded = manager.get_loaded_models()
        assert list(loaded) == ["valid:1"]

    def test_missing_name_skipped(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            {"models": [{"size_vram": 1000}, {"name": "valid:1", "size_vram": 1}]},
        )
        loaded = manager.get_loaded_models()
        assert list(loaded) == ["valid:1"]

    def test_non_string_name_skipped(self):
        manager = _make_manager()
        _apply_ps(
            manager,
            {
                "models": [
                    {"name": 12345, "size_vram": 1},
                    {"name": "ok:1", "size_vram": 1},
                ]
            },
        )
        loaded = manager.get_loaded_models()
        assert list(loaded) == ["ok:1"]

    def test_garbage_size_vram_defaults_to_zero(self):
        manager = _make_manager()
        _apply_ps(manager, {"models": [{"name": "x:1", "size_vram": "not-a-number"}]})
        assert manager.get_loaded_models()["x:1"].size_vram == 0


# ---------------------------------------------------------------------------
# Unexpected-unload detection (Surface C2)
# ---------------------------------------------------------------------------


class TestUnexpectedUnloadDetection:
    def test_intended_unload_does_not_trigger(self):
        manager = _make_manager()
        _set_loaded(
            manager,
            {"llama3:latest": LoadedModel(name="llama3:latest", size_vram=1)},
        )
        manager.mark_intended_unload("llama3:latest")
        _apply_ps(manager, {"models": []})
        assert manager.unexpected_unloads_observed == 0

    def test_unexpected_unload_increments_counter(self):
        # Marshal didn't ask for the unload — Ollama did it.
        manager = _make_manager()
        _set_loaded(
            manager,
            {"mistral:latest": LoadedModel(name="mistral:latest", size_vram=1)},
        )
        _apply_ps(manager, {"models": []})
        assert manager.unexpected_unloads_observed == 1

    def test_take_count_resets(self):
        manager = _make_manager()
        _set_loaded(
            manager,
            {
                "x:1": LoadedModel(name="x:1", size_vram=1),
                "x:2": LoadedModel(name="x:2", size_vram=1),
            },
        )
        _apply_ps(manager, {"models": []})
        assert manager.take_unexpected_unload_count() == 2
        # Subsequent take returns 0 (resets on read).
        assert manager.take_unexpected_unload_count() == 0

    def test_intended_set_consumed_only_once(self):
        # Marking intended-unload should consume the marker so a later
        # unexpected unload of the same name still triggers.
        manager = _make_manager()
        _set_loaded(
            manager,
            {"x:1": LoadedModel(name="x:1", size_vram=1)},
        )
        manager.mark_intended_unload("x:1")
        _apply_ps(manager, {"models": []})  # intended — no count
        assert manager.unexpected_unloads_observed == 0

        # Now load + unexpected-unload again; should count.
        _apply_ps(manager, {"models": [{"name": "x:1", "size_vram": 1}]})
        _apply_ps(manager, {"models": []})
        assert manager.unexpected_unloads_observed == 1


# ---------------------------------------------------------------------------
# Multi-instance state (v0.5.0+)
# ---------------------------------------------------------------------------


def _multi_instance_config(total_ram="64GB"):
    """Build a 3-instance config (f16 / q8_0 / q4_0)."""
    from ollama_marshal.config import KVCacheType, OllamaInstance

    return MarshalConfig(
        instances=[
            OllamaInstance(url="http://localhost:11434", kv_cache_type=KVCacheType.F16),
            OllamaInstance(
                url="http://localhost:11444", kv_cache_type=KVCacheType.Q8_0
            ),
            OllamaInstance(
                url="http://localhost:11454", kv_cache_type=KVCacheType.Q4_0
            ),
        ],
        memory=MemoryConfig(total_ram=total_ram),
    )


def _make_multi_manager(total_ram="64GB"):
    config = _multi_instance_config(total_ram=total_ram)
    with patch("ollama_marshal.memory.psutil"):
        return MemoryManager(config)


class TestMultiInstanceLoadedTracking:
    """Per-instance ``_loaded_models`` map + flat-view accessors."""

    def test_attribution_to_correct_instance(self):
        manager = _make_multi_manager()
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "f16-only:x", "size_vram": 1}]},
        )
        manager._update_from_ps(
            "http://localhost:11444",
            {"models": [{"name": "q8-only:x", "size_vram": 2}]},
        )
        # Per-instance lookup
        assert manager.is_loaded_on("f16-only:x", "http://localhost:11434")
        assert not manager.is_loaded_on("f16-only:x", "http://localhost:11444")
        assert manager.is_loaded_on("q8-only:x", "http://localhost:11444")
        # Flat view sees both
        flat = manager.get_loaded_models()
        assert set(flat.keys()) == {"f16-only:x", "q8-only:x"}
        assert flat["q8-only:x"].instance_url == "http://localhost:11444"

    def test_loaded_on_returns_url_to_set_map(self):
        manager = _make_multi_manager()
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "x:1", "size_vram": 1}]},
        )
        m = manager.loaded_on()
        assert m["http://localhost:11434"] == {"x:1"}
        assert m["http://localhost:11444"] == set()
        assert m["http://localhost:11454"] == set()

    def test_get_loaded_models_on_isolates_instance(self):
        manager = _make_multi_manager()
        manager._update_from_ps(
            "http://localhost:11444",
            {"models": [{"name": "shared:x", "size_vram": 1}]},
        )
        only_q8 = manager.get_loaded_models_on("http://localhost:11444")
        assert "shared:x" in only_q8
        assert manager.get_loaded_models_on("http://localhost:11434") == {}

    def test_find_instance_for_walks_in_precision_order(self):
        manager = _make_multi_manager()
        # Same model on both f16 and q8 (mid-promotion); higher-precision wins.
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "shared:x", "size_vram": 1}]},
        )
        manager._update_from_ps(
            "http://localhost:11444",
            {"models": [{"name": "shared:x", "size_vram": 2}]},
        )
        assert manager.find_instance_for("shared:x") == "http://localhost:11434"

    def test_find_instance_for_returns_none_when_unloaded(self):
        manager = _make_multi_manager()
        assert manager.find_instance_for("nope:x") is None

    def test_used_vram_sums_across_instances(self):
        manager = _make_multi_manager()
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "a:1", "size_vram": 5_000_000_000}]},
        )
        manager._update_from_ps(
            "http://localhost:11444",
            {"models": [{"name": "b:1", "size_vram": 3_000_000_000}]},
        )
        # GLOBAL budget — sum across instances on Mac unified memory.
        assert manager.used_vram() == 8_000_000_000


class TestProbeFit:
    """``probe_fit`` is the routing-aware fit answer.

    Three outcomes (see ``routing.FitProbe``):
    1. ``fits=True`` — global budget has room.
    2. ``fits=False, would_evict_non_idle=True`` — fits only by
       evicting work on this instance.
    3. ``fits=False, would_evict_non_idle=False`` — only idle models
       on this instance could be freed.
    """

    def test_fits_when_budget_available(self):
        manager = _make_multi_manager(total_ram="64GB")
        probe = manager.probe_fit(
            instance_url="http://localhost:11434",
            model_size=4 * 1024**3,
            non_idle_loaded_on_instance=set(),
        )
        assert probe.fits is True
        assert probe.would_evict_non_idle is False

    def test_would_evict_non_idle_when_pinned_loaded(self):
        manager = _make_multi_manager(total_ram="8GB")
        # Soak the global budget so probe_fit takes the eviction path.
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "pinned:x", "size_vram": 4 * 1024**3}]},
        )
        probe = manager.probe_fit(
            instance_url="http://localhost:11434",
            model_size=4 * 1024**3,
            non_idle_loaded_on_instance={"pinned:x"},
        )
        assert probe.fits is False
        assert probe.would_evict_non_idle is True

    def test_only_idle_eviction_needed(self):
        manager = _make_multi_manager(total_ram="8GB")
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "idle:x", "size_vram": 4 * 1024**3}]},
        )
        probe = manager.probe_fit(
            instance_url="http://localhost:11434",
            model_size=4 * 1024**3,
            non_idle_loaded_on_instance=set(),
        )
        assert probe.fits is False
        assert probe.would_evict_non_idle is False


class TestMultiInstanceAllocations:
    """Per-instance ``_allocated_num_ctx`` + ``_intended_unloads``."""

    def test_record_per_instance_isolation(self):
        manager = _make_multi_manager()
        manager.record_allocated_num_ctx(
            "x:1", 16384, instance_url="http://localhost:11434"
        )
        manager.record_allocated_num_ctx(
            "x:1", 4096, instance_url="http://localhost:11444"
        )
        assert (
            manager.get_allocated_num_ctx("x:1", instance_url="http://localhost:11434")
            == 16384
        )
        assert (
            manager.get_allocated_num_ctx("x:1", instance_url="http://localhost:11444")
            == 4096
        )

    def test_get_allocated_num_ctx_walks_instances_when_url_unset(self):
        manager = _make_multi_manager()
        # Only q8 has an allocation; the no-URL lookup should find it
        # by walking instances in declared (precision) order.
        manager.record_allocated_num_ctx(
            "x:1", 8192, instance_url="http://localhost:11444"
        )
        assert manager.get_allocated_num_ctx("x:1") == 8192

    def test_needs_reload_resolves_instance_when_url_unset(self):
        manager = _make_multi_manager()
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "x:1", "size_vram": 1}]},
        )
        manager.record_allocated_num_ctx(
            "x:1", 4096, instance_url="http://localhost:11434"
        )
        # No URL — resolve to the instance currently holding the model.
        assert manager.needs_reload("x:1", 16384) is True
        assert manager.needs_reload("x:1", 2048) is False

    def test_needs_reload_returns_false_when_unloaded_anywhere(self):
        manager = _make_multi_manager()
        assert manager.needs_reload("never-loaded:x", 16384) is False

    def test_mark_intended_unload_broadcasts_when_url_omitted(self):
        manager = _make_multi_manager()
        # Same model on two instances (transient, mid-promotion).
        manager._update_from_ps(
            "http://localhost:11434",
            {"models": [{"name": "shared:x", "size_vram": 1}]},
        )
        manager._update_from_ps(
            "http://localhost:11444",
            {"models": [{"name": "shared:x", "size_vram": 2}]},
        )
        manager.mark_intended_unload("shared:x")  # no URL
        # Both intended sets see the marker.
        manager._update_from_ps("http://localhost:11434", {"models": []})
        manager._update_from_ps("http://localhost:11444", {"models": []})
        assert manager.unexpected_unloads_observed == 0


class TestPerInstanceRefresh:
    """``refresh`` fans out across instances; per-instance polls are isolated."""

    async def test_refresh_polls_each_instance(self):
        manager = _make_multi_manager()
        urls_hit: list[str] = []

        async def fake_get(url, timeout=10):
            urls_hit.append(url)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": []}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        # Order is deterministic — instances are sorted by precision.
        assert urls_hit == [
            "http://localhost:11434/api/ps",
            "http://localhost:11444/api/ps",
            "http://localhost:11454/api/ps",
        ]

    async def test_one_instance_failure_does_not_abort_others(self):
        manager = _make_multi_manager()

        async def fake_get(url, timeout=10):
            if "11444" in url:
                raise httpx.HTTPError("q8 down")
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": [{"name": "x:1", "size_vram": 1}]}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        # f16 + q4_0 both saw their model load even though q8_0 errored.
        assert manager.is_loaded_on("x:1", "http://localhost:11434")
        assert manager.is_loaded_on("x:1", "http://localhost:11454")
        assert not manager.is_loaded_on("x:1", "http://localhost:11444")


# ---------------------------------------------------------------------------
# Per-instance reachability (v0.5.0+)
# ---------------------------------------------------------------------------


class TestIsInstanceReachable:
    """``is_instance_reachable`` flips per-poll based on outcome.

    Used by /api/marshal/status to expose per-instance health to
    operators. Starts False (no poll seen yet); flips True after a
    successful poll, False after a poll error.
    """

    def test_starts_false_before_first_poll(self):
        manager = _make_multi_manager()
        for inst in manager.instances:
            assert manager.is_instance_reachable(inst.url) is False

    def test_unknown_url_is_unreachable(self):
        # Defensive: a status payload consumer asking about an instance
        # that's not in the configured list gets False, not a KeyError.
        manager = _make_multi_manager()
        assert manager.is_instance_reachable("http://nope:9999") is False

    async def test_flips_true_on_successful_poll(self):
        manager = _make_multi_manager()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        # All three instances polled successfully → all reachable.
        for inst in manager.instances:
            assert manager.is_instance_reachable(inst.url) is True

    async def test_flips_false_on_poll_error(self):
        manager = _make_multi_manager()

        async def fake_get(url, timeout=10):
            # f16 succeeds, q8 errors, q4 succeeds.
            if "11444" in url:
                raise httpx.HTTPError("q8 down")
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": []}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        assert manager.is_instance_reachable("http://localhost:11434") is True
        assert manager.is_instance_reachable("http://localhost:11444") is False
        assert manager.is_instance_reachable("http://localhost:11454") is True

    async def test_flips_back_on_recovery(self):
        # Instance recovers between polls — reachability flips back to
        # True. Per-poll outcome is the source of truth, no time-decay.
        manager = _make_multi_manager()
        call_state = {"calls": 0}

        async def fake_get(url, timeout=10):
            call_state["calls"] += 1
            # First batch: q8 errors. Second batch: all succeed.
            if "11444" in url and call_state["calls"] <= 3:
                raise httpx.HTTPError("transient")
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"models": []}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        assert manager.is_instance_reachable("http://localhost:11444") is False

        # Second poll — q8 recovers.
        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await manager.refresh()

        assert manager.is_instance_reachable("http://localhost:11444") is True
