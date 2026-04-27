from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollama_marshal.registry import ModelRegistry

PATCH_ASYNC_CLIENT = "ollama_marshal.registry.httpx.AsyncClient"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(
        ollama_host="http://localhost:11434",
        registry_path=tmp_path / "model_sizes.json",
    )


@pytest.fixture
def populated_cache(tmp_path):
    cache_path = tmp_path / "model_sizes.json"
    data = {
        "llama3:latest": 4_661_224_676,
        "codellama:latest": 3_825_819_519,
    }
    cache_path.write_text(json.dumps(data))
    return cache_path


@pytest.fixture
def populated_registry(tmp_path, populated_cache):
    reg = ModelRegistry(
        ollama_host="http://localhost:11434",
        registry_path=populated_cache,
    )
    reg._load_cache()
    return reg


# ---------------------------------------------------------------------------
# _load_cache / _save_cache
# ---------------------------------------------------------------------------


class TestLoadCache:
    def test_load_existing_cache(self, tmp_path):
        cache_path = tmp_path / "model_sizes.json"
        data = {"llama3:latest": 4_661_224_676}
        cache_path.write_text(json.dumps(data))
        reg = ModelRegistry(registry_path=cache_path)
        reg._load_cache()
        assert reg._sizes == {"llama3:latest": 4_661_224_676}

    def test_load_empty_cache_file(self, tmp_path):
        cache_path = tmp_path / "model_sizes.json"
        cache_path.write_text("{}")
        reg = ModelRegistry(registry_path=cache_path)
        reg._load_cache()
        assert reg._sizes == {}

    def test_load_nonexistent_cache(self, tmp_path):
        cache_path = tmp_path / "nonexistent.json"
        reg = ModelRegistry(registry_path=cache_path)
        reg._load_cache()
        assert reg._sizes == {}

    def test_load_corrupt_json(self, tmp_path):
        cache_path = tmp_path / "model_sizes.json"
        cache_path.write_text("not valid json{{{")
        reg = ModelRegistry(registry_path=cache_path)
        reg._load_cache()
        assert reg._sizes == {}

    def test_load_non_dict_json(self, tmp_path):
        cache_path = tmp_path / "model_sizes.json"
        cache_path.write_text(json.dumps(["a", "list"]))
        reg = ModelRegistry(registry_path=cache_path)
        reg._load_cache()
        assert reg._sizes == {}

    def test_load_cache_coerces_values_to_int(self, tmp_path):
        cache_path = tmp_path / "model_sizes.json"
        cache_path.write_text(json.dumps({"model:latest": 1234.0}))
        reg = ModelRegistry(registry_path=cache_path)
        reg._load_cache()
        assert reg._sizes["model:latest"] == 1234
        assert isinstance(reg._sizes["model:latest"], int)


class TestSaveCache:
    def test_save_creates_file(self, registry):
        registry._sizes = {"llama3:latest": 5_000_000_000}
        registry._save_cache()
        data = json.loads(registry.registry_path.read_text())
        assert data == {"llama3:latest": 5_000_000_000}

    def test_save_creates_parent_dirs(self, tmp_path):
        cache_path = tmp_path / "subdir" / "deep" / "model_sizes.json"
        reg = ModelRegistry(registry_path=cache_path)
        reg._sizes = {"m:latest": 100}
        reg._save_cache()
        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert data == {"m:latest": 100}

    def test_save_overwrites_existing(self, tmp_path):
        cache_path = tmp_path / "model_sizes.json"
        cache_path.write_text(json.dumps({"old:model": 999}))
        reg = ModelRegistry(registry_path=cache_path)
        reg._sizes = {"new:model": 1234}
        reg._save_cache()
        data = json.loads(cache_path.read_text())
        assert data == {"new:model": 1234}


# ---------------------------------------------------------------------------
# _extract_model_vram
# ---------------------------------------------------------------------------


class TestExtractModelVram:
    def test_model_found_by_name(self):
        ps_data = {"models": [{"name": "llama3:latest", "size_vram": 4_661_224_676}]}
        result = ModelRegistry._extract_model_vram(ps_data, "llama3:latest")
        assert result == 4_661_224_676

    def test_model_found_by_model_field(self):
        ps_data = {
            "models": [
                {
                    "model": "llama3:latest",
                    "name": "other",
                    "size_vram": 1234,
                }
            ]
        }
        result = ModelRegistry._extract_model_vram(ps_data, "llama3:latest")
        assert result == 1234

    def test_model_not_found(self):
        ps_data = {"models": [{"name": "codellama:latest", "size_vram": 3_000_000_000}]}
        result = ModelRegistry._extract_model_vram(ps_data, "llama3:latest")
        assert result is None

    def test_empty_models_list(self):
        ps_data = {"models": []}
        result = ModelRegistry._extract_model_vram(ps_data, "llama3:latest")
        assert result is None

    def test_no_models_key(self):
        ps_data = {}
        result = ModelRegistry._extract_model_vram(ps_data, "llama3:latest")
        assert result is None

    def test_missing_size_vram_defaults_to_zero(self):
        ps_data = {"models": [{"name": "llama3:latest"}]}
        result = ModelRegistry._extract_model_vram(ps_data, "llama3:latest")
        assert result == 0

    def test_multiple_models_returns_correct_one(self):
        ps_data = {
            "models": [
                {"name": "codellama:latest", "size_vram": 1111},
                {"name": "llama3:latest", "size_vram": 2222},
                {"name": "mistral:latest", "size_vram": 3333},
            ]
        }
        result = ModelRegistry._extract_model_vram(ps_data, "llama3:latest")
        assert result == 2222


# ---------------------------------------------------------------------------
# get_model_size / is_benchmarked / list_models / remove_model
# ---------------------------------------------------------------------------


class TestGetModelSize:
    def test_cached_model(self, populated_registry):
        size = populated_registry.get_model_size("llama3:latest")
        assert size == 4_661_224_676

    def test_uncached_model(self, populated_registry):
        assert populated_registry.get_model_size("unknown:latest") is None

    def test_empty_registry(self, registry):
        assert registry.get_model_size("anything") is None


class TestIsBenchmarked:
    def test_benchmarked(self, populated_registry):
        assert populated_registry.is_benchmarked("llama3:latest") is True

    def test_not_benchmarked(self, populated_registry):
        assert populated_registry.is_benchmarked("unknown:latest") is False

    def test_empty_registry(self, registry):
        assert registry.is_benchmarked("anything") is False


class TestListModels:
    def test_list_populated(self, populated_registry):
        models = populated_registry.list_models()
        assert models == {
            "llama3:latest": 4_661_224_676,
            "codellama:latest": 3_825_819_519,
        }

    def test_list_returns_copy(self, populated_registry):
        models = populated_registry.list_models()
        models["extra:model"] = 999
        assert "extra:model" not in populated_registry.list_models()

    def test_list_empty(self, registry):
        assert registry.list_models() == {}


class TestRemoveModel:
    def test_remove_existing(self, populated_registry):
        populated_registry.remove_model("llama3:latest")
        assert "llama3:latest" not in populated_registry._sizes
        data = json.loads(populated_registry.registry_path.read_text())
        assert "llama3:latest" not in data

    def test_remove_nonexistent_no_error(self, populated_registry):
        populated_registry.remove_model("nonexistent:latest")
        assert len(populated_registry._sizes) == 2


# ---------------------------------------------------------------------------
# _fetch_model_list
# ---------------------------------------------------------------------------


def _mock_tags_response(models):
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": m} for m in models]}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_async_client(mock_client):
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestFetchModelList:
    async def test_fetches_model_names(self, registry):
        mock_resp = _mock_tags_response(
            ["llama3:latest", "mistral:latest"],
        )
        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry._fetch_model_list()

        assert result == ["llama3:latest", "mistral:latest"]

    async def test_raises_on_http_error(self, registry):
        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPError("connection error"),
        )

        with (
            patch(PATCH_ASYNC_CLIENT, return_value=mock_client),
            pytest.raises(httpx.HTTPError),
        ):
            await registry._fetch_model_list()

    async def test_empty_model_list(self, registry):
        mock_resp = _mock_tags_response([])
        mock_client = _mock_async_client(AsyncMock())
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry._fetch_model_list()

        assert result == []


# ---------------------------------------------------------------------------
# _sync_with_ollama
# ---------------------------------------------------------------------------


class TestSyncWithOllama:
    async def test_removes_stale_models(self, populated_registry):
        with patch.object(
            populated_registry,
            "_fetch_model_list",
            new_callable=AsyncMock,
            return_value=["llama3:latest"],
        ):
            await populated_registry._sync_with_ollama()

        assert "codellama:latest" not in populated_registry._sizes
        assert "llama3:latest" in populated_registry._sizes

    async def test_identifies_new_models(self, populated_registry):
        with patch.object(
            populated_registry,
            "_fetch_model_list",
            new_callable=AsyncMock,
            return_value=[
                "llama3:latest",
                "codellama:latest",
                "mistral:latest",
            ],
        ):
            await populated_registry._sync_with_ollama()

        # New model should NOT be added to _sizes (only identified)
        assert "mistral:latest" not in populated_registry._sizes
        assert "llama3:latest" in populated_registry._sizes
        assert "codellama:latest" in populated_registry._sizes

    async def test_no_changes_no_save(self, populated_registry):
        with (
            patch.object(
                populated_registry,
                "_fetch_model_list",
                new_callable=AsyncMock,
                return_value=["llama3:latest", "codellama:latest"],
            ),
            patch.object(populated_registry, "_save_cache") as mock_save,
        ):
            await populated_registry._sync_with_ollama()

        mock_save.assert_not_called()

    async def test_stale_removal_triggers_save(self, populated_registry):
        with (
            patch.object(
                populated_registry,
                "_fetch_model_list",
                new_callable=AsyncMock,
                return_value=["llama3:latest"],
            ),
            patch.object(
                populated_registry,
                "_save_cache",
            ) as mock_save,
        ):
            await populated_registry._sync_with_ollama()

        mock_save.assert_called_once()

    async def test_http_error_handled_gracefully(self, populated_registry):
        with patch.object(
            populated_registry,
            "_fetch_model_list",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPError("connection failed"),
        ):
            await populated_registry._sync_with_ollama()

        assert len(populated_registry._sizes) == 2


# ---------------------------------------------------------------------------
# benchmark_model
# ---------------------------------------------------------------------------


def _make_mock_client(ps_data, generate_side_effect=None):
    mock_resp_generate = MagicMock()
    mock_resp_generate.raise_for_status = MagicMock()

    mock_resp_ps = MagicMock()
    mock_resp_ps.json.return_value = ps_data
    mock_resp_ps.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    if generate_side_effect:
        mock_client.post = AsyncMock(
            side_effect=generate_side_effect,
        )
    else:
        mock_client.post = AsyncMock(return_value=mock_resp_generate)
    mock_client.get = AsyncMock(return_value=mock_resp_ps)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestBenchmarkModel:
    async def test_successful_benchmark(self, registry):
        ps_data = {"models": [{"name": "llama3:latest", "size_vram": 4_661_224_676}]}
        mock_client = _make_mock_client(ps_data)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry.benchmark_model("llama3:latest")

        assert result == 4_661_224_676
        assert registry._sizes["llama3:latest"] == 4_661_224_676

    async def test_benchmark_saves_cache(self, registry):
        ps_data = {"models": [{"name": "llama3:latest", "size_vram": 4_000_000_000}]}
        mock_client = _make_mock_client(ps_data)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await registry.benchmark_model("llama3:latest")

        data = json.loads(registry.registry_path.read_text())
        assert data["llama3:latest"] == 4_000_000_000

    async def test_benchmark_model_not_in_ps(self, registry):
        ps_data = {"models": []}
        mock_client = _make_mock_client(ps_data)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry.benchmark_model("llama3:latest")

        assert result is None
        assert "llama3:latest" not in registry._sizes

    async def test_benchmark_http_error(self, registry):
        mock_client = _mock_async_client(AsyncMock())
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPError("connection error"),
        )

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry.benchmark_model("llama3:latest")

        assert result is None

    async def test_benchmark_already_in_progress(self, registry):
        registry._benchmarking.add("llama3:latest")
        result = await registry.benchmark_model("llama3:latest")
        assert result is None

    async def test_clears_flag_on_success(self, registry):
        ps_data = {"models": [{"name": "llama3:latest", "size_vram": 1234}]}
        mock_client = _make_mock_client(ps_data)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await registry.benchmark_model("llama3:latest")

        assert "llama3:latest" not in registry._benchmarking

    async def test_clears_flag_on_failure(self, registry):
        mock_client = _mock_async_client(AsyncMock())
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPError("fail"),
        )

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await registry.benchmark_model("llama3:latest")

        assert "llama3:latest" not in registry._benchmarking

    async def test_calls_generate_to_load_and_unload(self, registry):
        ps_data = {"models": [{"name": "llama3:latest", "size_vram": 5000}]}
        mock_client = _make_mock_client(ps_data)

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            await registry.benchmark_model("llama3:latest")

        assert mock_client.post.call_count == 2
        assert mock_client.get.call_count == 1

        load_call = mock_client.post.call_args_list[0]
        assert "api/generate" in load_call.args[0]
        assert load_call.kwargs["json"]["keep_alive"] == "5m"

        unload_call = mock_client.post.call_args_list[1]
        assert "api/generate" in unload_call.args[0]
        assert unload_call.kwargs["json"]["keep_alive"] == "0"


# ---------------------------------------------------------------------------
# benchmark_unknown
# ---------------------------------------------------------------------------


class TestBenchmarkUnknown:
    async def test_benchmarks_unknown_models(self, populated_registry):
        with (
            patch.object(
                populated_registry,
                "_fetch_model_list",
                new_callable=AsyncMock,
                return_value=[
                    "llama3:latest",
                    "codellama:latest",
                    "mistral:latest",
                ],
            ),
            patch.object(
                populated_registry,
                "benchmark_model",
                new_callable=AsyncMock,
                return_value=4_000_000_000,
            ) as mock_bench,
        ):
            await populated_registry.benchmark_unknown()

        mock_bench.assert_called_once_with("mistral:latest")

    async def test_all_benchmarked_does_nothing(self, populated_registry):
        with (
            patch.object(
                populated_registry,
                "_fetch_model_list",
                new_callable=AsyncMock,
                return_value=["llama3:latest", "codellama:latest"],
            ),
            patch.object(
                populated_registry,
                "benchmark_model",
                new_callable=AsyncMock,
            ) as mock_bench,
        ):
            await populated_registry.benchmark_unknown()

        mock_bench.assert_not_called()

    async def test_http_error_returns_early(self, populated_registry):
        with (
            patch.object(
                populated_registry,
                "_fetch_model_list",
                new_callable=AsyncMock,
                side_effect=httpx.HTTPError("fail"),
            ),
            patch.object(
                populated_registry,
                "benchmark_model",
                new_callable=AsyncMock,
            ) as mock_bench,
        ):
            await populated_registry.benchmark_unknown()

        mock_bench.assert_not_called()


# ---------------------------------------------------------------------------
# get_or_estimate_size
# ---------------------------------------------------------------------------


def _mock_show_client(show_response):
    mock_resp = MagicMock()
    mock_resp.json.return_value = show_response
    mock_resp.raise_for_status = MagicMock()
    mock_client = _mock_async_client(AsyncMock())
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


class TestGetOrEstimateSize:
    async def test_returns_cached_value(self, populated_registry):
        result = await populated_registry.get_or_estimate_size(
            "llama3:latest",
        )
        assert result == 4_661_224_676

    async def test_estimates_from_api_show(self, registry):
        mock_client = _mock_show_client(
            {
                "model_info": {
                    "general.parameter_count": 8_030_261_248,
                }
            }
        )

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry.get_or_estimate_size(
                "llama3:latest",
            )

        expected = 8_030_261_248 * 4
        assert result == expected

    async def test_falls_back_to_4gb_on_no_params(self, registry):
        mock_client = _mock_show_client({"model_info": {}})

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry.get_or_estimate_size(
                "unknown:latest",
            )

        assert result == 4 * 1024**3

    async def test_falls_back_to_4gb_on_http_error(self, registry):
        mock_client = _mock_async_client(AsyncMock())
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPError("fail"),
        )

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry.get_or_estimate_size(
                "unknown:latest",
            )

        assert result == 4 * 1024**3

    async def test_zero_param_count_falls_back(self, registry):
        mock_client = _mock_show_client(
            {
                "model_info": {"general.parameter_count": 0},
            }
        )

        with patch(PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await registry.get_or_estimate_size(
                "unknown:latest",
            )

        assert result == 4 * 1024**3


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    async def test_initialize_loads_and_syncs(self, registry):
        with (
            patch.object(
                registry,
                "_load_cache",
            ) as mock_load,
            patch.object(
                registry,
                "_sync_with_ollama",
                new_callable=AsyncMock,
            ) as mock_sync,
        ):
            await registry.initialize()

        mock_load.assert_called_once()
        mock_sync.assert_called_once()


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_host(self):
        reg = ModelRegistry()
        assert reg.ollama_host == "http://localhost:11434"

    def test_custom_host(self):
        reg = ModelRegistry(ollama_host="http://remote:11434")
        assert reg.ollama_host == "http://remote:11434"

    def test_default_registry_path(self):
        reg = ModelRegistry()
        expected = Path.home() / ".ollama-marshal" / "model_sizes.json"
        assert reg.registry_path == expected

    def test_custom_registry_path(self, tmp_path):
        path = tmp_path / "custom.json"
        reg = ModelRegistry(registry_path=path)
        assert reg.registry_path == path

    def test_initial_state_empty(self):
        reg = ModelRegistry()
        assert reg._sizes == {}
        assert reg._benchmarking == set()
