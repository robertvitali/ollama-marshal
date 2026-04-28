from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollama_marshal.registry import ModelMetadata, ModelRegistry

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
                "_load_metadata_cache",
            ) as mock_load_meta,
            patch.object(
                registry,
                "_sync_with_ollama",
                new_callable=AsyncMock,
            ) as mock_sync,
        ):
            await registry.initialize()

        mock_load.assert_called_once()
        mock_load_meta.assert_called_once()
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
        assert reg._metadata == {}
        assert reg._benchmarking == set()


# ---------------------------------------------------------------------------
# ModelMetadata dataclass + KV-cache math
# ---------------------------------------------------------------------------


def _qwen3_meta() -> ModelMetadata:
    """Realistic Qwen3-9B-style metadata for KV math tests."""
    return ModelMetadata(
        name="qwen3.5:9b-bf16",
        architecture="qwen3",
        max_context_length=32768,
        num_layers=28,
        embedding_length=3584,
        head_count=28,
        head_count_kv=4,  # GQA: 28 attention heads, 4 KV heads
        kv_dtype_bytes=2,
        probed_at="2026-04-28T03:00:00+00:00",
    )


class TestModelMetadataMath:
    def test_head_dim(self):
        m = _qwen3_meta()
        # head_dim = embedding_length / head_count = 3584 / 28 = 128
        assert m.head_dim == 128

    def test_kv_dim(self):
        m = _qwen3_meta()
        # kv_dim = head_dim * head_count_kv = 128 * 4 = 512
        assert m.kv_dim == 512

    def test_kv_per_slot_at_max_ctx(self):
        m = _qwen3_meta()
        # max_ctx * num_layers * kv_dim * dtype_bytes * 2
        # 32768 * 28 * 512 * 2 * 2
        expected = 32768 * 28 * 512 * 2 * 2
        assert m.kv_per_slot_at_max_ctx == expected

    def test_kv_per_slot_at_smaller_ctx(self):
        m = _qwen3_meta()
        # Halving context halves the KV cost.
        small = m.kv_per_slot_at_ctx(4096)
        full = m.kv_per_slot_at_max_ctx
        assert small * 8 == full  # 32768 / 4096 = 8

    def test_head_dim_zero_safe(self):
        # Defensive: avoid div-by-zero if head_count is unknown.
        m = ModelMetadata(
            name="x",
            architecture="x",
            max_context_length=1,
            num_layers=1,
            embedding_length=1,
            head_count=0,
            head_count_kv=0,
        )
        assert m.head_dim == 0
        assert m.kv_dim == 0
        assert m.kv_per_slot_at_max_ctx == 0


class TestModelMetadataRoundtrip:
    def test_to_and_from_json_dict(self):
        original = _qwen3_meta()
        d = original.to_json_dict()
        # Computed field must be present in serialized form.
        assert d["kv_per_slot_at_max_ctx"] == original.kv_per_slot_at_max_ctx
        roundtripped = ModelMetadata.from_json_dict(d)
        assert roundtripped == original

    def test_from_json_dict_with_extra_fields(self):
        # Extra (computed) fields like kv_per_slot_at_max_ctx should be
        # ignored on read since they're derived.
        d = _qwen3_meta().to_json_dict()
        d["unrelated"] = 42
        m = ModelMetadata.from_json_dict(d)
        assert m.architecture == "qwen3"

    def test_from_json_dict_missing_optional_fields(self):
        # kv_dtype_bytes and probed_at have defaults.
        d = {
            "name": "x",
            "architecture": "llama",
            "max_context_length": 8192,
            "num_layers": 32,
            "embedding_length": 4096,
            "head_count": 32,
            "head_count_kv": 8,
        }
        m = ModelMetadata.from_json_dict(d)
        assert m.kv_dtype_bytes == 2
        assert m.probed_at == ""


# ---------------------------------------------------------------------------
# _parse_show_response
# ---------------------------------------------------------------------------


def _show_response(arch: str = "qwen3", **overrides) -> dict:
    """Build a realistic /api/show JSON body for tests."""
    info = {
        "general.architecture": arch,
        f"{arch}.context_length": 32768,
        f"{arch}.block_count": 28,
        f"{arch}.embedding_length": 3584,
        f"{arch}.attention.head_count": 28,
        f"{arch}.attention.head_count_kv": 4,
    }
    info.update(overrides)
    return {"model_info": info}


class TestParseShowResponse:
    def test_full_response(self):
        result = ModelRegistry._parse_show_response("qwen3.5:9b-bf16", _show_response())
        assert result.used_fallback is False
        assert result.metadata.architecture == "qwen3"
        assert result.metadata.max_context_length == 32768
        assert result.metadata.num_layers == 28
        assert result.metadata.head_count == 28
        assert result.metadata.head_count_kv == 4

    def test_missing_context_length_uses_fallback(self):
        body = _show_response()
        del body["model_info"]["qwen3.context_length"]
        result = ModelRegistry._parse_show_response("x", body)
        assert result.used_fallback is True
        assert "context_length" in result.missing_fields
        assert result.metadata.max_context_length == 4096  # fallback

    def test_missing_head_count_kv_defaults_to_head_count(self):
        # No GQA — many older models. head_count_kv inherits head_count.
        body = _show_response()
        del body["model_info"]["qwen3.attention.head_count_kv"]
        result = ModelRegistry._parse_show_response("x", body)
        # Inheriting from head_count is NOT a fallback (it's the documented
        # behavior for non-GQA models).
        assert result.metadata.head_count_kv == 28
        assert "attention.head_count_kv" not in result.missing_fields

    def test_unknown_architecture(self):
        # No general.architecture → arch="unknown" → no per-arch keys exist
        # → all fallbacks.
        result = ModelRegistry._parse_show_response("x", {"model_info": {}})
        assert result.used_fallback is True
        assert result.metadata.architecture == "unknown"

    def test_invalid_int_uses_fallback(self):
        body = _show_response()
        body["model_info"]["qwen3.context_length"] = "not-a-number"
        result = ModelRegistry._parse_show_response("x", body)
        assert result.used_fallback is True
        assert result.metadata.max_context_length == 4096


# ---------------------------------------------------------------------------
# probe_metadata + cache I/O
# ---------------------------------------------------------------------------


class TestProbeMetadata:
    async def test_caches_and_persists(self, tmp_path):
        meta_path = tmp_path / "metadata.json"
        reg = ModelRegistry(
            registry_path=tmp_path / "sizes.json",
            metadata_path=meta_path,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=MagicMock(
                json=MagicMock(return_value=_show_response()),
                raise_for_status=MagicMock(),
            )
        )
        with patch(PATCH_ASYNC_CLIENT) as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_client
            result = await reg.probe_metadata("qwen3.5:9b-bf16")

        assert result is not None
        assert result.architecture == "qwen3"
        # Second probe hits the in-memory cache, no HTTP call.
        with patch(PATCH_ASYNC_CLIENT) as mock_cls2:
            again = await reg.probe_metadata("qwen3.5:9b-bf16")
            mock_cls2.assert_not_called()
        assert again is result
        # And the on-disk cache exists with the new entry.
        assert meta_path.exists()
        on_disk = json.loads(meta_path.read_text())
        assert "qwen3.5:9b-bf16" in on_disk

    async def test_returns_none_on_http_error(self, registry):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("down"))
        with patch(PATCH_ASYNC_CLIENT) as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_client
            result = await registry.probe_metadata("model:x")
        assert result is None
        # Cache stays empty so next probe will retry.
        assert registry.get_metadata("model:x") is None


class TestMetadataCacheIO:
    def test_load_existing(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps({"qwen3.5:9b-bf16": _qwen3_meta().to_json_dict()}))
        reg = ModelRegistry(metadata_path=path)
        reg._load_metadata_cache()
        assert "qwen3.5:9b-bf16" in reg._metadata
        assert reg._metadata["qwen3.5:9b-bf16"].max_context_length == 32768

    def test_load_corrupt_recovers(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text("{not valid json")
        reg = ModelRegistry(metadata_path=path)
        reg._load_metadata_cache()
        assert reg._metadata == {}

    def test_load_skips_bad_entries(self, tmp_path):
        # Mix one good entry and one missing-key entry — only the good
        # one should be loaded.
        path = tmp_path / "metadata.json"
        path.write_text(
            json.dumps(
                {
                    "good": _qwen3_meta().to_json_dict(),
                    "bad": {"name": "bad"},  # missing required keys
                }
            )
        )
        reg = ModelRegistry(metadata_path=path)
        reg._load_metadata_cache()
        assert "good" in reg._metadata
        assert "bad" not in reg._metadata


# ---------------------------------------------------------------------------
# Convenience getters
# ---------------------------------------------------------------------------


class TestMetadataGetters:
    def test_get_metadata_missing_returns_none(self, registry):
        assert registry.get_metadata("not-cached") is None

    def test_get_metadata_returns_cached(self, registry):
        registry._metadata["qwen3.5:9b-bf16"] = _qwen3_meta()
        assert registry.get_metadata("qwen3.5:9b-bf16").architecture == "qwen3"

    def test_get_max_context(self, registry):
        registry._metadata["qwen3.5:9b-bf16"] = _qwen3_meta()
        assert registry.get_max_context("qwen3.5:9b-bf16") == 32768

    def test_get_max_context_missing(self, registry):
        assert registry.get_max_context("x") is None

    def test_get_kv_per_slot_default_max(self, registry):
        registry._metadata["qwen3.5:9b-bf16"] = _qwen3_meta()
        # Default ctx = max
        assert (
            registry.get_kv_per_slot("qwen3.5:9b-bf16")
            == _qwen3_meta().kv_per_slot_at_max_ctx
        )

    def test_get_kv_per_slot_explicit_ctx(self, registry):
        registry._metadata["qwen3.5:9b-bf16"] = _qwen3_meta()
        small = registry.get_kv_per_slot("qwen3.5:9b-bf16", ctx=4096)
        full = registry.get_kv_per_slot("qwen3.5:9b-bf16")
        assert small * 8 == full

    def test_get_kv_per_slot_missing(self, registry):
        assert registry.get_kv_per_slot("x") is None


# ---------------------------------------------------------------------------
# remove_model: clears both caches
# ---------------------------------------------------------------------------


class TestRemoveModelClearsBothCaches:
    def test_removes_size_and_metadata(self, tmp_path):
        reg = ModelRegistry(
            registry_path=tmp_path / "sizes.json",
            metadata_path=tmp_path / "metadata.json",
        )
        reg._sizes["qwen3.5:9b-bf16"] = 6_000_000_000
        reg._metadata["qwen3.5:9b-bf16"] = _qwen3_meta()

        reg.remove_model("qwen3.5:9b-bf16")

        assert "qwen3.5:9b-bf16" not in reg._sizes
        assert "qwen3.5:9b-bf16" not in reg._metadata
        # Both files written.
        assert "qwen3.5:9b-bf16" not in json.loads(
            (tmp_path / "sizes.json").read_text()
        )
        assert "qwen3.5:9b-bf16" not in json.loads(
            (tmp_path / "metadata.json").read_text()
        )


# ---------------------------------------------------------------------------
# is_known_model + opportunistic resync
# ---------------------------------------------------------------------------


class TestIsKnownModel:
    async def test_returns_true_for_cached_model(self, registry):
        registry._known_models = {"llama3:latest", "mistral:latest"}
        registry._known_models_last_sync = 9_999_999.0  # block resync
        assert await registry.is_known_model("llama3:latest") is True

    async def test_returns_false_for_unknown_within_resync_window(self, registry):
        # Simulate a recent sync — should not re-fetch.
        import time as _time

        registry._known_models = {"llama3:latest"}
        registry._known_models_last_sync = _time.monotonic()
        # No mock — if it tried to fetch, it'd raise (no Ollama in tests).
        assert await registry.is_known_model("nope:latest") is False

    async def test_resyncs_after_window_expires(self, registry):
        # Stale last_sync forces a re-fetch.
        registry._known_models = set()
        registry._known_models_last_sync = 0.0  # very old

        with patch.object(
            registry,
            "_fetch_model_list",
            new_callable=AsyncMock,
            return_value=["just-pulled:latest"],
        ):
            result = await registry.is_known_model("just-pulled:latest")

        assert result is True
        assert "just-pulled:latest" in registry._known_models

    async def test_resync_failure_fails_open(self, registry):
        # If Ollama is unreachable mid-flight, prefer letting the request
        # through over wrongly 404-ing.
        registry._known_models = set()
        registry._known_models_last_sync = 0.0

        with patch.object(
            registry,
            "_fetch_model_list",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPError("ollama down"),
        ):
            assert await registry.is_known_model("anything:latest") is True

    async def test_returns_false_after_resync_still_missing(self, registry):
        # User asked for a model that wasn't pulled — re-sync confirms it.
        registry._known_models = {"old:latest"}
        registry._known_models_last_sync = 0.0

        with patch.object(
            registry,
            "_fetch_model_list",
            new_callable=AsyncMock,
            return_value=["old:latest"],
        ):
            assert await registry.is_known_model("never-pulled:latest") is False

    async def test_sync_with_ollama_populates_known_models(self, populated_registry):
        with patch.object(
            populated_registry,
            "_fetch_model_list",
            new_callable=AsyncMock,
            return_value=["llama3:latest", "mistral:latest"],
        ):
            await populated_registry._sync_with_ollama()

        assert populated_registry._known_models == {
            "llama3:latest",
            "mistral:latest",
        }
        assert populated_registry._known_models_last_sync > 0.0
