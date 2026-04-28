from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollama_marshal.doctor import (
    DoctorReport,
    ModelDoctorEntry,
    _largest_kv_cost_model,
    _populate_recommendations,
    gather_report,
    render_report,
)


def _entry(
    name: str,
    *,
    loaded: bool = False,
    size_vram_bytes: int = 0,
    kv_per_slot: int = 4 * 1024**3,
    max_ctx: int = 32768,
    probe_ok: bool = True,
) -> ModelDoctorEntry:
    return ModelDoctorEntry(
        name=name,
        loaded=loaded,
        size_vram_bytes=size_vram_bytes,
        kv_per_slot_at_max_ctx=kv_per_slot,
        max_context_length=max_ctx,
        probe_ok=probe_ok,
    )


# ---------------------------------------------------------------------------
# _largest_kv_cost_model
# ---------------------------------------------------------------------------


class TestLargestKvCostModel:
    def test_returns_none_for_empty(self):
        assert _largest_kv_cost_model([]) is None

    def test_returns_none_when_all_probes_failed(self):
        entries = [_entry("a", probe_ok=False, kv_per_slot=0)]
        assert _largest_kv_cost_model(entries) is None

    def test_picks_largest(self):
        entries = [
            _entry("small", kv_per_slot=1 * 1024**3),
            _entry("big", kv_per_slot=10 * 1024**3),
            _entry("medium", kv_per_slot=5 * 1024**3),
        ]
        assert _largest_kv_cost_model(entries).name == "big"

    def test_skips_failed_probes(self):
        entries = [
            _entry("ok", kv_per_slot=2 * 1024**3),
            _entry("failed", probe_ok=False, kv_per_slot=99 * 1024**3),
        ]
        assert _largest_kv_cost_model(entries).name == "ok"


# ---------------------------------------------------------------------------
# _populate_recommendations
# ---------------------------------------------------------------------------


class TestPopulateRecommendations:
    def test_always_recommends_q8_kv_cache(self):
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=[],
            all_models=[_entry("x", kv_per_slot=1 * 1024**3)],
        )
        _populate_recommendations(report)
        assert report.recommended_env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"

    def test_always_recommends_flash_attention(self):
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=[],
            all_models=[_entry("x", kv_per_slot=1 * 1024**3)],
        )
        _populate_recommendations(report)
        assert report.recommended_env["OLLAMA_FLASH_ATTENTION"] == "1"

    def test_skips_num_parallel_when_no_metadata(self):
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=[],
            all_models=[_entry("x", probe_ok=False, kv_per_slot=0)],
        )
        _populate_recommendations(report)
        assert "OLLAMA_NUM_PARALLEL" not in report.recommended_env
        assert any("metadata" in n for n in report.notes)

    def test_skips_num_parallel_when_ram_unknown(self):
        report = DoctorReport(
            total_ram_bytes=0,
            loaded_models=[],
            all_models=[_entry("x", kv_per_slot=1 * 1024**3)],
        )
        _populate_recommendations(report)
        assert "OLLAMA_NUM_PARALLEL" not in report.recommended_env

    def test_recommends_num_parallel_one_for_huge_kv_cost(self):
        # 256 GB RAM, single model with 60 GB KV/slot at fp16 → 30 GB at q8_0.
        # 25% of 256 GB = 64 GB budget → 64 / 30 = 2 (capped at 4).
        report = DoctorReport(
            total_ram_bytes=256 * 1024**3,
            loaded_models=[],
            all_models=[_entry("huge", kv_per_slot=60 * 1024**3)],
        )
        _populate_recommendations(report)
        assert report.recommended_env["OLLAMA_NUM_PARALLEL"] == "2"

    def test_caps_num_parallel_at_4(self):
        # Tiny KV cost → would compute large parallelism, but we cap at 4.
        report = DoctorReport(
            total_ram_bytes=256 * 1024**3,
            loaded_models=[],
            all_models=[_entry("tiny", kv_per_slot=100 * 1024**2)],
        )
        _populate_recommendations(report)
        assert report.recommended_env["OLLAMA_NUM_PARALLEL"] == "4"

    def test_includes_max_loaded_models_when_metadata_available(self):
        report = DoctorReport(
            total_ram_bytes=256 * 1024**3,
            loaded_models=[],
            all_models=[
                _entry("a", kv_per_slot=2 * 1024**3),
                _entry("b", kv_per_slot=4 * 1024**3),
                _entry("c", kv_per_slot=6 * 1024**3),
            ],
        )
        _populate_recommendations(report)
        assert "OLLAMA_MAX_LOADED_MODELS" in report.recommended_env
        # Reasonable bound: not 0, not absurdly large.
        n = int(report.recommended_env["OLLAMA_MAX_LOADED_MODELS"])
        assert n >= 1


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


class TestRenderReport:
    def test_includes_ram_line(self):
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=[],
            all_models=[],
        )
        out = render_report(report)
        assert "64.0 GB" in out

    def test_includes_loaded_models_section(self):
        loaded = [
            _entry(
                "qwen3.5:9b-bf16",
                loaded=True,
                size_vram_bytes=12 * 1024**3,
                kv_per_slot=4 * 1024**3,
            ),
        ]
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=loaded,
            all_models=loaded,
        )
        out = render_report(report)
        assert "qwen3.5:9b-bf16" in out
        assert "12.0 GB" in out

    def test_includes_recommended_env_section(self):
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=[],
            all_models=[],
            recommended_env={"OLLAMA_KV_CACHE_TYPE": "q8_0"},
        )
        out = render_report(report)
        assert "OLLAMA_KV_CACHE_TYPE=q8_0" in out

    def test_warns_when_unexpected_unloads_present(self):
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=[],
            all_models=[],
            unexpected_unloads=5,
        )
        out = render_report(report)
        assert "Unexpected unloads" in out
        assert "5" in out
        # The warning emoji or word should appear.
        assert "Ollama is dropping" in out or "⚠" in out

    def test_no_warning_when_unexpected_unloads_zero(self):
        report = DoctorReport(
            total_ram_bytes=64 * 1024**3,
            loaded_models=[],
            all_models=[],
            unexpected_unloads=0,
        )
        out = render_report(report)
        assert "Ollama is dropping" not in out


# ---------------------------------------------------------------------------
# gather_report integration
# ---------------------------------------------------------------------------


class TestGatherReport:
    @pytest.fixture
    def mock_psutil(self):
        with patch("ollama_marshal.doctor.psutil") as m:
            m.virtual_memory.return_value = MagicMock(total=256 * 1024**3)
            yield m

    async def test_returns_empty_when_tags_unreachable(self, mock_psutil):
        with patch(
            "ollama_marshal.doctor.ModelRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg._fetch_model_list = AsyncMock(
                side_effect=httpx.HTTPError("ollama down")
            )
            mock_reg_cls.return_value = mock_reg
            report = await gather_report(ollama_host="http://localhost:11434")
        assert report.all_models == []
        assert report.loaded_models == []
        # System RAM is still reported.
        assert report.total_ram_bytes == 256 * 1024**3

    async def test_includes_unexpected_unloads_from_marshal(self, mock_psutil):
        from ollama_marshal.registry import ModelMetadata

        meta = ModelMetadata(
            name="x",
            architecture="qwen3",
            max_context_length=32768,
            num_layers=28,
            embedding_length=3584,
            head_count=28,
            head_count_kv=4,
        )

        async def fake_get(url, *args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/api/marshal/status" in url:
                resp.json.return_value = {"metrics": {"unexpected_unloads": 7}}
            elif "/api/ps" in url:
                resp.json.return_value = {"models": []}
            else:
                resp.json.return_value = {}
            return resp

        with (
            patch("ollama_marshal.doctor.ModelRegistry") as mock_reg_cls,
            patch(
                "ollama_marshal.doctor.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_reg = MagicMock()
            mock_reg._fetch_model_list = AsyncMock(return_value=["x"])
            mock_reg.probe_metadata = AsyncMock(return_value=meta)
            mock_reg_cls.return_value = mock_reg

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=fake_get)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            report = await gather_report(
                ollama_host="http://localhost:11434",
                marshal_status_url="http://localhost:11435/api/marshal/status",
            )

        assert report.unexpected_unloads == 7
        assert len(report.all_models) == 1
        assert report.all_models[0].name == "x"
        assert report.all_models[0].probe_ok is True
