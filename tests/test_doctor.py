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
    """Integration tests for gather_report.

    Mocks only at the httpx HTTP boundary — never patches internal
    `ollama_marshal.*` classes (Bright-line #1). The real ModelRegistry
    is exercised, hitting the patched httpx.AsyncClient for /api/tags
    and /api/show.
    """

    @pytest.fixture
    def mock_psutil(self):
        with patch("ollama_marshal.doctor.psutil") as m:
            m.virtual_memory.return_value = MagicMock(total=256 * 1024**3)
            yield m

    @staticmethod
    def _install_httpx(handlers: dict[str, object]):
        """Patch both registry+doctor httpx clients with one shared handler.

        `handlers` maps a URL substring to a callable
        `(method, url, **kwargs) -> response` (or a callable that raises
        an exception). Each AsyncClient method (.get, .post) routes to
        the first matching handler.

        Returns a context manager that installs both patches.
        """

        def _build_response(payload):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = payload
            return resp

        def _route(method: str, url: str, **kwargs):
            for substr, handler in handlers.items():
                if substr in url:
                    result = handler(method, url, **kwargs)
                    if isinstance(result, Exception):
                        raise result
                    return _build_response(result)
            return _build_response({})

        def _make_client():
            client = AsyncMock()
            # AsyncMock with side_effect=sync-fn awaits the return value.
            # Returning the response directly (not a coroutine) keeps it
            # simple and matches httpx.AsyncClient.get's return type.
            client.get = AsyncMock(
                side_effect=lambda url, **kw: _route("GET", url, **kw)
            )
            client.post = AsyncMock(
                side_effect=lambda url, **kw: _route("POST", url, **kw)
            )
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            return client

        # Each `with httpx.AsyncClient() as client:` block needs its own
        # client (calling __aenter__ a second time on a used mock would
        # confuse some tests). Use a factory side_effect.
        registry_patch = patch(
            "ollama_marshal.registry.httpx.AsyncClient",
            side_effect=lambda *a, **kw: _make_client(),
        )
        doctor_patch = patch(
            "ollama_marshal.doctor.httpx.AsyncClient",
            side_effect=lambda *a, **kw: _make_client(),
        )

        class _Combined:
            def __enter__(self):
                registry_patch.start()
                doctor_patch.start()

            def __exit__(self, *exc):
                doctor_patch.stop()
                registry_patch.stop()
                return False

        return _Combined()

    @staticmethod
    def _qwen3_show_payload():
        """A realistic /api/show body for an arch-known qwen3 model."""
        return {
            "model_info": {
                "general.architecture": "qwen3",
                "qwen3.context_length": 32768,
                "qwen3.block_count": 28,
                "qwen3.embedding_length": 3584,
                "qwen3.attention.head_count": 28,
                "qwen3.attention.head_count_kv": 4,
            }
        }

    async def test_returns_empty_when_tags_unreachable(self, mock_psutil):
        def _tags_fail(method, url, **kw):
            return httpx.HTTPError("ollama down")

        with self._install_httpx({"/api/tags": _tags_fail}):
            report = await gather_report(ollama_host="http://localhost:11434")
        assert report.all_models == []
        assert report.loaded_models == []
        assert report.total_ram_bytes == 256 * 1024**3

    async def test_includes_unexpected_unloads_from_marshal(self, mock_psutil):
        handlers = {
            "/api/tags": lambda m, u, **kw: {"models": [{"name": "x"}]},
            "/api/show": lambda m, u, **kw: self._qwen3_show_payload(),
            "/api/ps": lambda m, u, **kw: {"models": []},
            "/api/marshal/status": lambda m, u, **kw: {
                "metrics": {"unexpected_unloads": 7}
            },
        }
        with self._install_httpx(handlers):
            report = await gather_report(
                ollama_host="http://localhost:11434",
                marshal_status_url="http://localhost:11435/api/marshal/status",
            )

        assert report.unexpected_unloads == 7
        assert len(report.all_models) == 1
        assert report.all_models[0].name == "x"
        assert report.all_models[0].probe_ok is True

    async def test_failed_probe_recorded_with_probe_ok_false(self, mock_psutil):
        # When /api/show fails, ModelRegistry.probe_metadata returns
        # None and the doctor records probe_ok=False with KV cost 0.
        def _show_fail(method, url, **kw):
            return httpx.HTTPError("show down")

        handlers = {
            "/api/tags": lambda m, u, **kw: {"models": [{"name": "unprobeable:x"}]},
            "/api/show": _show_fail,
            "/api/ps": lambda m, u, **kw: {"models": []},
        }
        with self._install_httpx(handlers):
            report = await gather_report(ollama_host="http://localhost:11434")

        assert len(report.all_models) == 1
        assert report.all_models[0].probe_ok is False
        assert report.all_models[0].kv_per_slot_at_max_ctx == 0

    async def test_loaded_models_have_size_vram_populated(self, mock_psutil):
        handlers = {
            "/api/tags": lambda m, u, **kw: {"models": [{"name": "x"}]},
            "/api/show": lambda m, u, **kw: self._qwen3_show_payload(),
            "/api/ps": lambda m, u, **kw: {
                "models": [{"name": "x", "size_vram": 5_000_000_000}]
            },
        }
        with self._install_httpx(handlers):
            report = await gather_report(ollama_host="http://localhost:11434")

        assert len(report.loaded_models) == 1
        assert report.loaded_models[0].size_vram_bytes == 5_000_000_000

    async def test_unexpected_unloads_marshal_unreachable_returns_none(
        self, mock_psutil
    ):
        def _marshal_down(method, url, **kw):
            return httpx.HTTPError("marshal down")

        handlers = {
            "/api/tags": lambda m, u, **kw: {"models": []},
            "/api/ps": lambda m, u, **kw: {"models": []},
            "/api/marshal/status": _marshal_down,
        }
        with self._install_httpx(handlers):
            report = await gather_report(
                ollama_host="http://localhost:11434",
                marshal_status_url="http://localhost:11435/api/marshal/status",
            )

        assert report.unexpected_unloads is None

    async def test_unexpected_unloads_handles_non_int_payload(self, mock_psutil):
        handlers = {
            "/api/tags": lambda m, u, **kw: {"models": []},
            "/api/ps": lambda m, u, **kw: {"models": []},
            "/api/marshal/status": lambda m, u, **kw: {
                "metrics": {"unexpected_unloads": "not-an-int"}
            },
        }
        with self._install_httpx(handlers):
            report = await gather_report(
                ollama_host="http://localhost:11434",
                marshal_status_url="http://localhost:11435/api/marshal/status",
            )

        assert report.unexpected_unloads is None
