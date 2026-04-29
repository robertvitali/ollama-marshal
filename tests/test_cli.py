from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import typer
from typer.testing import CliRunner

from ollama_marshal import __version__
from ollama_marshal.cli import _setup_logging, _version_callback, app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestVersionCallback:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert __version__ in result.output
        assert result.exit_code == 0

    def test_short_version_flag(self):
        result = runner.invoke(app, ["-v"])
        assert __version__ in result.output

    def test_version_callback_raises_exit(self):
        with pytest.raises(typer.Exit):
            _version_callback(True)

    def test_version_callback_false_noop(self):
        # Should not raise
        _version_callback(False)


# ---------------------------------------------------------------------------
# Main callback (no-args-is-help)
# ---------------------------------------------------------------------------


class TestMainCallback:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer's no_args_is_help exits with code 0 or 2 depending on version
        assert result.exit_code in (0, 2)
        assert "ollama-marshal" in result.output


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    @patch("ollama_marshal.cli.structlog")
    def test_console_format(self, mock_structlog):
        _setup_logging("INFO", "console")
        mock_structlog.configure.assert_called_once()

    @patch("ollama_marshal.cli.structlog")
    def test_json_format(self, mock_structlog):
        _setup_logging("DEBUG", "json")
        mock_structlog.configure.assert_called_once()

    # Unmocked tests — exercise the real structlog and stdlib logging codepath
    # so a missing/wrong attribute call surfaces. The mocked tests above do
    # not catch this because every structlog attribute resolves to MagicMock.
    @pytest.mark.parametrize(
        "level", ["DEBUG", "INFO", "WARNING", "ERROR", "info", "warning"]
    )
    def test_real_setup_accepts_standard_levels(self, level):
        # Should not raise. Idempotent — re-running just reconfigures structlog.
        _setup_logging(level, "console")
        _setup_logging(level, "json")

    def test_real_setup_unknown_level_falls_back_to_info(self):
        # Defensive: unknown level shouldn't crash the CLI.
        _setup_logging("NOPE", "console")


# ---------------------------------------------------------------------------
# start command
# ---------------------------------------------------------------------------


class TestStartCommand:
    @patch("ollama_marshal.cli.uvicorn")
    @patch("ollama_marshal.cli.load_config")
    def test_start_default(self, mock_load_config, mock_uvicorn):
        from ollama_marshal.config import MarshalConfig

        mock_load_config.return_value = MarshalConfig()

        with patch("ollama_marshal.cli.structlog"):
            runner.invoke(app, ["start"])

        mock_load_config.assert_called_once()
        mock_uvicorn.run.assert_called_once()

    @patch("ollama_marshal.cli.uvicorn")
    @patch("ollama_marshal.cli.load_config")
    def test_start_with_host_and_port(self, mock_load_config, mock_uvicorn):
        from ollama_marshal.config import MarshalConfig

        mock_load_config.return_value = MarshalConfig()

        with patch("ollama_marshal.cli.structlog"):
            runner.invoke(
                app,
                ["start", "--host", "127.0.0.1", "--port", "9999"],
            )

        call_kwargs = mock_load_config.call_args
        cli_overrides = call_kwargs.kwargs.get("cli_overrides") or call_kwargs[1].get(
            "cli_overrides"
        )
        assert cli_overrides["proxy.host"] == "127.0.0.1"
        assert cli_overrides["proxy.port"] == 9999

    @patch("ollama_marshal.cli.uvicorn")
    @patch("ollama_marshal.cli.load_config")
    def test_start_with_config_file(self, mock_load_config, mock_uvicorn):
        from ollama_marshal.config import MarshalConfig

        mock_load_config.return_value = MarshalConfig()

        with patch("ollama_marshal.cli.structlog"):
            runner.invoke(
                app,
                ["start", "--config", "/tmp/test.yaml"],  # noqa: S108
            )

        call_kwargs = mock_load_config.call_args
        config_path = call_kwargs.kwargs.get("config_path") or call_kwargs[1].get(
            "config_path"
        )
        assert config_path == "/tmp/test.yaml"  # noqa: S108

    @patch("ollama_marshal.cli.uvicorn")
    @patch("ollama_marshal.cli.load_config")
    def test_start_with_ollama_host(self, mock_load_config, mock_uvicorn):
        from ollama_marshal.config import MarshalConfig

        mock_load_config.return_value = MarshalConfig()

        with patch("ollama_marshal.cli.structlog"):
            runner.invoke(
                app,
                ["start", "--ollama-host", "http://remote:11434"],
            )

        call_kwargs = mock_load_config.call_args
        cli_overrides = call_kwargs.kwargs.get("cli_overrides") or call_kwargs[1].get(
            "cli_overrides"
        )
        assert cli_overrides["ollama.host"] == "http://remote:11434"

    @patch("ollama_marshal.cli.uvicorn")
    @patch("ollama_marshal.cli.load_config")
    def test_start_with_log_level(self, mock_load_config, mock_uvicorn):
        from ollama_marshal.config import MarshalConfig

        mock_load_config.return_value = MarshalConfig()

        with patch("ollama_marshal.cli.structlog"):
            runner.invoke(
                app,
                ["start", "--log-level", "DEBUG"],
            )

        call_kwargs = mock_load_config.call_args
        cli_overrides = call_kwargs.kwargs.get("cli_overrides") or call_kwargs[1].get(
            "cli_overrides"
        )
        assert cli_overrides["logging.level"] == "DEBUG"

    @patch("ollama_marshal.cli.uvicorn")
    @patch("ollama_marshal.cli.load_config")
    def test_start_with_log_format(self, mock_load_config, mock_uvicorn):
        from ollama_marshal.config import MarshalConfig

        mock_load_config.return_value = MarshalConfig()

        with patch("ollama_marshal.cli.structlog"):
            runner.invoke(
                app,
                ["start", "--log-format", "json"],
            )

        call_kwargs = mock_load_config.call_args
        cli_overrides = call_kwargs.kwargs.get("cli_overrides") or call_kwargs[1].get(
            "cli_overrides"
        )
        assert cli_overrides["logging.format"] == "json"

    @patch("ollama_marshal.cli.uvicorn")
    @patch("ollama_marshal.cli.load_config")
    def test_start_passes_config_to_uvicorn(self, mock_load_config, mock_uvicorn):
        from ollama_marshal.config import MarshalConfig, ProxyConfig

        bind_addr = "0.0.0.0"  # noqa: S104
        cfg = MarshalConfig(proxy=ProxyConfig(host=bind_addr, port=12345))
        mock_load_config.return_value = cfg

        with patch("ollama_marshal.cli.structlog"):
            runner.invoke(app, ["start"])

        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs.kwargs["host"] == bind_addr
        assert call_kwargs.kwargs["port"] == 12345


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


class TestStatusCommand:
    @patch("ollama_marshal.cli.httpx")
    def test_status_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "uptime_seconds": 3661.0,
            "loaded_models": [
                {
                    "name": "llama3:latest",
                    "size_vram": 4 * 1024**3,
                    "pending_requests": 0,
                }
            ],
            "memory": {
                "total": 64 * 1024**3,
                "used_by_models": 4 * 1024**3,
                "available": 58 * 1024**3,
            },
            "queue": {
                "total_pending": 0,
                "by_model": {},
            },
            "metrics": {
                "requests_served": 42,
                "model_swaps": 5,
                "evictions": 2,
                "average_wait_ms": 15.3,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_resp

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "ollama-marshal status" in result.output
        assert "llama3:latest" in result.output
        assert "42" in result.output

    @patch("ollama_marshal.cli.httpx")
    def test_status_with_custom_host(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "uptime_seconds": 10.0,
            "loaded_models": [],
            "memory": {"total": 0, "used_by_models": 0, "available": 0},
            "queue": {"total_pending": 0, "by_model": {}},
            "metrics": {
                "requests_served": 0,
                "model_swaps": 0,
                "evictions": 0,
                "average_wait_ms": 0.0,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_resp

        result = runner.invoke(app, ["status", "--host", "http://custom:9999"])
        assert result.exit_code == 0
        mock_httpx.get.assert_called_once_with(
            "http://custom:9999/api/marshal/status", timeout=5
        )

    @patch("ollama_marshal.cli.httpx")
    def test_status_connection_error(self, mock_httpx):
        mock_httpx.get.side_effect = httpx.HTTPError("connection refused")
        mock_httpx.HTTPError = httpx.HTTPError

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert "Cannot reach" in result.output

    @patch("ollama_marshal.cli.httpx")
    def test_status_no_models_shows_none(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "uptime_seconds": 5.0,
            "loaded_models": [],
            "memory": {"total": 0, "used_by_models": 0, "available": 0},
            "queue": {"total_pending": 0, "by_model": {}},
            "metrics": {
                "requests_served": 0,
                "model_swaps": 0,
                "evictions": 0,
                "average_wait_ms": 0.0,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_resp

        result = runner.invoke(app, ["status"])
        assert "(none)" in result.output

    @patch("ollama_marshal.cli.httpx")
    def test_status_with_queue_by_model(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "uptime_seconds": 100.0,
            "loaded_models": [],
            "memory": {"total": 0, "used_by_models": 0, "available": 0},
            "queue": {
                "total_pending": 3,
                "by_model": {"llama3:latest": 2, "mistral:latest": 1},
            },
            "metrics": {
                "requests_served": 0,
                "model_swaps": 0,
                "evictions": 0,
                "average_wait_ms": 0.0,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_resp

        result = runner.invoke(app, ["status"])
        assert "3 pending" in result.output
        assert "llama3:latest" in result.output


# ---------------------------------------------------------------------------
# stop command
# ---------------------------------------------------------------------------


class TestStopCommand:
    def test_stop_exits_cleanly(self):
        result = runner.invoke(app, ["stop"])
        assert result.exit_code == 0
        assert "Targeting proxy at" in result.output

    def test_stop_with_custom_host(self):
        result = runner.invoke(app, ["stop", "--host", "http://custom:9999"])
        assert result.exit_code == 0
        assert "http://custom:9999" in result.output


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------


class TestDoctorCommand:
    @patch("ollama_marshal.doctor.gather_report")
    def test_doctor_runs_and_renders(self, mock_gather):
        from ollama_marshal.doctor import DoctorReport

        async def _fake(*args, **kwargs):
            return DoctorReport(
                total_ram_bytes=64 * 1024**3,
                loaded_models=[],
                all_models=[],
                recommended_env={"OLLAMA_KV_CACHE_TYPE": "q8_0"},
            )

        mock_gather.side_effect = _fake
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "OLLAMA_KV_CACHE_TYPE=q8_0" in result.output
        assert "ollama-marshal doctor" in result.output

    @patch("ollama_marshal.doctor.gather_report")
    def test_doctor_handles_http_error(self, mock_gather):
        async def _fake(*args, **kwargs):
            raise httpx.HTTPError("ollama down")

        mock_gather.side_effect = _fake
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "doctor probe failed" in result.output
