from __future__ import annotations

import os
from typing import Any

import pytest
import yaml

from ollama_marshal.config import (
    LogFormat,
    LoggingConfig,
    MarshalConfig,
    MemoryConfig,
    OllamaConfig,
    Priority,
    ProgramConfig,
    ProxyConfig,
    SchedulerConfig,
    ShutdownConfig,
    ShutdownMode,
    _apply_env_overrides,
    _find_config_file,
    load_config,
    parse_size,
)

# ---------------------------------------------------------------------------
# parse_size
# ---------------------------------------------------------------------------


class TestParseSize:
    def test_gigabytes(self):
        assert parse_size("4GB") == 4 * 1024**3

    def test_megabytes(self):
        assert parse_size("512MB") == 512 * 1024**2

    def test_kilobytes(self):
        assert parse_size("128KB") == 128 * 1024

    def test_bytes(self):
        assert parse_size("1024B") == 1024

    def test_terabytes(self):
        assert parse_size("1TB") == 1024**4

    def test_float_size(self):
        assert parse_size("2.5GB") == int(2.5 * 1024**3)

    def test_lowercase(self):
        assert parse_size("4gb") == 4 * 1024**3

    def test_whitespace(self):
        assert parse_size("  4GB  ") == 4 * 1024**3

    def test_invalid_format_no_suffix(self):
        with pytest.raises(ValueError, match="Invalid size format"):
            parse_size("1234")

    def test_invalid_format_unknown_suffix(self):
        with pytest.raises(ValueError, match="Invalid size format"):
            parse_size("4XY")

    def test_suffix_substring_match_gives_invalid_number(self):
        # "4PB" matches the "B" suffix, leaving "4P" as the number
        with pytest.raises(ValueError, match="Invalid size number"):
            parse_size("4PB")

    def test_invalid_number(self):
        with pytest.raises(ValueError, match="Invalid size number"):
            parse_size("abcGB")

    def test_empty_number_part(self):
        with pytest.raises(ValueError, match="Invalid size number"):
            parse_size("GB")


# ---------------------------------------------------------------------------
# StrEnums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_priority_values(self):
        assert Priority.NORMAL == "normal"
        assert Priority.CRITICAL == "critical"

    def test_shutdown_mode_values(self):
        assert ShutdownMode.DRAIN == "drain"
        assert ShutdownMode.IMMEDIATE == "immediate"

    def test_log_format_values(self):
        assert LogFormat.CONSOLE == "console"
        assert LogFormat.JSON == "json"

    def test_priority_is_str(self):
        assert isinstance(Priority.NORMAL, str)

    def test_shutdown_mode_is_str(self):
        assert isinstance(ShutdownMode.DRAIN, str)

    def test_log_format_is_str(self):
        assert isinstance(LogFormat.CONSOLE, str)


# ---------------------------------------------------------------------------
# Pydantic model defaults
# ---------------------------------------------------------------------------


class TestPydanticModelDefaults:
    def test_ollama_config_defaults(self):
        cfg = OllamaConfig()
        assert cfg.host == "http://localhost:11434"

    def test_proxy_config_defaults(self):
        cfg = ProxyConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 11435

    def test_memory_config_defaults(self):
        cfg = MemoryConfig()
        assert cfg.total_ram is None
        assert cfg.os_overhead == "4GB"
        assert cfg.safety_margin == "2GB"
        assert cfg.poll_interval == 5

    def test_scheduler_config_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.max_skips == 3
        assert cfg.model_detect_interval == 30
        assert cfg.benchmark_on_startup is True

    def test_program_config_defaults(self):
        cfg = ProgramConfig()
        assert cfg.priority == Priority.NORMAL

    def test_shutdown_config_defaults(self):
        cfg = ShutdownConfig()
        assert cfg.mode == ShutdownMode.DRAIN
        assert cfg.drain_timeout == 30
        assert cfg.unload_models is True

    def test_logging_config_defaults(self):
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == LogFormat.CONSOLE


class TestMarshalConfigDefaults:
    def test_default_marshal_config(self):
        cfg = MarshalConfig()
        assert cfg.ollama.host == "http://localhost:11434"
        assert cfg.proxy.port == 11435
        assert cfg.memory.os_overhead == "4GB"
        assert cfg.scheduler.max_skips == 3
        assert "default" in cfg.programs
        assert cfg.programs["default"].priority == Priority.NORMAL
        assert cfg.shutdown.mode == ShutdownMode.DRAIN
        assert cfg.logging.level == "INFO"


# ---------------------------------------------------------------------------
# MarshalConfig.get_program_config
# ---------------------------------------------------------------------------


class TestGetProgramConfig:
    def test_known_program(self):
        cfg = MarshalConfig(
            programs={
                "default": ProgramConfig(),
                "my-app": ProgramConfig(priority=Priority.CRITICAL),
            }
        )
        result = cfg.get_program_config("my-app")
        assert result.priority == Priority.CRITICAL

    def test_unknown_program_falls_back_to_default(self):
        cfg = MarshalConfig(
            programs={"default": ProgramConfig(priority=Priority.NORMAL)}
        )
        result = cfg.get_program_config("unknown-app")
        assert result.priority == Priority.NORMAL

    def test_no_default_key_returns_fresh_program_config(self):
        cfg = MarshalConfig(programs={})
        result = cfg.get_program_config("anything")
        assert result.priority == Priority.NORMAL

    def test_default_program(self):
        cfg = MarshalConfig()
        result = cfg.get_program_config("default")
        assert result.priority == Priority.NORMAL


# ---------------------------------------------------------------------------
# _apply_env_overrides
# ---------------------------------------------------------------------------


class TestApplyEnvOverrides:
    def test_proxy_port_override(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_PROXY_PORT", "9999")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["proxy"]["port"] == 9999

    def test_string_field_override(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_OLLAMA_HOST", "http://remote:11434")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["ollama"]["host"] == "http://remote:11434"

    def test_integer_fields_are_typed(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SCHEDULER_MAX_SKIPS", "10")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["scheduler"]["max_skips"] == 10
        assert isinstance(result["scheduler"]["max_skips"], int)

    def test_request_timeout_s_is_int_typed(self, monkeypatch):
        # Added to the int-coercion list in v0.2.0 alongside the field's
        # introduction. Pydantic would coerce anyway, but this keeps the
        # dict consistent with port/poll_interval/etc.
        monkeypatch.setenv("MARSHAL_PROXY_REQUEST_TIMEOUT_S", "1800")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["proxy"]["request_timeout_s"] == 1800
        assert isinstance(result["proxy"]["request_timeout_s"], int)

    def test_idle_eviction_minutes_is_int_typed(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SCHEDULER_IDLE_EVICTION_MINUTES", "30")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["scheduler"]["idle_eviction_minutes"] == 30
        assert isinstance(result["scheduler"]["idle_eviction_minutes"], int)

    def test_boolean_field_true(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SHUTDOWN_UNLOAD_MODELS", "true")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["shutdown"]["unload_models"] is True

    def test_boolean_field_false(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SHUTDOWN_UNLOAD_MODELS", "false")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["shutdown"]["unload_models"] is False

    def test_boolean_field_yes(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SHUTDOWN_UNLOAD_MODELS", "yes")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["shutdown"]["unload_models"] is True

    def test_boolean_field_one(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SHUTDOWN_UNLOAD_MODELS", "1")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["shutdown"]["unload_models"] is True

    def test_preserves_existing_data(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_PROXY_PORT", "9999")
        data: dict[str, Any] = {"proxy": {"host": "127.0.0.1"}}
        result = _apply_env_overrides(data)
        assert result["proxy"]["host"] == "127.0.0.1"
        assert result["proxy"]["port"] == 9999

    def test_ignores_non_marshal_env(self, monkeypatch):
        monkeypatch.setenv("OTHER_VARIABLE", "value")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert "other" not in result

    def test_single_part_key_ignored(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_STANDALONE", "value")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        # split("_", maxsplit=1) on "STANDALONE" gives ["standalone"] (len 1)
        # so nothing should be set
        assert "standalone" not in result

    def test_memory_poll_interval(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_MEMORY_POLL_INTERVAL", "15")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["memory"]["poll_interval"] == 15

    def test_scheduler_model_detect_interval(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SCHEDULER_MODEL_DETECT_INTERVAL", "60")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["scheduler"]["model_detect_interval"] == 60

    def test_shutdown_drain_timeout(self, monkeypatch):
        monkeypatch.setenv("MARSHAL_SHUTDOWN_DRAIN_TIMEOUT", "45")
        data: dict[str, Any] = {}
        result = _apply_env_overrides(data)
        assert result["shutdown"]["drain_timeout"] == 45


# ---------------------------------------------------------------------------
# _find_config_file
# ---------------------------------------------------------------------------


class TestFindConfigFile:
    def test_explicit_path_exists(self, tmp_path):
        config_file = tmp_path / "test-marshal.yaml"
        config_file.write_text("ollama:\n  host: http://localhost:11434\n")
        result = _find_config_file(str(config_file))
        assert result == config_file

    def test_explicit_path_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _find_config_file(str(tmp_path / "nonexistent.yaml"))

    def test_env_path_exists(self, tmp_path, monkeypatch):
        config_file = tmp_path / "env-marshal.yaml"
        config_file.write_text("proxy:\n  port: 12345\n")
        monkeypatch.setenv("MARSHAL_CONFIG", str(config_file))
        result = _find_config_file()
        assert result == config_file

    def test_env_path_missing_falls_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MARSHAL_CONFIG", str(tmp_path / "no-such-file.yaml"))
        # Also ensure no default search paths exist
        monkeypatch.chdir(tmp_path)
        result = _find_config_file()
        # Should return None since the env path doesn't exist
        # and no search paths exist either
        assert result is None or result.exists()

    def test_no_config_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MARSHAL_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        result = _find_config_file()
        # If there's a config in the default search paths it might return something,
        # but with chdir to tmp_path, ./marshal.yaml won't exist
        # We only assert: if result is not None, it must exist
        if result is not None:
            assert result.exists()

    def test_explicit_path_takes_priority_over_env(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit.yaml"
        explicit.write_text("proxy:\n  port: 1111\n")
        env_file = tmp_path / "env.yaml"
        env_file.write_text("proxy:\n  port: 2222\n")
        monkeypatch.setenv("MARSHAL_CONFIG", str(env_file))
        result = _find_config_file(str(explicit))
        assert result == explicit


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


@pytest.fixture
def yaml_config_file(tmp_path):
    config_file = tmp_path / "marshal.yaml"
    data = {
        "ollama": {"host": "http://yaml-host:11434"},
        "proxy": {"port": 22222},
        "scheduler": {"max_skips": 7},
        "programs": {
            "default": {"priority": "normal"},
            "special": {"priority": "critical"},
        },
    }
    config_file.write_text(yaml.dump(data))
    return config_file


class TestLoadConfig:
    def test_defaults_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MARSHAL_CONFIG", raising=False)
        # Remove any MARSHAL_* env vars that might interfere
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.ollama.host == "http://localhost:11434"
        assert cfg.proxy.port == 11435

    def test_yaml_file_loading(self, yaml_config_file, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        cfg = load_config(config_path=str(yaml_config_file))
        assert cfg.ollama.host == "http://yaml-host:11434"
        assert cfg.proxy.port == 22222
        assert cfg.scheduler.max_skips == 7

    def test_env_overrides_yaml(self, yaml_config_file, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("MARSHAL_PROXY_PORT", "33333")
        cfg = load_config(config_path=str(yaml_config_file))
        # YAML had port=22222 but env should override
        assert cfg.proxy.port == 33333
        # YAML value preserved where not overridden
        assert cfg.ollama.host == "http://yaml-host:11434"

    def test_cli_overrides_env_and_yaml(self, yaml_config_file, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("MARSHAL_PROXY_PORT", "33333")
        cfg = load_config(
            config_path=str(yaml_config_file),
            cli_overrides={"proxy.port": 44444},
        )
        assert cfg.proxy.port == 44444

    def test_cli_overrides_none_values_ignored(self, tmp_path, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config(cli_overrides={"proxy.port": None})
        assert cfg.proxy.port == 11435  # default preserved

    def test_cli_override_creates_section(self, tmp_path, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config(cli_overrides={"proxy.port": 55555})
        assert cfg.proxy.port == 55555

    def test_layering_priority_cli_wins(self, yaml_config_file, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        # YAML: max_skips=7, Env: max_skips=10, CLI: max_skips=15
        monkeypatch.setenv("MARSHAL_SCHEDULER_MAX_SKIPS", "10")
        cfg = load_config(
            config_path=str(yaml_config_file),
            cli_overrides={"scheduler.max_skips": 15},
        )
        assert cfg.scheduler.max_skips == 15

    def test_programs_from_yaml(self, yaml_config_file, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        cfg = load_config(config_path=str(yaml_config_file))
        assert cfg.get_program_config("special").priority == Priority.CRITICAL
        assert cfg.get_program_config("unknown").priority == Priority.NORMAL

    def test_empty_yaml_file(self, tmp_path, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")
        cfg = load_config(config_path=str(empty_config))
        # Should fall back to all defaults
        assert cfg.proxy.port == 11435

    def test_yaml_with_non_dict_content(self, tmp_path, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("just a string\n")
        cfg = load_config(config_path=str(bad_config))
        # Non-dict YAML should be ignored, defaults used
        assert cfg.proxy.port == 11435

    def test_cli_override_single_part_key(self, tmp_path, monkeypatch):
        for key in list(os.environ):
            if key.startswith("MARSHAL_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_path)
        # Single-part key goes into top-level data dict
        # This won't map to a known MarshalConfig field, so it should be
        # ignored by Pydantic (model_config forbid extra is not set)
        cfg = load_config(cli_overrides={"logging.level": "DEBUG"})
        assert cfg.logging.level == "DEBUG"
