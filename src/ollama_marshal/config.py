"""Configuration loading with YAML, environment variable, and CLI overrides."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Default config search paths (in priority order)
_CONFIG_SEARCH_PATHS = [
    Path("./marshal.yaml"),
    Path.home() / ".ollama-marshal" / "marshal.yaml",
]


class Priority(StrEnum):
    """Program priority level."""

    NORMAL = "normal"
    CRITICAL = "critical"


class ShutdownMode(StrEnum):
    """Shutdown behavior mode."""

    DRAIN = "drain"
    IMMEDIATE = "immediate"


class LogFormat(StrEnum):
    """Logging output format."""

    CONSOLE = "console"
    JSON = "json"


class OllamaConfig(BaseModel):
    """Ollama connection settings."""

    host: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )


class ProxyConfig(BaseModel):
    """Proxy server settings."""

    host: str = Field(default="127.0.0.1", description="Proxy bind address")
    port: int = Field(default=11435, description="Proxy listen port")
    request_timeout_s: int = Field(
        default=3600,
        description=(
            "Server-side max wait for a queued request to be served, in seconds. "
            "Default is 1 hour — generous enough for big-model loads. "
            "Returns 504 if exceeded. Clients can override per-request via the "
            "X-Request-Timeout header (value in seconds, must be > 0)."
        ),
    )


class MemoryConfig(BaseModel):
    """Memory management settings."""

    total_ram: str | None = Field(
        default=None,
        description="Total RAM override (e.g., '64GB'). Auto-detected if omitted.",
    )
    os_overhead: str = Field(
        default="4GB",
        description="RAM reserved for the operating system",
    )
    safety_margin: str = Field(
        default="2GB",
        description="Safety buffer to avoid OOM",
    )
    poll_interval: int = Field(
        default=5,
        description="Seconds between /api/ps polls",
    )


class SchedulerConfig(BaseModel):
    """Scheduler settings."""

    max_skips: int = Field(
        default=3,
        description="Max times a request can be skipped before forced service",
    )
    model_detect_interval: int = Field(
        default=30,
        description="Seconds between /api/tags polls for new models",
    )
    idle_eviction_minutes: int = Field(
        default=15,
        description=(
            "Evict a loaded model after this many minutes of inactivity. "
            "0 disables time-based eviction (only memory-pressure eviction). "
            "Models with pending requests are never time-evicted."
        ),
    )
    parallel_per_model: int = Field(
        default=1,
        description=(
            "Max concurrent inference dispatches per loaded model. Default 1 "
            "matches v0.2.x sequential behavior. To benefit from parallel "
            "execution: (1) set OLLAMA_NUM_PARALLEL >= this value at Ollama "
            "server startup so Ollama allocates enough KV cache slots, "
            "(2) raise this value to match. Marshal scales actual concurrency "
            "with queue depth (a 1-envelope queue still serves 1 at a time)."
        ),
    )
    burst_hint_ttl_s: float = Field(
        default=30.0,
        description=(
            "Seconds an X-Burst-Size hint stays active without renewal. "
            "Each request from the same program-model pair refreshes the "
            "timer; after this many seconds of silence the hint expires "
            "and falls back to actual queue depth. Programs that pace "
            "their tool-calling loops slower than this default will need "
            "to raise the value."
        ),
    )
    burst_hint_cap_multiplier: int = Field(
        default=4,
        description=(
            "Per-pair X-Burst-Size cap = max_skips * this. With defaults "
            "(max_skips=3, multiplier=4) a single program can claim up to "
            "12 boost on its model. Caps adversarial header values."
        ),
    )
    burst_hint_aggregate_multiplier: int = Field(
        default=8,
        description=(
            "Per-model aggregate burst-boost cap = max_skips * this. "
            "Even if many distinct program_ids each register hints at "
            "the per-pair cap, the per-model summed boost is clamped at "
            "this value. Defends against program_id-flooding attacks."
        ),
    )
    burst_hint_max_live: int = Field(
        default=256,
        description=(
            "Max number of live (program_id, model) burst-hint entries "
            "stored at any time. New pairs are dropped (record returns 0) "
            "when the dict is at capacity. Refreshing an existing pair "
            "always succeeds. Prevents memory exhaustion from flooded "
            "distinct pairs within the TTL window."
        ),
    )
    metrics_path: str = Field(
        default="~/.ollama-marshal/metrics.json",
        description=(
            "Path to the persisted-metrics JSON file. Lifetime counters "
            "(requests_served, model_swaps, evictions, total_wait_ms) "
            "are restored from this file on startup and saved on shutdown "
            "+ every metrics_persist_interval_s seconds during runtime."
        ),
    )
    metrics_persist_interval_s: float = Field(
        default=60.0,
        description=(
            "Seconds between background metrics-snapshot writes. Trades "
            "off disk wear vs. data-loss window on hard crash. Lower = "
            "tighter recovery, higher disk I/O; higher = vice versa."
        ),
    )


class ProgramConfig(BaseModel):
    """Per-program configuration."""

    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Program priority level",
    )


class ShutdownConfig(BaseModel):
    """Shutdown behavior settings."""

    mode: ShutdownMode = Field(
        default=ShutdownMode.DRAIN,
        description="Shutdown mode",
    )
    drain_timeout: int = Field(
        default=30,
        description="Max seconds to wait for drain",
    )
    unload_models: bool = Field(
        default=True,
        description="Unload all models from Ollama on exit",
    )


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = Field(default="INFO", description="Log level")
    format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description="Log output format",
    )


class AuditConfig(BaseModel):
    """Audit-log feature flag and tuning.

    OFF by default. When enabled, marshal appends one JSON record per
    request lifecycle event (enqueued, served, failed, evicted) to
    `audit.jsonl`. Records contain ONLY metadata — never prompt text or
    response content. Designed for compliance / forensics / per-program
    usage analytics, not perf debugging.
    """

    enabled: bool = Field(
        default=False,
        description="Enable audit-log writes (default off — opt-in)",
    )
    path: str = Field(
        default="~/.ollama-marshal/audit.jsonl",
        description="Path to the JSONL audit file",
    )
    retention_days: int = Field(
        default=30,
        description=(
            "Auto-delete audit entries older than this many days. "
            "0 disables retention (file grows unbounded)."
        ),
    )
    max_size_mb: int = Field(
        default=100,
        description=(
            "Rotate the audit file (.1, .2, ...) when it exceeds this "
            "size in megabytes. 0 disables size-based rotation."
        ),
    )


class RetryConfig(BaseModel):
    """Marshal-side retry tuning for transient Ollama failures.

    When Ollama briefly flaps (daemon recycling, transient 502/503),
    marshal absorbs the blip via in-process retry so the client never
    sees the failure. Retries are conservative by default:

    - **Streaming requests are NEVER retried.** Once chunks have shipped,
      we can't safely re-issue.
    - **ReadTimeout is NOT retried** unless `retry_read_timeouts` is
      enabled — Ollama may have already started generating, so retrying
      could double-bill or re-execute a tool call.
    - Only `ConnectError`/`ConnectTimeout` and HTTP 502/503/504 retry
      by default.

    Per-request override: clients can send `X-Marshal-Retry-Max: 0` to
    disable retry on a single request, or a higher number to opt into
    more aggressive retry for known-idempotent calls.
    """

    enabled: bool = Field(
        default=True,
        description="Enable marshal-side retry on transient failures",
    )
    max_attempts: int = Field(
        default=3,
        description=(
            "Total attempts including the first try. 1 = no retry, "
            "3 = up to 2 retries. Bounded by total wall-clock cost: "
            "with default backoff, 3 attempts take ~3-5s worst case."
        ),
    )
    base_delay_s: float = Field(
        default=0.5,
        description=(
            "First-retry backoff before doubling. Full-jitter random "
            "in [0, base_delay_s] for first retry."
        ),
    )
    max_delay_s: float = Field(
        default=10.0,
        description=(
            "Cap on a single backoff sleep. Prevents pathological "
            "geometric growth on long retry budgets."
        ),
    )
    retry_read_timeouts: bool = Field(
        default=False,
        description=(
            "When True, retry on ReadTimeout (Ollama may have already "
            "started generating — risk of re-execution). Default False. "
            "Safe to enable for idempotent endpoints (embeddings)."
        ),
    )


class MarshalConfig(BaseModel):
    """Root configuration for ollama-marshal."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    programs: dict[str, ProgramConfig] = Field(
        default_factory=lambda: {"default": ProgramConfig()},
    )
    shutdown: ShutdownConfig = Field(default_factory=ShutdownConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    def get_program_config(self, program_id: str) -> ProgramConfig:
        """Get config for a program, falling back to 'default'."""
        return self.programs.get(
            program_id, self.programs.get("default", ProgramConfig())
        )


def parse_size(size_str: str) -> int:
    """Parse a human-readable size string to bytes.

    Args:
        size_str: Size string like '4GB', '512MB', '2.5GB'.

    Returns:
        Size in bytes.

    Raises:
        ValueError: If the size string format is invalid.
    """
    size_str = size_str.strip().upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    for suffix, multiplier in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            number_part = size_str[: -len(suffix)].strip()
            try:
                return int(float(number_part) * multiplier)
            except ValueError:
                msg = f"Invalid size number: {number_part}"
                raise ValueError(msg) from None

    msg = f"Invalid size format: {size_str}. Use B, KB, MB, GB, or TB suffix."
    raise ValueError(msg)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply MARSHAL_* environment variable overrides to config data.

    Environment variables use underscore-separated paths:
    MARSHAL_PROXY_PORT=11435 -> data["proxy"]["port"] = 11435
    MARSHAL_MEMORY_SAFETY_MARGIN=4GB -> data["memory"]["safety_margin"] = "4GB"
    """
    prefix = "MARSHAL_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("_", maxsplit=1)
        if len(parts) == 2:
            section, field = parts
            if section not in data:
                data[section] = {}
            # Try to preserve types for known integer fields
            if field in (
                "port",
                "poll_interval",
                "max_skips",
                "model_detect_interval",
                "drain_timeout",
                "request_timeout_s",
                "idle_eviction_minutes",
                "parallel_per_model",
                "burst_hint_cap_multiplier",
                "burst_hint_aggregate_multiplier",
                "burst_hint_max_live",
                "retention_days",
                "max_size_mb",
            ):
                data[section][field] = int(value)
            elif field in (
                "burst_hint_ttl_s",
                "metrics_persist_interval_s",
            ):
                data[section][field] = float(value)
            elif field in ("unload_models", "enabled"):
                data[section][field] = value.lower() in ("true", "1", "yes")
            else:
                data[section][field] = value
    return data


def _find_config_file(config_path: str | None = None) -> Path | None:
    """Find the config file using the search priority.

    Priority: explicit path -> MARSHAL_CONFIG env -> ./marshal.yaml
    -> ~/.ollama-marshal/marshal.yaml

    Args:
        config_path: Explicit config file path (from CLI flag).

    Returns:
        Path to config file, or None if no config file found.

    Raises:
        FileNotFoundError: If an explicit path was given but doesn't exist.
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        return path

    env_path = os.environ.get("MARSHAL_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    for search_path in _CONFIG_SEARCH_PATHS:
        if search_path.exists():
            return search_path

    return None


def load_config(
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> MarshalConfig:
    """Load configuration with layered overrides.

    Priority: CLI flags > env vars > YAML file > defaults.

    Args:
        config_path: Explicit config file path (from CLI --config flag).
        cli_overrides: Dict of overrides from CLI flags.

    Returns:
        Validated MarshalConfig instance.
    """
    data: dict[str, Any] = {}

    # Layer 1: Load from YAML file
    found_path = _find_config_file(config_path)
    if found_path:
        with open(found_path, encoding="utf-8") as f:
            file_data = yaml.safe_load(f)
            if file_data and isinstance(file_data, dict):
                data = file_data

    # Layer 2: Apply environment variable overrides
    data = _apply_env_overrides(data)

    # Layer 3: Apply CLI flag overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            parts = key.split(".", maxsplit=1)
            if len(parts) == 2:
                section, field = parts
                if section not in data:
                    data[section] = {}
                data[section][field] = value
            else:
                data[key] = value

    return MarshalConfig(**data)
