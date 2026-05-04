"""Configuration loading with YAML, environment variable, and CLI overrides."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class KVCacheType(StrEnum):
    """KV cache quantization for an Ollama instance.

    Lower precision shrinks per-slot VRAM (KV cache scales with
    ``num_ctx``) at some quality cost. f16 is full half-precision;
    q8_0 is 8-bit quantized (~50% smaller, usually invisible quality
    drop on chat); q4_0 is 4-bit (~75% smaller, more visible on
    long-context reasoning).
    """

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


# Tier labels used in audit logs and ``app.state`` introspection. Free-form
# strings, but the canonical names match what's recommended in the README.
TIER_PRIMARY = "primary"
TIER_FALLBACK = "fallback"
TIER_LAST_RESORT = "last_resort"


class OllamaInstance(BaseModel):
    """One Ollama process with a specific KV cache precision.

    Multi-instance setup runs two or three Ollama processes on
    different ports, each with a different ``OLLAMA_KV_CACHE_TYPE``
    env var. Marshal routes requests between them to relieve memory
    pressure without summarizing or trimming the prompt.

    Single-instance setups (the default) just use one entry with
    ``kv_cache_type=f16``. The legacy singular ``ollama: {host: ...}``
    config form is auto-promoted to a single ``[{...}]`` list at load
    time — see ``MarshalConfig`` validator.

    Frozen so instances are hashable: routing uses them as dict keys
    + set members, and they're set-once-at-startup config data anyway.
    """

    model_config = ConfigDict(frozen=True)

    url: str = Field(
        ...,
        description=(
            "Full base URL of this Ollama instance, e.g. "
            "'http://localhost:11434'. Must be reachable from marshal's "
            "host (typically all instances run on the same machine). "
            "Normalized at validation time: trailing slash stripped + "
            "scheme/host lowercased so cosmetic-only differences "
            "(e.g. 'http://Localhost:11434/' vs 'http://localhost:11434') "
            "don't bypass the duplicate-URL check on MarshalConfig."
        ),
    )

    @field_validator("url", mode="after")
    @classmethod
    def _normalize_url(cls, v: str) -> str:
        """Strip trailing slash + lowercase the scheme/host."""
        v = v.rstrip("/")
        # Lowercase scheme and host but leave anything after the first
        # slash alone (Ollama doesn't use path components today, but
        # don't lowercase a future query string in place).
        if "://" in v:
            scheme, _, rest = v.partition("://")
            host_part, slash, after = rest.partition("/")
            v = f"{scheme.lower()}://{host_part.lower()}"
            if slash:
                v = f"{v}/{after}"
        return v

    kv_cache_type: KVCacheType = Field(
        default=KVCacheType.F16,
        description=(
            "Must match the ``OLLAMA_KV_CACHE_TYPE`` env var the "
            "instance was launched with. Marshal does NOT verify this "
            "at startup — mismatches surface as wrong fit calculations "
            "and surprising OOMs."
        ),
    )
    tier_label: str = Field(
        default=TIER_PRIMARY,
        description=(
            "Free-form label used in audit logs + status output. "
            "Canonical names: 'primary' (highest precision, default "
            "target), 'fallback' (mid precision, used to relieve "
            "pressure), 'last_resort' (lowest precision, used only "
            "when nothing else fits)."
        ),
    )


class OllamaConfig(BaseModel):
    """Legacy single-instance Ollama config.

    Kept for backward compatibility. New configs should use the list
    form ``ollama: [...]`` with explicit ``OllamaInstance`` entries.
    A ``MarshalConfig`` validator converts singular form to a single-
    instance list at load time.
    """

    host: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )


class ProxyConfig(BaseModel):
    """Proxy server settings."""

    host: str = Field(default="127.0.0.1", description="Proxy bind address")
    port: int = Field(default=11435, description="Proxy listen port")


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
        ge=1,
        description=(
            "Seconds between /api/tags polls for model adds/removals. "
            "Drives ModelRegistry's periodic refresh: a removed model "
            "(`ollama rm`) starts fail-fasting with 404 within this window "
            "instead of preload-looping into 502. Minimum 1s — values "
            "below would tight-loop /api/tags."
        ),
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
    benchmark_on_startup: bool = Field(
        default=True,
        description=(
            "Run the model-size benchmark task on lifespan startup. "
            "Probes /api/show metadata for every installed model and "
            "loads each unbenchmarked model briefly to record its VRAM "
            "footprint. Default ON for production so routing decisions "
            "have accurate sizes. Disable in integration tests that use "
            "fault-injection proxies — the benchmark would saturate the "
            "real Ollama through the proxy and starve other test "
            "requests."
        ),
    )
    ollama_forward_timeout_s: int = Field(
        default=3600,
        ge=1,
        description=(
            "Wall-clock budget for a single marshal→Ollama HTTP forward, "
            "in seconds. Default 3600 (1h) — generous enough for big-"
            "model long-context inference. Catches a hung Ollama "
            "process so the request fails with a clear 504 instead of "
            "blocking the queue forever. Per-request override via the "
            "``X-Request-Timeout`` header. Replaces the v0.6.3 "
            "``proxy.request_timeout_s`` field which gated the "
            "client→marshal hop (now unbounded — async clients wait "
            "indefinitely)."
        ),
    )
    preload_backoff_base_s: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Initial backoff after a preload failure, in seconds. The "
            "scheduler tracks per-model preload failure state; the "
            "first failure parks future preload attempts on that model "
            "for ``base_s`` (with full jitter), the second for ``base_s "
            "* 2``, etc., capped at ``preload_backoff_max_s``. Without "
            "this, the 0.1s scheduler tick would hammer a recovering "
            "Ollama with ~10 preload calls/sec while it's unreachable."
        ),
    )
    preload_backoff_max_s: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Cap on a single preload backoff sleep, in seconds. "
            "Bounds geometric growth so the backoff doesn't drift to "
            "minutes after many failures."
        ),
    )
    preload_max_consecutive_failures: int = Field(
        default=5,
        ge=1,
        description=(
            "After this many consecutive preload failures for the same "
            "model, give up and fail every queued envelope for that "
            "model with ``PreloadFailedError`` (surfaced to clients as "
            "a 502). Without this, a permanently-unreachable model "
            "would leave its envelopes parked in the queue forever. "
            "Per-model failure state resets on the next successful "
            "preload."
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


class ProgramContextProfile(BaseModel):
    """Per-program num_ctx floor and ceiling.

    `typical_num_ctx` is a FLOOR: a tool-calling program's first call
    should already get enough room for later rounds, so we don't pay
    a reload penalty on round 2. `max_num_ctx` is a CEILING: defensive
    against runaway prompts that would force a model reload at maximum
    context.

    If a program has no profile, its requests size to actual prompt
    need only — no floor, no ceiling beyond the model's max.
    """

    typical_num_ctx: int = Field(
        default=4096,
        description=(
            "Floor: prompt-driven sizing won't go below this for this "
            "program. Set to the typical context the program actually "
            "needs after a few tool-calling rounds."
        ),
    )
    max_num_ctx: int = Field(
        default=131072,
        description=(
            "Ceiling: clamp prompt-driven sizing to at most this. "
            "Defensive against runaway prompts."
        ),
    )


class ContextConfig(BaseModel):
    """Dynamic `num_ctx` injection settings.

    v0.3.0 injected `num_ctx = model_max_context` unconditionally,
    which forces Ollama to pre-allocate KV cache for the full window
    even on a 100-token prompt — a 70 GB allocation on a 4B model with
    a 262K context window. v0.4.0 instead sizes `num_ctx` to actual
    prompt + completion budget, rounded up to a power-of-2 boundary,
    then clamped to `[program.typical_num_ctx, program.max_num_ctx]`
    if a profile exists, and finally to `model.max_context_length`.

    Marshal NEVER silently truncates a real prompt. When a request
    needs more context than the model's currently-allocated slot size,
    marshal triggers a reload at the larger size (see Surface C1
    Dim 4). To opt out and restore Ollama's default behavior (and its
    silent-truncation bug), set `injection_enabled: false`.
    """

    injection_enabled: bool = Field(
        default=True,
        description=(
            "Inject `options.num_ctx` based on prompt size. Disable to "
            "fall back to Ollama's default (which may silently "
            "truncate)."
        ),
    )
    default_completion_budget: int = Field(
        default=4096,
        description=(
            "Tokens reserved for the model's response when the client "
            "doesn't set `options.num_predict`. Added to the input "
            "estimate before rounding to a power-of-2 boundary."
        ),
    )
    safety_buffer_tokens: int = Field(
        default=256,
        description=(
            "Extra tokens added to the prompt+completion estimate to "
            "absorb tokenizer-overestimate inaccuracy."
        ),
    )
    programs: dict[str, ProgramContextProfile] = Field(
        default_factory=dict,
        description=(
            "Per-program floor/ceiling profiles keyed by `X-Program-ID` "
            "header value. Programs without a profile get pure "
            "prompt-driven sizing."
        ),
    )


class RetryConfig(BaseModel):
    """Marshal-side retry tuning for transient Ollama failures.

    When Ollama briefly flaps (daemon recycling, transient 502/503),
    marshal absorbs the blip via in-process retry so the client never
    sees the failure. Retries are conservative by default:

    - **Streaming requests are NEVER retried.** Once chunks have shipped,
      we can't safely re-issue.
    - **ReadTimeout is NOT retried** unless `read_timeouts` is enabled
      — Ollama may have already started generating, so retrying could
      double-bill or re-execute a tool call.
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
    read_timeouts: bool = Field(
        default=False,
        description=(
            "When True, retry on ReadTimeout (Ollama may have already "
            "started generating — risk of re-execution). Default False. "
            "Safe to enable for idempotent endpoints (embeddings). "
            "Field name kept short ('read_timeouts' not "
            "'retry_read_timeouts') so the env-var override "
            "MARSHAL_RETRY_READ_TIMEOUTS parses correctly: the env "
            "parser splits at the FIRST underscore, so the field name "
            "must be the part after 'retry'."
        ),
    )
    max_per_request_attempts: int = Field(
        default=10,
        description=(
            "Server-side cap on the per-request `X-Marshal-Retry-Max` "
            "header. Even an adversarial client can't extend retry "
            "beyond this number. Default 10 — generous, but bounds the "
            "worst-case wall-clock on a stuck request."
        ),
    )


class AdminConfig(BaseModel):
    """Operator/admin endpoints for pause/resume + future maintenance ops.

    OFF by default — admin endpoints are gated behind
    ``pause_endpoints_enabled`` AND a bearer token. Operators opt in
    by setting both fields in ``marshal.yaml``. Used by integration
    tests (and future deploy-time draining workflows) to safely
    suspend dispatch from the queue without rejecting incoming
    requests.

    Pause semantics: SOFT. Setting the dispatch-paused flag freezes
    the scheduler's queue draining but does NOT reject new requests
    (they continue to enqueue normally). Tests carrying
    ``X-Marshal-Test-Bypass: <token>`` (matching ``test_bypass_token``)
    bypass the pause and dispatch immediately. Resume drops the flag
    and the scheduler picks up where it left off, draining the
    accumulated queue at normal speed.
    """

    pause_endpoints_enabled: bool = Field(
        default=False,
        description=(
            "Expose ``POST /api/marshal/admin/pause`` and "
            "``POST /api/marshal/admin/resume``. Default OFF — "
            "operators opt in explicitly. When True, ``admin_token`` "
            "must also be set."
        ),
    )
    admin_token: str | None = Field(
        default=None,
        description=(
            "Bearer token for the ``X-Marshal-Admin-Token`` header on "
            "admin endpoints. Required when "
            "``pause_endpoints_enabled=True``. Generate with "
            "``openssl rand -hex 32``."
        ),
    )
    test_bypass_token: str | None = Field(
        default=None,
        description=(
            "Token for the ``X-Marshal-Test-Bypass`` header. Requests "
            "carrying this header bypass the dispatch-pause guard. "
            "Used by integration tests to fire requests against a "
            "paused prod marshal. Optional — leave unset to disallow "
            "bypass entirely."
        ),
    )

    @model_validator(mode="after")
    def _require_admin_token_when_enabled(self) -> AdminConfig:
        """Reject configs that enable admin endpoints without a token.

        Without this guard, an operator who sets
        ``pause_endpoints_enabled=True`` and forgets the token would
        expose unauthenticated admin endpoints. Fail fast at config
        load instead of silently accepting them.
        """
        if self.pause_endpoints_enabled and not self.admin_token:
            msg = (
                "admin.pause_endpoints_enabled=True requires "
                "admin.admin_token to be set (use `openssl rand -hex 32`)"
            )
            raise ValueError(msg)
        return self


class DebugConfig(BaseModel):
    """Internal-state debug endpoints for integration tests.

    OFF by default — production marshal stays lean. Test configs
    flip ``endpoint_enabled`` to True. The env override
    ``MARSHAL_DEBUG_ENDPOINT_ENABLED=true`` works the same.
    """

    endpoint_enabled: bool = Field(
        default=False,
        description=(
            "Expose ``GET /api/marshal/debug`` returning scheduler "
            "metrics, ``allocated_num_ctx_per_model``, registry "
            "metadata cache contents, and other internal state. "
            "Used by integration tests that need to assert on "
            "marshal-internal state via HTTP rather than reaching "
            "into ``app.state._marshal_internals``."
        ),
    )


class MarshalConfig(BaseModel):
    """Root configuration for ollama-marshal."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    instances: list[OllamaInstance] = Field(
        default_factory=list,
        description=(
            "Multi-instance Ollama setup. Each entry is one Ollama "
            "process with its own URL + KV cache precision. Marshal "
            "routes requests between them to relieve memory pressure "
            "without summarizing or trimming the prompt. Empty list "
            "(the default) → marshal auto-derives a single-instance "
            "list from the legacy ``ollama.host`` field, mirroring "
            "v0.4.0 behavior. See README 'Multi-instance setup'."
        ),
    )
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
    context: ContextConfig = Field(default_factory=ContextConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @model_validator(mode="after")
    def _normalize_instances(self) -> MarshalConfig:
        """Backfill ``instances`` from singular ``ollama.host`` if empty.

        Existing v0.4.0 configs ship the legacy singular form
        (``ollama: {host: ...}``); marshal continues to honor that
        without requiring a config rewrite. After this validator,
        ``self.instances`` always has at least one entry, ordered by
        descending precision (f16 → q8_0 → q4_0). Routing logic
        treats ``instances[0]`` as the highest-precision tier.
        """
        if not self.instances:
            self.instances = [
                OllamaInstance(
                    url=self.ollama.host,
                    kv_cache_type=KVCacheType.F16,
                    tier_label=TIER_PRIMARY,
                )
            ]
            return self
        # Both legacy ``ollama.host`` and explicit ``instances`` set:
        # explicit list wins, but sync the legacy field to the primary
        # instance URL so consumers that still read ``config.ollama.host``
        # (memory.py, server.py, etc — Stage 2 will refactor these to
        # walk ``instances`` instead) don't silently use the legacy
        # value while routing uses a different one. The sync removes
        # the split-brain risk during the Stage 1 → Stage 2 transition.
        primary_url = self.instances[0].url
        if self.ollama.host != primary_url:
            # Pydantic's ``frozen=True`` on OllamaConfig isn't set, so
            # this assignment works. We don't log here because config
            # loaders may run before logging is configured; a startup
            # smoke-check in server.py logs the sync if it fired.
            self.ollama = OllamaConfig(host=primary_url)
        # Validate the explicit list form.
        seen_urls: set[str] = set()
        for inst in self.instances:
            if inst.url in seen_urls:
                msg = f"duplicate Ollama instance URL: {inst.url!r}"
                raise ValueError(msg)
            seen_urls.add(inst.url)
        # Sort by precision (highest first) so routing.pick_instance
        # can walk top-down. KVCacheType members are declared in
        # highest-to-lowest precision order, so deriving the rank
        # from enum declaration order keeps a single source of truth
        # — adding a new variant in the right slot is enough; no
        # parallel rank dict to update.
        precision_rank = {kv: i for i, kv in enumerate(KVCacheType)}
        self.instances.sort(key=lambda i: precision_rank[i.kv_cache_type])
        return self

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
                "idle_eviction_minutes",
                "parallel_per_model",
                "burst_hint_cap_multiplier",
                "burst_hint_aggregate_multiplier",
                "burst_hint_max_live",
                "retention_days",
                "max_size_mb",
                # v0.4.0: retry + context int fields
                "max_attempts",
                "max_per_request_attempts",
                "default_completion_budget",
                "safety_buffer_tokens",
                # v0.6.4: Hop 2 forward timeout + preload backoff giveup
                "ollama_forward_timeout_s",
                "preload_max_consecutive_failures",
            ):
                data[section][field] = int(value)
            elif field in (
                "burst_hint_ttl_s",
                "metrics_persist_interval_s",
                # v0.4.0: retry float fields
                "base_delay_s",
                "max_delay_s",
                # v0.6.4: preload backoff floats
                "preload_backoff_base_s",
                "preload_backoff_max_s",
            ):
                data[section][field] = float(value)
            elif field in (
                "unload_models",
                "enabled",
                # v0.4.0: retry + context bool fields
                "read_timeouts",
                "injection_enabled",
                # v0.5.0: scheduler benchmark gate
                "benchmark_on_startup",
                # v0.6.0: admin endpoint gate + debug endpoint gate
                "pause_endpoints_enabled",
                "endpoint_enabled",
            ):
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
