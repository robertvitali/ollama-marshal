"""FastAPI application wiring all components together."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import psutil
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ollama_marshal import __version__
from ollama_marshal.audit import NULL_AUDIT, AuditLogger
from ollama_marshal.config import MarshalConfig, ShutdownMode, load_config
from ollama_marshal.lifecycle import ModelLifecycle
from ollama_marshal.memory import MemoryManager
from ollama_marshal.openai_compat import (
    ollama_chat_to_openai,
    ollama_embedding_to_openai,
    ollama_generate_to_openai,
    parse_openai_chat_request,
    parse_openai_completion_request,
    parse_openai_embedding_request,
)
from ollama_marshal.queue import ModelQueues, PreloadFailedError, RequestEnvelope
from ollama_marshal.registry import ModelRegistry
from ollama_marshal.scheduler import Scheduler, SchedulerMetrics
from ollama_marshal.stream import forward_passthrough

logger = structlog.get_logger()

# Module-level state (initialized in lifespan)
_config: MarshalConfig
_queues: ModelQueues
_memory: MemoryManager
_registry: ModelRegistry
_lifecycle: ModelLifecycle
_scheduler: Scheduler
_audit: AuditLogger | Any = NULL_AUDIT
_started_at: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan managing startup and shutdown of all components."""
    global _config, _queues, _memory, _registry, _lifecycle, _scheduler
    global _audit, _started_at

    # Load config from app state (set by create_app)
    _config = app.state.config
    _started_at = time.monotonic()

    # Initialize components. ModelRegistry's on-disk cache paths can
    # be overridden via `app.state.registry_path` and
    # `app.state.metadata_path` — only used by integration tests so
    # the suite doesn't clobber the user's production-marshal registry
    # at ~/.ollama-marshal/. Production code never sets these
    # attributes; the registry uses its built-in defaults.
    _queues = ModelQueues()
    _memory = MemoryManager(_config)
    _registry = ModelRegistry(
        ollama_host=_config.ollama.host,
        registry_path=getattr(app.state, "registry_path", None),
        metadata_path=getattr(app.state, "metadata_path", None),
    )
    _lifecycle = ModelLifecycle(ollama_host=_config.ollama.host)
    _scheduler = Scheduler(
        queues=_queues,
        memory=_memory,
        registry=_registry,
        lifecycle=_lifecycle,
        config=_config,
    )

    # Restore lifetime metrics from disk so counters (requests_served,
    # model_swaps, evictions, total_wait_ms) survive restarts. Falls
    # back to fresh on any failure (logs a warning). Path comes from
    # config (overridable per-test via app.state.metrics_path).
    metrics_path = Path(
        getattr(app.state, "metrics_path", _config.scheduler.metrics_path)
    ).expanduser()
    restored = SchedulerMetrics.load_from(metrics_path)
    # Preserve the freshly-set started_at; only seed the persisted counters.
    _scheduler.metrics.requests_served = restored.requests_served
    _scheduler.metrics.model_swaps = restored.model_swaps
    _scheduler.metrics.evictions = restored.evictions
    _scheduler.metrics.total_wait_ms = restored.total_wait_ms
    # v0.4.0 counters — also persist across restarts.
    _scheduler.metrics.retries_attempted = restored.retries_attempted
    _scheduler.metrics.retries_succeeded = restored.retries_succeeded
    _scheduler.metrics.unexpected_unloads = restored.unexpected_unloads
    _scheduler.metrics.reload_count = restored.reload_count
    if restored.requests_served or restored.model_swaps or restored.evictions:
        logger.info(
            "server.metrics_restored",
            path=str(metrics_path),
            requests_served=restored.requests_served,
            model_swaps=restored.model_swaps,
            evictions=restored.evictions,
        )

    # Audit logger — opt-in feature flag. Off by default.
    _audit = AuditLogger(_config.audit) if _config.audit.enabled else NULL_AUDIT
    await _audit.start()
    # Hand to the scheduler so request lifecycle calls can record events
    # without taking a server-module dependency.
    _scheduler.audit = _audit

    # Initialize fire-and-forget task handles BEFORE the try block so the
    # finally clause can clean them up regardless of which startup step
    # raised. Without this pre-declaration, an exception between
    # `create_task(...)` and the yield would leak the task — the cleanup
    # branch would skip it because the variable would be unbound.
    benchmark_task: asyncio.Task[None] | None = None
    metrics_persister: asyncio.Task[None] | None = None

    try:
        # Start background tasks
        await _memory.start_polling()
        await _registry.initialize()
        # Benchmark task is gated by config so integration tests can opt
        # out — running the full per-model load/unload sweep through a
        # fault-injection proxy saturates the upstream and turns
        # /api/ps polls into 10s timeouts.
        if _config.scheduler.benchmark_on_startup:
            benchmark_task = asyncio.create_task(_registry.benchmark_unknown())
        await _scheduler.start()

        # Expose component instances on app.state under a private,
        # namespaced handle for integration tests (and any future consumer
        # that wants in-process introspection without reaching into module
        # globals). Underscore-prefixed name + SimpleNamespace wrapper
        # signals "test-only, not part of the public app surface" — any
        # third-party middleware or future endpoint that goes looking for
        # `app.state._marshal_internals` is making an explicit choice to
        # cross the encapsulation boundary, vs. accidentally bumping into
        # the components via attribute lookup. Production request handlers
        # continue to use the module globals (`_scheduler`, etc.) as they
        # always have. Additive only; no behavior change.
        #
        # TODO: when request handlers are refactored off module globals
        # (`_scheduler`, `_memory`, etc) and start reading components from
        # `request.app.state` directly, promote these to a public
        # `app.state.components` dataclass and drop the underscore prefix.
        # That refactor unblocks safe pytest-xdist parallelism inside
        # tests/integration/.
        app.state._marshal_internals = SimpleNamespace(
            scheduler=_scheduler,
            memory=_memory,
            registry=_registry,
            lifecycle=_lifecycle,
            queues=_queues,
        )

        # Periodically snapshot metrics to disk so a hard crash loses at
        # most metrics_persist_interval_s of counter data.
        metrics_persister = asyncio.create_task(
            _persist_metrics_loop(
                metrics_path, _config.scheduler.metrics_persist_interval_s
            )
        )

        logger.info(
            "server.started",
            proxy_port=_config.proxy.port,
            ollama_host=_config.ollama.host,
        )

        yield
    finally:
        # Stop the background persister cleanly before final save below.
        # Always await tasks (don't skip on .done()) so a task that finished
        # by raising surfaces its exception via structlog instead of riding
        # silently as "Task exception was never retrieved" at GC time.
        if metrics_persister is not None:
            await _shutdown_task(metrics_persister, "server.metrics_persister_failed")
        if benchmark_task is not None:
            await _shutdown_task(benchmark_task, "server.benchmark_task_failed")

        # Shutdown — drain BEFORE stopping the scheduler so requests
        # can still be processed during the drain phase
        logger.info("server.shutting_down", mode=_config.shutdown.mode.value)

        if _config.shutdown.mode == ShutdownMode.DRAIN:
            pending = await _queues.total_pending()
            if pending > 0:
                logger.info("server.draining", pending_requests=pending)
                deadline = time.monotonic() + _config.shutdown.drain_timeout
                while await _queues.total_pending() > 0 and time.monotonic() < deadline:
                    await asyncio.sleep(0.5)

        await _scheduler.stop()
        await _memory.stop_polling()

        if _config.shutdown.unload_models:
            loaded = list(_memory.get_loaded_models().keys())
            if loaded:
                logger.info("server.unloading_models", models=loaded)
                await _lifecycle.unload_all(loaded)

    # Final metrics snapshot — captures any counter changes from drain
    # phase plus any work that happened after the last periodic write.
    # Hoist sync I/O to a thread per CLAUDE.md async correctness rules.
    await asyncio.to_thread(_scheduler.metrics.save_to, metrics_path)
    logger.info("server.metrics_persisted", path=str(metrics_path))

    # Flush audit buffer + close (no-op when disabled).
    await _audit.stop()

    logger.info("server.stopped")


async def _shutdown_task(task: asyncio.Task[None], failure_event: str) -> None:
    """Cancel + await a fire-and-forget background task during shutdown.

    Always awaits the task (does not skip on ``done()``) so a task that
    finished by raising surfaces its exception via structlog instead of
    riding silently as ``Task exception was never retrieved`` at GC.
    Cancelled tasks log nothing — that's the expected shutdown path.

    Args:
        task: The fire-and-forget task to clean up.
        failure_event: structlog event name to use if the task raised
            something other than CancelledError. Caller picks a name
            that identifies which background task failed.
    """
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning(failure_event, error=str(exc), exc_info=True)


async def _persist_metrics_loop(path: Path, interval_s: float) -> None:
    """Background task: rewrite metrics snapshot every `interval_s` seconds.

    Disk write is hoisted to a thread (asyncio.to_thread) so a slow disk
    can't briefly stall the event loop and starve request dispatch.
    Matches the pattern used by the audit module's _flush_now.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        sched = globals().get("_scheduler")
        if sched is not None and hasattr(sched, "metrics"):
            await asyncio.to_thread(sched.metrics.save_to, path)


def create_app(config: MarshalConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Configuration to use. Loads from default sources if None.

    Returns:
        Configured FastAPI app instance.
    """
    if config is None:
        config = load_config()

    app = FastAPI(
        title="ollama-marshal",
        description="Model-aware scheduling proxy for Ollama",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.config = config

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Register all route handlers on the FastAPI app.

    Args:
        app: The FastAPI application.
    """
    # -- Queued inference endpoints (Ollama-native) --

    @app.post("/api/chat")
    async def api_chat(request: Request) -> Response:
        """Handle Ollama /api/chat requests through the scheduler."""
        body = await request.json()
        return await _enqueue_inference(request, body, "/api/chat")

    @app.post("/api/generate")
    async def api_generate(request: Request) -> Response:
        """Handle Ollama /api/generate requests through the scheduler."""
        body = await request.json()
        return await _enqueue_inference(request, body, "/api/generate")

    @app.post("/api/embeddings")
    async def api_embeddings(request: Request) -> Response:
        """Handle Ollama /api/embeddings requests through the scheduler."""
        body = await request.json()
        return await _enqueue_inference(request, body, "/api/embeddings")

    # -- Queued inference endpoints (OpenAI-compatible) --

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:
        """Handle OpenAI /v1/chat/completions requests."""
        body = await request.json()
        model, ollama_body, stream = parse_openai_chat_request(body)
        resp = await _enqueue_and_wait(request, model, ollama_body, "/api/chat", stream)
        if isinstance(resp, dict):
            return JSONResponse(ollama_chat_to_openai(resp, model))
        return resp

    @app.post("/v1/completions")
    async def openai_completions(request: Request) -> Response:
        """Handle OpenAI /v1/completions requests."""
        body = await request.json()
        model, ollama_body, stream = parse_openai_completion_request(body)
        resp = await _enqueue_and_wait(
            request, model, ollama_body, "/api/generate", stream
        )
        if isinstance(resp, dict):
            return JSONResponse(ollama_generate_to_openai(resp, model))
        return resp

    @app.post("/v1/embeddings")
    async def openai_embeddings(request: Request) -> Response:
        """Handle OpenAI /v1/embeddings requests."""
        body = await request.json()
        model, ollama_body = parse_openai_embedding_request(body)
        resp = await _enqueue_and_wait(
            request, model, ollama_body, "/api/embeddings", stream=False
        )
        if isinstance(resp, dict):
            return JSONResponse(ollama_embedding_to_openai(resp, model))
        return resp

    # -- Marshal status endpoint (registered at two paths) --

    async def _marshal_status_payload() -> dict[str, Any]:
        """Build the marshal status JSON payload (shared by both routes).

        The `memory` section has three subsections:
        - `system`: actual host RAM via psutil (what `top` would show)
        - `swap`: host swap usage via psutil (0 if unused)
        - top-level `total`/`available`/`used_by_models`: marshal's *budget*
          (total_ram - os_overhead - safety_margin), preserved for backward
          compatibility with v0.1.x consumers.
        """
        loaded = _memory.get_loaded_models()
        pending_by_model = await _queues.pending_by_model()
        pending_progs = await _queues.pending_programs_by_model()
        active_progs = _scheduler.active_programs_by_model()
        sysmem = psutil.virtual_memory()
        sysswap = psutil.swap_memory()
        # url → tier_label map for enriching loaded_models entries
        # (so consumers don't need to cross-reference the instances
        # array separately to get the tier of each loaded model).
        url_to_tier = {inst.url: inst.tier_label for inst in _memory.instances}
        # Per-instance breakdown — sibling of memory/queue/metrics.
        # Always present; for legacy single-instance configs the
        # validator backfills a single f16 entry, so this is a list
        # of length 1 in that case (consistent shape).
        instances_payload = []
        for inst in _memory.instances:
            here = _memory.get_loaded_models_on(inst.url)
            instances_payload.append(
                {
                    "url": inst.url,
                    "kv_cache_type": inst.kv_cache_type.value,
                    "tier_label": inst.tier_label,
                    "reachable": _memory.is_instance_reachable(inst.url),
                    "loaded_models": sorted(here.keys()),
                    "used_vram": sum(m.size_vram for m in here.values()),
                }
            )
        return {
            "uptime_seconds": round(time.monotonic() - _started_at, 1),
            "loaded_models": [
                {
                    "name": m.name,
                    "size_vram": m.size_vram,
                    "pending_requests": pending_by_model.get(m.name, 0),
                    # Union of programs with currently-pending requests and
                    # programs that have dispatched against this loaded model
                    # since it was loaded. Sorted, deduped.
                    "programs": sorted(
                        set(pending_progs.get(m.name, []))
                        | set(active_progs.get(m.name, []))
                    ),
                    # v0.5.0+: which instance holds this copy + its tier.
                    # For legacy single-instance setups this is the
                    # primary URL with tier_label="primary".
                    "instance_url": m.instance_url,
                    "tier_label": url_to_tier.get(m.instance_url),
                }
                for m in loaded.values()
            ],
            "instances": instances_payload,
            "memory": {
                # Marshal's budget (kept for backward compat with v0.1.x).
                "total": _memory.budget.total_ram,
                "available": _memory.available_vram(),
                "used_by_models": _memory.used_vram(),
                # NEW in v0.2.0: actual host RAM and swap from psutil.
                "system": {
                    "total": sysmem.total,
                    "available": sysmem.available,
                    "used": sysmem.used,
                    "percent": sysmem.percent,
                },
                "swap": {
                    "total": sysswap.total,
                    "used": sysswap.used,
                    "free": sysswap.free,
                    "percent": sysswap.percent,
                },
            },
            "queue": {
                "total_pending": await _queues.total_pending(),
                "by_model": pending_by_model,
            },
            "metrics": {
                "requests_served": _scheduler.metrics.requests_served,
                "model_swaps": _scheduler.metrics.model_swaps,
                "evictions": _scheduler.metrics.evictions,
                "average_wait_ms": round(_scheduler.metrics.average_wait_ms, 1),
                # v0.4.0 counters — wired through to /api/marshal/status
                # so the doctor CLI (and dashboard) can read them.
                "retries_attempted": _scheduler.metrics.retries_attempted,
                "retries_succeeded": _scheduler.metrics.retries_succeeded,
                "unexpected_unloads": _scheduler.metrics.unexpected_unloads,
                "reload_count": _scheduler.metrics.reload_count,
            },
        }

    @app.get("/api/marshal/status")
    async def marshal_status() -> dict[str, Any]:
        """Return proxy status as JSON (canonical path)."""
        return await _marshal_status_payload()

    @app.get("/status")
    async def status_alias() -> dict[str, Any]:
        """Short alias for /api/marshal/status.

        Convenient for `curl localhost:11435/status` instead of the longer path.
        """
        return await _marshal_status_payload()

    # -- Admin endpoints (gated by config + bearer token) --

    @app.post("/api/marshal/admin/pause")
    async def admin_pause(request: Request) -> Response:
        """Pause scheduler dispatch and wait for in-flight to drain.

        Soft-pause semantics: incoming requests still enqueue; only
        dispatch is frozen. Bypass-token requests still flow. Returns
        once the in-flight count drops to zero, or 409 if drain
        timeout exceeded. Schedules an auto-resume failsafe so a
        crashed test session doesn't leave prod paused forever.
        """
        if not _config.admin.pause_endpoints_enabled:
            return JSONResponse(
                {"error": "admin pause endpoints not enabled"}, status_code=404
            )
        if not _check_admin_token(request, _config):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body_bytes = await request.body()
        body_data: dict[str, Any] = {}
        if body_bytes:
            try:
                import json

                body_data = json.loads(body_bytes)
            except (ValueError, TypeError):
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        drain_timeout_s = float(body_data.get("drain_timeout_s", 60.0))
        auto_resume_after_seconds = float(
            body_data.get("auto_resume_after_seconds", 300.0)
        )

        queue_depth_before = await _queues.total_pending()
        drained = await _scheduler.pause(
            drain_timeout_s=drain_timeout_s,
            auto_resume_after_seconds=auto_resume_after_seconds,
        )
        auto_resume_at = (
            datetime.now(UTC) + timedelta(seconds=auto_resume_after_seconds)
        ).isoformat()

        if not drained:
            await _audit.record(event="admin.drain_timeout_exceeded")
            logger.warning(
                "admin.drain_timeout_exceeded",
                drain_timeout_s=drain_timeout_s,
                in_flight=_scheduler.in_flight_count(),
            )
            return JSONResponse(
                {
                    "error": (
                        "drain timeout exceeded — one or more inferences "
                        "still in flight"
                    ),
                    "in_flight": _scheduler.in_flight_count(),
                    "queued_at_pause": queue_depth_before,
                    "auto_resume_at": auto_resume_at,
                },
                status_code=409,
            )

        await _audit.record(event="admin.dispatch_paused")
        logger.info(
            "admin.dispatch_paused",
            queued_at_pause=queue_depth_before,
            auto_resume_after_seconds=auto_resume_after_seconds,
        )
        return JSONResponse(
            {
                "drained_in_flight": 0,
                "queued_at_pause": queue_depth_before,
                "auto_resume_at": auto_resume_at,
            }
        )

    @app.post("/api/marshal/admin/resume")
    async def admin_resume(request: Request) -> Response:
        """Resume scheduler dispatch and cancel the auto-resume timer."""
        if not _config.admin.pause_endpoints_enabled:
            return JSONResponse(
                {"error": "admin pause endpoints not enabled"}, status_code=404
            )
        if not _check_admin_token(request, _config):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        _scheduler.resume()
        queue_depth = await _queues.total_pending()
        await _audit.record(event="admin.dispatch_resumed")
        logger.info("admin.dispatch_resumed", queue_depth=queue_depth)
        return JSONResponse({"queue_depth": queue_depth})

    # -- Debug endpoint (gated by config) --

    @app.get("/api/marshal/debug")
    async def marshal_debug() -> Response:
        """Internal-state inspection for integration tests.

        Returns scheduler metrics, pause state, in-flight count, plus
        memory-manager allocated_num_ctx_per_model and registry
        metadata cache. Production marshal keeps this endpoint
        disabled (returns 404) to avoid leaking scheduler internals
        over the proxy port.
        """
        if not _config.debug.endpoint_enabled:
            return JSONResponse(
                {"error": "debug endpoint not enabled"}, status_code=404
            )
        # Flatten allocated_num_ctx across all instances. Tests usually
        # care "did marshal allocate a slot for this model" without
        # caring which instance — first-instance-wins matches the
        # MemoryManager.get_allocated_num_ctx(instance_url=None) path.
        allocated_per_model: dict[str, int] = {}
        for inst_models in _memory._allocated_num_ctx.values():
            for model_name, num_ctx in inst_models.items():
                allocated_per_model.setdefault(model_name, num_ctx)
        # Per-instance loaded models so tests can assert "model X is
        # loaded on instance Y" without parsing /api/marshal/status.
        loaded_per_instance: dict[str, list[str]] = {
            inst.url: list(_memory._loaded_models.get(inst.url, {}).keys())
            for inst in _memory._instances
        }
        # Registry metadata cache — surfaces architecture/max_ctx/
        # kv_per_slot_at_max_ctx per model. Uses to_json_dict() so
        # the computed kv_per_slot_at_max_ctx field is included
        # (asdict would skip it — it's a @property, not a field).
        # Only includes models marshal has probed; cold registry
        # returns empty dict.
        metadata_per_model = {
            model: meta.to_json_dict() for model, meta in _registry._metadata.items()
        }
        return JSONResponse(
            {
                "metrics": {
                    "requests_served": _scheduler.metrics.requests_served,
                    "model_swaps": _scheduler.metrics.model_swaps,
                    "evictions": _scheduler.metrics.evictions,
                    "retries_attempted": _scheduler.metrics.retries_attempted,
                    "retries_succeeded": _scheduler.metrics.retries_succeeded,
                    "reload_count": _scheduler.metrics.reload_count,
                    "unexpected_unloads": _scheduler.metrics.unexpected_unloads,
                },
                "scheduler": {
                    "is_paused": _scheduler.is_paused(),
                    "in_flight_count": _scheduler.in_flight_count(),
                },
                "memory": {
                    "allocated_num_ctx_per_model": allocated_per_model,
                    "loaded_per_instance": loaded_per_instance,
                },
                "registry": {
                    "metadata_per_model": metadata_per_model,
                },
            }
        )

    # -- Pass-through endpoints (safe read-only allowlist) --

    safe_passthrough = {
        "/api/tags",
        "/api/ps",
        "/api/show",
        "/api/version",
    }

    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "POST"],
    )
    async def passthrough(request: Request, path: str) -> Response:
        """Pass safe read-only requests to Ollama.

        Only allowlisted endpoints are forwarded. Destructive
        endpoints (/api/pull, /api/delete, /api/copy) are blocked.
        """
        full_path = f"/api/{path}"
        if full_path not in safe_passthrough:
            return JSONResponse(
                {
                    "error": f"Endpoint {full_path} is not proxied. "
                    "Use Ollama directly for model management."
                },
                status_code=403,
            )
        body = await request.body() if request.method == "POST" else None
        resp = await forward_passthrough(
            ollama_host=_config.ollama.host,
            method=request.method,
            path=full_path,
            body=body,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
        )


# Allowed character set for `X-Program-ID`. Restricting to a-zA-Z0-9
# plus a few punctuation chars prevents log injection (a value
# containing newlines or ANSI escapes corrupts structlog console
# output) and key-bytes amplification (an adversarial client cycling
# 10MB header values would otherwise inflate burst-hint dicts,
# `_active_programs`, and audit.jsonl by 10MB per distinct value).
_PROGRAM_ID_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)
# Cap on `X-Program-ID` length. 64 chars is more than enough for any
# legitimate program name. Longer values are truncated rather than
# rejected so a typo'd suffix doesn't surface as a fail-fast 400.
_PROGRAM_ID_MAX_LEN = 64


def _check_admin_token(request: Request, cfg: MarshalConfig) -> bool:
    """Validate the ``X-Marshal-Admin-Token`` header against config.

    Uses ``secrets.compare_digest`` for constant-time comparison so an
    adversarial caller can't probe the token byte-by-byte via timing.
    Returns False if the config has no token configured (admin endpoints
    are unusable without one — see ``AdminConfig`` validator).
    """
    if not cfg.admin.admin_token:
        return False
    received = request.headers.get("x-marshal-admin-token")
    if not received:
        return False
    return secrets.compare_digest(received, cfg.admin.admin_token)


def _is_bypass_pause(request: Request) -> bool:
    """Return True iff request carries a valid ``X-Marshal-Test-Bypass``.

    Used to flag envelopes that should dispatch even when the
    scheduler is paused. Reads config from ``request.app.state``
    rather than the module global so tests that bypass lifespan
    (don't initialize the global) still work. Returns False when:
    - app.state.config is missing (test bypassed lifespan)
    - admin.test_bypass_token is unset (bypass disabled by omission)
    - the header is missing or doesn't match

    Constant-time comparison defends against timing-leak token probing.
    """
    cfg = getattr(request.app.state, "config", None)
    if cfg is None or not cfg.admin.test_bypass_token:
        return False
    received = request.headers.get("x-marshal-test-bypass")
    if not received:
        return False
    return secrets.compare_digest(received, cfg.admin.test_bypass_token)


def _normalize_program_id(raw: str | None) -> str:
    """Sanitize a client-supplied `X-Program-ID` header value.

    Returns "default" for None/empty/all-disallowed input so downstream
    code paths see a stable identifier without special-casing.
    Otherwise: keep allowed chars only, truncate to `_PROGRAM_ID_MAX_LEN`.
    """
    if not raw:
        return "default"
    cleaned = "".join(c for c in raw if c in _PROGRAM_ID_ALLOWED)
    if not cleaned:
        return "default"
    return cleaned[:_PROGRAM_ID_MAX_LEN]


def _record_burst_hint(request: Request, program_id: str, model: str) -> None:
    """Read the optional X-Burst-Size header and forward to scheduler.

    Programs that submit work sequentially (e.g. tool-calling loops)
    can declare expected total demand via this header so marshal's
    eviction scorer treats their model as "expecting N more calls"
    even when only 1 is currently queued. Header is parsed defensively
    — non-numeric or non-positive values are silently ignored.
    """
    raw = request.headers.get("x-burst-size")
    if not raw:
        return
    try:
        n = int(raw)
    except ValueError:
        return
    if n <= 0:
        return
    sched = globals().get("_scheduler")
    cfg = globals().get("_config")
    if sched is None or not hasattr(sched, "burst_hints"):
        return
    max_skips = cfg.scheduler.max_skips if cfg is not None else 3
    sched.burst_hints.record(program_id, model, n, max_skips)


async def _is_known_model(model: str) -> bool:
    """Check if `model` is installed in Ollama (defensive against missing registry).

    Wraps `_registry.is_known_model(...)` with a fallback when the
    registry isn't initialized (test paths that bypass lifespan). In
    that case we fail open — let the request through and let the
    scheduler handle the missing-model error normally.
    """
    registry = globals().get("_registry")
    if registry is None or not hasattr(registry, "is_known_model"):
        return True
    result: bool = await registry.is_known_model(model)
    return result


# Power-of-2 boundaries used by the prompt-driven num_ctx sizer.
# Rounding UP to one of these absorbs the char/4 + 20% tokenizer
# overestimate cheaply and keeps Ollama's KV cache slot allocation on
# friendly sizes. Last entry caps below typical model maxes (262144).
_NUM_CTX_BOUNDARIES: tuple[int, ...] = (
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)


def _estimate_prompt_tokens(body: dict[str, Any]) -> int:
    """Cheap upper-bound token estimate from request body.

    char-count / 4 + 20% buffer over-estimates by 5-15% for English/code
    and more for CJK; the round-up to power-of-2 boundary absorbs that
    slack. Avoids the dependency on `tiktoken` or per-model tokenizers.

    Reads from the three shapes Ollama accepts:
    - `messages: [{role, content}]` — /api/chat, /v1/chat/completions
    - `prompt: "..."`               — /api/generate, /v1/completions
    - falls through to 0 if neither is present.

    Returns:
        Token count (int).
    """
    chars = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    chars += len(content)
                elif isinstance(content, list):
                    # OpenAI-style multimodal content: list of parts.
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text", "")
                            if isinstance(text, str):
                                chars += len(text)
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        chars += len(prompt)
    # 1 token ≈ 4 chars + 20% buffer = chars / 4 * 1.2 = chars * 0.3
    return int(chars * 0.3)


def _round_up_to_boundary(value: int) -> int:
    """Round `value` up to the next entry in `_NUM_CTX_BOUNDARIES`.

    Below the smallest boundary returns the smallest. Above the largest
    returns the largest (caller is expected to clamp to the model max).
    """
    for boundary in _NUM_CTX_BOUNDARIES:
        if value <= boundary:
            return boundary
    return _NUM_CTX_BOUNDARIES[-1]


def _resolve_num_ctx_decision(
    *,
    prompt_tokens: int,
    program_id: str,
    model_max_context: int,
    config: Any,
) -> tuple[int, str]:
    """Compute the per-request num_ctx and the mode that produced it.

    Decision tree:
    1. Estimate need: `prompt_tokens + default_completion_budget +
       safety_buffer_tokens`.
    2. Round UP to the next power-of-2 boundary.
    3. If a `ContextConfig.programs[program_id]` profile exists, clamp
       to `[typical_num_ctx, max_num_ctx]`. The floor protects
       tool-calling programs from a reload-stall on round 2; the
       ceiling protects against runaway prompts.
    4. Final clamp: `min(value, model_max_context)`.

    Returns:
        (num_ctx, mode) — `mode` is one of "prompt_driven",
        "program_floor", "program_ceiling", "model_max" — for logging.
    """
    ctx_cfg = config.context
    needed = (
        prompt_tokens + ctx_cfg.default_completion_budget + ctx_cfg.safety_buffer_tokens
    )
    rounded = _round_up_to_boundary(needed)
    chosen = rounded
    mode = "prompt_driven"

    profile = ctx_cfg.programs.get(program_id)
    if profile is not None:
        if chosen < profile.typical_num_ctx:
            chosen = profile.typical_num_ctx
            mode = "program_floor"
        elif chosen > profile.max_num_ctx:
            chosen = profile.max_num_ctx
            mode = "program_ceiling"

    if chosen > model_max_context:
        chosen = model_max_context
        mode = "model_max"

    return chosen, mode


async def _inject_num_ctx(
    model: str, body: dict[str, Any], program_id: str = "default"
) -> None:
    """Inject `options.num_ctx` sized to actual prompt + program profile.

    v0.4.0 behavior (replaces v0.3.0's force-to-max):
    - Estimate prompt tokens from request body (chars/4 + 20% buffer).
    - Add completion budget + safety; round UP to next power-of-2.
    - Clamp to program profile's [typical_num_ctx, max_num_ctx] if set.
    - Final-clamp to model.max_context_length.

    The result is the smallest num_ctx that still fits the actual prompt,
    so Ollama doesn't pre-allocate KV cache for the full architectural
    window on every request. v0.4.0 reload-on-need (Dim 4) handles the
    case where a request actually NEEDS more context than the model
    currently has allocated.

    Two paths:

    1. **Trust-boundary clamp** runs UNCONDITIONALLY when registry
       metadata is available. Even when prompt-driven injection is
       disabled (`context.injection_enabled: false`), an adversarial
       client-supplied `options.num_ctx: 999_999_999` is still clamped
       to the model's max. Skipping this would let one bad request
       trigger reload-on-need, fail preload, infinite-loop the
       scheduler, and unboundedly grow `metrics.reload_count` — i.e.
       the same DoS the v0.4.0 clamp was meant to prevent.
    2. **Prompt-driven injection** only runs when
       `context.injection_enabled: true` AND no client value is set.

    Skipped entirely when:
    - Model name is empty.
    - `options` is set to a non-dict value.
    - Registry metadata isn't available (model unknown, Ollama down).
      Without metadata we don't know the model's max, so neither clamp
      nor injection can run.
    """
    if not model:
        return
    options = body.setdefault("options", {})
    if not isinstance(options, dict):
        return

    registry = globals().get("_registry")
    if registry is None:
        return
    meta = await registry.probe_metadata(model)
    if meta is None:
        return

    # Trust-boundary clamp — runs regardless of injection_enabled.
    # An adversarial or buggy client sending num_ctx: 999_999_999
    # must always be clamped, even when the operator has opted out
    # of prompt-driven injection.
    client_value = options.get("num_ctx")
    if client_value is not None:
        if not isinstance(client_value, int) or client_value <= 0:
            # Drop a malformed client value. If injection is enabled,
            # we'll fall through to prompt-driven sizing below; if
            # disabled, the request just goes out without a num_ctx
            # (Ollama default behavior).
            options.pop("num_ctx", None)
        else:
            clamped = min(client_value, meta.max_context_length)
            if clamped != client_value:
                logger.info(
                    "server.num_ctx_clamped",
                    model=model,
                    program=program_id,
                    requested=client_value,
                    clamped_to=clamped,
                )
            options["num_ctx"] = clamped
            # Client value (clamped) wins — skip prompt-driven sizing.
            return

    # Prompt-driven sizing — only runs when injection is enabled.
    config = globals().get("_config")
    if config is None or not config.context.injection_enabled:
        return

    prompt_tokens = _estimate_prompt_tokens(body)
    chosen, mode = _resolve_num_ctx_decision(
        prompt_tokens=prompt_tokens,
        program_id=program_id,
        model_max_context=meta.max_context_length,
        config=config,
    )
    options["num_ctx"] = chosen
    logger.debug(
        "server.num_ctx_decision",
        model=model,
        program=program_id,
        prompt_tokens=prompt_tokens,
        completion_budget=config.context.default_completion_budget,
        chosen=chosen,
        mode=mode,
        model_max=meta.max_context_length,
    )


def _parse_retry_max_header(request: Request) -> int | None:
    """Parse `X-Marshal-Retry-Max` header into a per-request retry budget.

    Returns None when the header is absent or malformed (use config
    default). Returns 0 when the client explicitly disables retry. Caps
    at `retry.max_per_request_attempts` to keep an adversarial header
    from arbitrarily extending the retry budget.

    Returns:
        None to defer to config.retry, or an int >= 0 (clamped to
        `retry.max_per_request_attempts`, default 10).
    """
    hdr = request.headers.get("x-marshal-retry-max")
    if hdr is None:
        return None
    try:
        v = int(hdr)
    except ValueError:
        return None
    if v < 0:
        return 0
    # Read the cap from config, with a defensive default for test paths
    # that bypass lifespan (no _config global set).
    config = globals().get("_config")
    cap = config.retry.max_per_request_attempts if config is not None else 10
    return min(v, cap)


def _resolve_forward_timeout(request: Request) -> int:
    """Resolve the marshal→Ollama (Hop 2) wall-clock budget for this request.

    Per-request override via `X-Request-Timeout: <seconds>` header wins;
    otherwise fall back to `scheduler.ollama_forward_timeout_s` from
    marshal.yaml. Defaults to 3600s (1h) if no config is reachable
    (defensive — only happens in test paths that bypass lifespan).

    Semantics changed in v0.6.4: previously this header gated the
    client→marshal wait time (Hop 1, now unbounded). It now sizes the
    marshal→Ollama HTTP call. See CHANGELOG for migration notes.
    """
    hdr = request.headers.get("x-request-timeout")
    if hdr:
        try:
            v = int(hdr)
            if v > 0:
                return v
        except ValueError:
            pass
    cfg = getattr(request.app.state, "config", None)
    if cfg is not None:
        return int(cfg.scheduler.ollama_forward_timeout_s)
    return 3600


def _http_status_for_error(exc: BaseException) -> int:
    """Map a forward-side exception class to an HTTP status code.

    The granularity matters because operators triage by status: a 504
    ("Ollama took too long") points at model size or num_ctx; a 503
    ("can't reach Ollama" or "couldn't load model") points at process
    or VRAM state; a 502 (default) points at protocol or unknown.
    Pre-v0.6.5 every failure surfaced as 502.

    The httpx parent classes ``TimeoutException`` and ``NetworkError``
    are matched (not just ``ReadTimeout`` / ``ConnectError``) so the
    less common siblings — ``ConnectTimeout``, ``WriteTimeout``,
    ``PoolTimeout``, ``ReadError``, ``WriteError`` — get the right
    code instead of silently falling through to 502.
    """
    if isinstance(exc, httpx.TimeoutException):
        return 504
    if isinstance(exc, httpx.NetworkError | PreloadFailedError):
        return 503
    return 502


def _build_error_response(
    envelope: RequestEnvelope,
    model: str,
    *,
    openai_compat: bool,
) -> JSONResponse:
    """Build the JSONResponse for a failed envelope with consistent shape.

    Both Ollama-native and OpenAI-compat paths derive ``error_type``
    from the exception class and ``status`` from
    ``_http_status_for_error``. The body shape switches: Ollama-native
    returns a flat ``{error, error_type, model}`` and OpenAI-compat
    preserves the spec envelope ``{error: {message, type, code}}``
    with ``type`` pinned to the slug ``"proxy_error"`` (matching
    OpenAI's ``invalid_request_error`` / ``rate_limit_error``
    convention) and the actual exception class name in
    ``error.exception_type`` so clients matching on the slug stay
    backwards-compatible.
    """
    err = envelope.error
    err_type = type(err).__name__ if err is not None else "UnknownError"
    err_msg = (str(err) if err is not None else "") or err_type
    status = _http_status_for_error(err) if err is not None else 502
    logger.error(
        "server.request_error",
        model=model,
        error=err_msg,
        error_type=err_type,
        status=status,
    )
    if openai_compat:
        return JSONResponse(
            {
                "error": {
                    "message": err_msg,
                    "type": "proxy_error",
                    "code": "proxy_error",
                    "exception_type": err_type,
                }
            },
            status_code=status,
        )
    return JSONResponse(
        {"error": err_msg, "error_type": err_type, "model": model},
        status_code=status,
    )


async def _enqueue_inference(
    request: Request,
    body: dict[str, Any],
    endpoint: str,
) -> Response:
    """Enqueue an Ollama-native inference request and wait for the result.

    Args:
        request: The incoming FastAPI request.
        body: Parsed request body.
        endpoint: The Ollama endpoint path.

    Returns:
        The proxied response from Ollama.
    """
    model = body.get("model", "")
    stream = body.get("stream", False)
    program_id = _normalize_program_id(request.headers.get("x-program-id"))
    forward_timeout_s = _resolve_forward_timeout(request)

    # Fail-fast on unknown models. Without this, marshal would let the
    # request sit in the queue indefinitely while lifecycle.preload
    # retries trying to load a model Ollama doesn't have. Better to
    # return 404 in milliseconds so the client gets a clear, fast error.
    if model and not await _is_known_model(model):
        logger.warning(
            "server.request_rejected_unknown_model",
            model=model,
            program=program_id,
            endpoint=endpoint,
        )
        return JSONResponse(
            {
                "error": (
                    f"Model {model!r} is not installed in Ollama. "
                    f"Run `ollama pull {model}` or check the model name."
                )
            },
            status_code=404,
        )

    # Capture optional X-Burst-Size hint so the eviction scorer protects
    # this model across the rest of the burst even if the client submits
    # the remaining calls sequentially.
    _record_burst_hint(request, program_id, model)

    # Size num_ctx to actual prompt need + program profile (Dim 1).
    # Skip embeddings — they don't generate, so KV cache pre-allocation
    # is irrelevant.
    if endpoint not in ("/api/embeddings", "/v1/embeddings"):
        await _inject_num_ctx(model, body, program_id)

    envelope = RequestEnvelope(
        model=model,
        program_id=program_id,
        request_body=body,
        endpoint=endpoint,
        stream=stream,
        retry_max_override=_parse_retry_max_header(request),
        bypass_pause=_is_bypass_pause(request),
        ollama_forward_timeout_s=forward_timeout_s,
    )

    # Defensive: a client sending `options: null` (the JSON literal, not
    # an absent key) returns None from .get("options", {}) — chaining
    # .get on None would crash. Coerce to dict before reading num_ctx.
    options = body.get("options") or {}
    logger.info(
        "server.request_enqueued",
        model=model,
        program=program_id,
        endpoint=endpoint,
        stream=stream,
        forward_timeout_s=forward_timeout_s,
        num_ctx=options.get("num_ctx") if isinstance(options, dict) else None,
    )

    await _queues.enqueue(envelope)
    # Hop 1 (client→marshal wait) is unbounded as of v0.6.4. Async
    # clients wait until the response or an error arrives. Hop 2
    # (marshal→Ollama) is bounded by ``ollama_forward_timeout_s`` on
    # the envelope.
    await envelope.done_event.wait()

    if envelope.error:
        return _build_error_response(envelope, model, openai_compat=False)

    response = envelope.response
    if stream and hasattr(response, "__aiter__"):
        return StreamingResponse(
            response,
            media_type="application/x-ndjson",
        )

    # Non-streaming httpx.Response
    if hasattr(response, "status_code"):
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    return JSONResponse(response)


async def _enqueue_and_wait(
    request: Request,
    model: str,
    ollama_body: dict[str, Any],
    endpoint: str,
    stream: bool,
) -> dict[str, Any] | Response:
    """Enqueue a translated request and return the Ollama response data.

    For OpenAI-compat endpoints, we need the parsed response dict
    (not the raw httpx.Response) so we can translate it.

    Args:
        request: The incoming FastAPI request.
        model: Extracted model name.
        ollama_body: Translated Ollama request body.
        endpoint: The Ollama endpoint to forward to.
        stream: Whether streaming is requested.

    Returns:
        Parsed response dict, or a Response for streaming/errors.
    """
    program_id = _normalize_program_id(request.headers.get("x-program-id"))
    forward_timeout_s = _resolve_forward_timeout(request)

    # Same fail-fast unknown-model check as _enqueue_inference. OpenAI
    # clients are especially likely to mis-name models since they
    # share name conventions with the OpenAI catalog (e.g. "gpt-4")
    # rather than Ollama's local-model names.
    if model and not await _is_known_model(model):
        logger.warning(
            "server.request_rejected_unknown_model",
            model=model,
            program=program_id,
            endpoint=endpoint,
        )
        return JSONResponse(
            {
                "error": {
                    "message": (
                        f"Model {model!r} is not installed in Ollama. "
                        f"Run `ollama pull {model}` or check the model name."
                    ),
                    "type": "model_not_found",
                    "code": "model_not_found",
                }
            },
            status_code=404,
        )

    # Capture optional burst-size hint (same semantics as the Ollama-
    # native path).
    _record_burst_hint(request, program_id, model)

    # Same num_ctx injection as the Ollama-native path; OpenAI clients
    # rarely set num_ctx explicitly so they're the most likely to be
    # bitten by Ollama's silent context truncation. Skip embeddings.
    if endpoint not in ("/api/embeddings", "/v1/embeddings"):
        await _inject_num_ctx(model, ollama_body, program_id)

    envelope = RequestEnvelope(
        model=model,
        program_id=program_id,
        request_body=ollama_body,
        endpoint=endpoint,
        stream=stream,
        retry_max_override=_parse_retry_max_header(request),
        bypass_pause=_is_bypass_pause(request),
        ollama_forward_timeout_s=forward_timeout_s,
    )

    await _queues.enqueue(envelope)
    # Hop 1 (client→marshal wait) is unbounded as of v0.6.4. The Hop 2
    # forward call is bounded by ``ollama_forward_timeout_s`` on the
    # envelope and surfaces as a forward-side error if exceeded.
    await envelope.done_event.wait()

    if envelope.error:
        return _build_error_response(envelope, model, openai_compat=True)

    response = envelope.response
    if stream and hasattr(response, "__aiter__"):
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )

    # Extract JSON from httpx.Response
    if hasattr(response, "json"):
        return response.json()  # type: ignore[no-any-return]

    return response  # type: ignore[no-any-return]
