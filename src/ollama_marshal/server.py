"""FastAPI application wiring all components together."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from ollama_marshal.queue import ModelQueues, RequestEnvelope
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

    # Initialize components
    _queues = ModelQueues()
    _memory = MemoryManager(_config)
    _registry = ModelRegistry(
        ollama_host=_config.ollama.host,
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

    # Start background tasks
    await _memory.start_polling()
    await _registry.initialize()
    # Store the task reference per CLAUDE.md async correctness rule —
    # without this, Python may garbage-collect the task before it runs
    # to completion (and emit a warning on 3.12+).
    benchmark_task = asyncio.create_task(_registry.benchmark_unknown())
    await _scheduler.start()

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

    # Stop the background persister cleanly before final save below.
    metrics_persister.cancel()
    try:
        await metrics_persister
    except asyncio.CancelledError:
        pass
    # Cancel the benchmark task too — if it's still running on shutdown,
    # we don't want to wait for it to finish probing every model.
    if not benchmark_task.done():
        benchmark_task.cancel()
        try:
            await benchmark_task
        except asyncio.CancelledError:
            pass

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
                }
                for m in loaded.values()
            ],
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


async def _inject_num_ctx(model: str, body: dict[str, Any]) -> None:
    """Inject `options.num_ctx = model_max_context` if the client didn't set one.

    Without this, Ollama uses its server-side default (2048) and may
    *silently truncate* a model's effective context window down from its
    architectural max. That's a correctness bug for any client that
    expects the model's full context to be available — analyses get
    quietly worse with no error or warning.

    Behavior:
    - Probes registry metadata if not yet cached (one-shot `/api/show`,
      ~10ms locally on first sight; cached forever after).
    - If the client already set `options.num_ctx`, we don't override it.
    - If metadata can't be resolved (model unknown, Ollama down), we
      skip injection and fall back to current behavior.
    """
    if not model:
        return
    options = body.setdefault("options", {})
    if not isinstance(options, dict):
        # Client sent something weird as `options`; don't touch it.
        return
    if "num_ctx" in options:
        # Client knows what it wants. Respect it.
        return
    # Registry may not yet be initialized (e.g. in unit tests that don't
    # exercise the full lifespan). Bail silently in that case.
    registry = globals().get("_registry")
    if registry is None:
        return
    # probe_metadata is idempotent — returns the cached entry if present.
    meta = await registry.probe_metadata(model)
    if meta is None:
        return
    options["num_ctx"] = meta.max_context_length


def _resolve_timeout(request: Request) -> int:
    """Resolve the wait-for-scheduler timeout for this request.

    Per-request override via `X-Request-Timeout: <seconds>` header wins;
    otherwise fall back to `proxy.request_timeout_s` from marshal.yaml.
    Defaults to 3600s (1h) if no config is reachable (defensive).
    """
    hdr = request.headers.get("x-request-timeout")
    if hdr:
        try:
            v = int(hdr)
            if v > 0:
                return v
        except ValueError:
            pass
    # Read from app.state (set in create_app), with a sane default if missing.
    cfg = getattr(request.app.state, "config", None)
    if cfg is not None:
        return int(cfg.proxy.request_timeout_s)
    return 3600


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
    program_id = request.headers.get("x-program-id", "default")
    timeout_s = _resolve_timeout(request)

    # Capture optional X-Burst-Size hint so the eviction scorer protects
    # this model across the rest of the burst even if the client submits
    # the remaining calls sequentially.
    _record_burst_hint(request, program_id, model)

    # Stop Ollama from silently shrinking num_ctx to fit its slot budget.
    # Skip embeddings — they use input_length not generation context, so
    # forcing model_max_context wastes KV cache without preventing
    # truncation (embedding workloads aren't bitten by the bug this fix
    # addresses).
    if endpoint not in ("/api/embeddings", "/v1/embeddings"):
        await _inject_num_ctx(model, body)

    envelope = RequestEnvelope(
        model=model,
        program_id=program_id,
        request_body=body,
        endpoint=endpoint,
        stream=stream,
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
        timeout_s=timeout_s,
        num_ctx=options.get("num_ctx") if isinstance(options, dict) else None,
    )

    await _queues.enqueue(envelope)
    try:
        await asyncio.wait_for(envelope.done_event.wait(), timeout=timeout_s)
    except TimeoutError:
        logger.error(
            "server.request_timeout",
            model=model,
            program=program_id,
            wait_s=timeout_s,
        )
        return JSONResponse(
            {"error": "Request timed out waiting for model scheduling"},
            status_code=504,
        )

    if envelope.error:
        logger.error(
            "server.request_error",
            model=model,
            error=str(envelope.error),
        )
        return JSONResponse(
            {"error": "Internal proxy error"},
            status_code=502,
        )

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
    program_id = request.headers.get("x-program-id", "default")
    timeout_s = _resolve_timeout(request)

    # Capture optional burst-size hint (same semantics as the Ollama-
    # native path).
    _record_burst_hint(request, program_id, model)

    # Same num_ctx injection as the Ollama-native path; OpenAI clients
    # rarely set num_ctx explicitly so they're the most likely to be
    # bitten by Ollama's silent context truncation. Skip embeddings.
    if endpoint not in ("/api/embeddings", "/v1/embeddings"):
        await _inject_num_ctx(model, ollama_body)

    envelope = RequestEnvelope(
        model=model,
        program_id=program_id,
        request_body=ollama_body,
        endpoint=endpoint,
        stream=stream,
    )

    await _queues.enqueue(envelope)
    try:
        await asyncio.wait_for(envelope.done_event.wait(), timeout=timeout_s)
    except TimeoutError:
        return JSONResponse(
            {"error": {"message": "Request timed out", "type": "timeout"}},
            status_code=504,
        )

    if envelope.error:
        return JSONResponse(
            {"error": {"message": "Internal proxy error", "type": "proxy_error"}},
            status_code=502,
        )

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
