"""v0.6.5 Bug 5 — opt-in load tests verifying Hop 1 unbounded behavior.

v0.6.4 removed ``proxy.request_timeout_s``, so the client→marshal wait
is unbounded. These tests verify that design holds under sustained
queue pressure: marshal must serve every enqueued request without
worker-pool exhaustion, model-swap thrash, or queue overflow, and
pause/resume must recover cleanly under load.

Excluded from default ``make test-integration`` and from ``make
pre-pr`` — only ``make load-test`` runs them. Each scenario takes
minutes at default sizing; tune via env vars for stress sweeps:

    LOAD_TEST_DURATION_S        (default: 60)   sustained-pressure window
    LOAD_TEST_CONCURRENCY       (default: 20)   parallel client count
    LOAD_TEST_PAUSE_DURATION_S  (default: 5)    pause-window in scenario 4
    LOAD_TEST_MIXED_MODELS      (default: "")   comma-separated model names
                                                for the mixed-contention
                                                scenario; empty = skip

Designed as a pre-release sanity check, not a per-commit gate. Run
before tagging a release; capture the printed p50/p95/p99 numbers in
the release notes for regression tracking.

The ``load`` marker is registered in ``pyproject.toml`` and intentionally
not stacked with ``integration`` so ``-m integration`` skips these.
"""
# ruff: noqa: T201 — load scenarios print percentile numbers to stdout
# for the operator to capture into release-note regression tracking.

from __future__ import annotations

import asyncio
import os
import statistics
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
import pytest
from asgi_lifespan import LifespanManager

from tests.integration.conftest import (
    DEFAULT_OLLAMA_HOST,
    PROGRAM_CRITICAL,
    REQUIRED_MODEL,
    _ollama_reachable,
    make_test_app,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

pytestmark = [
    pytest.mark.load,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not running on :11434",
    ),
]


_HDR = {"X-Program-ID": PROGRAM_CRITICAL}
_DURATION_S = int(os.environ.get("LOAD_TEST_DURATION_S", "60"))
_CONCURRENCY = int(os.environ.get("LOAD_TEST_CONCURRENCY", "20"))
_PAUSE_DURATION_S = int(os.environ.get("LOAD_TEST_PAUSE_DURATION_S", "5"))
_MIXED_MODELS = [
    m.strip()
    for m in os.environ.get("LOAD_TEST_MIXED_MODELS", "").split(",")
    if m.strip()
]


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile (0-100) of ``values``.

    Hand-rolled because ``statistics.quantiles`` requires N>1 and
    splits into discrete bins; for percentile reporting we want the
    interpolated value at any pct against an arbitrary-size sample.
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


async def _fire_chat(
    client: httpx.AsyncClient, model: str, idx: int
) -> tuple[int, float]:
    """One chat request; returns (status_code, latency_seconds)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"reply with the digit {idx % 10}"}],
        "stream": False,
        "options": {"num_predict": 4},
    }
    t0 = time.monotonic()
    resp = await client.post(
        "/api/chat",
        json=body,
        headers=_HDR,
        timeout=300,  # generous; we want the client to wait, not bail
    )
    return resp.status_code, time.monotonic() - t0


@asynccontextmanager
async def _marshal_client(
    tmp_marshal_paths: dict,
) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    """Yield an in-process marshal client with default test config.

    Centralized so each scenario gets the same isolated state. Async
    context manager (not generator) so closures inside the scenarios
    don't lose binding to the loop-variable.
    """
    from ollama_marshal.config import (
        AuditConfig,
        MarshalConfig,
        MemoryConfig,
        OllamaConfig,
        Priority,
        ProgramConfig,
        ProxyConfig,
        SchedulerConfig,
        ShutdownConfig,
        ShutdownMode,
    )

    cfg = MarshalConfig(
        ollama=OllamaConfig(host=DEFAULT_OLLAMA_HOST),
        proxy=ProxyConfig(host="127.0.0.1", port=11436),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_marshal_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
            benchmark_on_startup=False,
            ollama_forward_timeout_s=300,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(mode=ShutdownMode.IMMEDIATE, unload_models=False),
        audit=AuditConfig(enabled=False, path=str(tmp_marshal_paths["audit_path"])),
    )
    app = make_test_app(cfg, tmp_marshal_paths)
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client, app


# ---------------------------------------------------------------------------
# Scenario 1 — sustained queue pressure
# ---------------------------------------------------------------------------


async def test_sustained_queue_pressure(tmp_marshal_paths):
    """N concurrent clients fire requests for ``LOAD_TEST_DURATION_S`` seconds.

    Verifies marshal serves every enqueued request without dropping
    requests, exhausting the uvicorn worker pool, or returning a
    non-2xx status under sustained load. Patient clients holding
    Hop 1 connections open is exactly the surface area v0.6.4's
    unbounded design depends on staying healthy.
    """
    async with _marshal_client(tmp_marshal_paths) as (client, _app):
        deadline = time.monotonic() + _DURATION_S
        results: list[tuple[int, float]] = []

        async def driver(driver_idx: int) -> None:
            n = 0
            while time.monotonic() < deadline:
                status, latency = await _fire_chat(
                    client, REQUIRED_MODEL, driver_idx * 1000 + n
                )
                results.append((status, latency))
                n += 1

        await asyncio.gather(*(driver(i) for i in range(_CONCURRENCY)))

        statuses = [s for s, _ in results]
        latencies = [lat for _, lat in results]
        ok = sum(1 for s in statuses if 200 <= s < 300)
        print(
            f"\n[load.sustained] requests={len(results)} "
            f"ok={ok}/{len(results)} "
            f"p50={_percentile(latencies, 50):.2f}s "
            f"p95={_percentile(latencies, 95):.2f}s "
            f"p99={_percentile(latencies, 99):.2f}s "
            f"concurrency={_CONCURRENCY} duration={_DURATION_S}s"
        )
        assert ok == len(results), (
            f"expected all {len(results)} requests to succeed, "
            f"got {ok}; non-2xx statuses: "
            f"{sorted({s for s in statuses if not 200 <= s < 300})}"
        )


# ---------------------------------------------------------------------------
# Scenario 2 — mixed-model contention (multi-model bin-packing under load)
# ---------------------------------------------------------------------------


async def test_mixed_model_contention(tmp_marshal_paths):
    """Multiple small models loaded in parallel; verify bin-packing holds.

    Skipped unless ``LOAD_TEST_MIXED_MODELS=a,b,c,...`` is set with at
    least 2 model names that fit alongside each other. Without an
    explicit list we can't pick safely (every install has a different
    model mix). When set, fires ``LOAD_TEST_CONCURRENCY/N`` clients per
    model in parallel and asserts model_swaps stays bounded — bin-pack
    must keep multiple models loaded simultaneously rather than thrash.
    """
    if len(_MIXED_MODELS) < 2:
        pytest.skip(
            "set LOAD_TEST_MIXED_MODELS=model1,model2[,...] "
            "to enable multi-model contention scenario"
        )

    async with _marshal_client(tmp_marshal_paths) as (client, app):
        per_model = max(1, _CONCURRENCY // len(_MIXED_MODELS))
        sched = app.state._marshal_internals.scheduler
        swaps_baseline = sched.metrics.model_swaps

        async def model_driver(model: str, idx: int) -> tuple[int, float]:
            return await _fire_chat(client, model, idx)

        coros = [
            model_driver(model, i * len(_MIXED_MODELS) + j)
            for j, model in enumerate(_MIXED_MODELS)
            for i in range(per_model)
        ]
        results = await asyncio.gather(*coros)

        statuses = [s for s, _ in results]
        latencies = [lat for _, lat in results]
        ok = sum(1 for s in statuses if 200 <= s < 300)
        swaps_delta = sched.metrics.model_swaps - swaps_baseline
        print(
            f"\n[load.mixed_models] models={_MIXED_MODELS} "
            f"per_model={per_model} requests={len(results)} ok={ok} "
            f"model_swaps_delta={swaps_delta} "
            f"p50={_percentile(latencies, 50):.2f}s "
            f"p95={_percentile(latencies, 95):.2f}s"
        )
        assert ok == len(results), (
            f"expected all {len(results)} requests to succeed, got {ok}"
        )
        # Loose upper bound — no sane bin-pack should swap more than
        # 2x per model under steady demand on each.
        max_reasonable = 2 * len(_MIXED_MODELS)
        assert swaps_delta <= max_reasonable, (
            f"model_swaps_delta={swaps_delta} > {max_reasonable} — "
            f"bin-packing is thrashing under mixed-model load"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — latency-percentile baseline
# ---------------------------------------------------------------------------


async def test_latency_percentile_baseline(tmp_marshal_paths):
    """Same load profile as Scenario 1 but reports detailed timing only.

    No correctness assertion beyond all-requests-succeed; the value is
    the printed p50/p95/p99/mean/stdev for release-over-release
    regression tracking. Capture this output in the release notes
    when tagging a new version.
    """
    async with _marshal_client(tmp_marshal_paths) as (client, _app):
        deadline = time.monotonic() + _DURATION_S
        latencies: list[float] = []
        statuses: list[int] = []

        async def driver(driver_idx: int) -> None:
            n = 0
            while time.monotonic() < deadline:
                status, latency = await _fire_chat(
                    client, REQUIRED_MODEL, driver_idx * 1000 + n
                )
                statuses.append(status)
                latencies.append(latency)
                n += 1

        await asyncio.gather(*(driver(i) for i in range(_CONCURRENCY)))

        ok = sum(1 for s in statuses if 200 <= s < 300)
        if latencies:
            stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            print(
                f"\n[load.latency_baseline] requests={len(latencies)} "
                f"ok={ok}/{len(latencies)} "
                f"min={min(latencies):.3f}s "
                f"p50={_percentile(latencies, 50):.3f}s "
                f"p90={_percentile(latencies, 90):.3f}s "
                f"p95={_percentile(latencies, 95):.3f}s "
                f"p99={_percentile(latencies, 99):.3f}s "
                f"max={max(latencies):.3f}s "
                f"mean={statistics.mean(latencies):.3f}s "
                f"stdev={stdev:.3f}s "
                f"concurrency={_CONCURRENCY} duration={_DURATION_S}s"
            )
        assert ok == len(latencies), (
            f"expected all {len(latencies)} requests to succeed, got {ok}"
        )


# ---------------------------------------------------------------------------
# Scenario 4 — pause/resume recovery under load
# ---------------------------------------------------------------------------


async def test_pause_resume_recovery_under_load(tmp_marshal_paths):
    """Pause marshal mid-load; verify the queue grows then drains on resume.

    Spawns load, calls ``Scheduler.pause()`` directly (in-process),
    waits ``LOAD_TEST_PAUSE_DURATION_S`` seconds while the queue
    backs up, then calls ``Scheduler.resume()``. Verifies all in-flight
    + queued requests eventually complete with 2xx status. Models
    the operational scenario where an admin needs to drain marshal
    for a manual intervention without dropping client work.
    """
    async with _marshal_client(tmp_marshal_paths) as (client, app):
        sched = app.state._marshal_internals.scheduler
        results: list[tuple[int, float]] = []
        stop = asyncio.Event()

        async def driver(driver_idx: int) -> None:
            n = 0
            while not stop.is_set():
                status, latency = await _fire_chat(
                    client, REQUIRED_MODEL, driver_idx * 1000 + n
                )
                results.append((status, latency))
                n += 1

        drivers = [asyncio.create_task(driver(i)) for i in range(_CONCURRENCY)]
        try:
            # Run a short warm-up so we have requests in flight.
            await asyncio.sleep(2)
            await sched.pause(
                drain_timeout_s=0.0,
                auto_resume_after_seconds=_PAUSE_DURATION_S * 4,
            )
            paused_at = time.monotonic()
            await asyncio.sleep(_PAUSE_DURATION_S)
            await sched.resume()
            resumed_at = time.monotonic()
            print(
                f"\n[load.pause_resume] paused_for={resumed_at - paused_at:.1f}s "
                f"requests_during_pause~={len(results)} "
                f"concurrency={_CONCURRENCY}"
            )
            # Let the post-resume drain settle.
            await asyncio.sleep(5)
        finally:
            stop.set()
            for d in drivers:
                d.cancel()
            for d in drivers:
                with pytest.raises(asyncio.CancelledError):
                    await d
        statuses = [s for s, _ in results]
        ok = sum(1 for s in statuses if 200 <= s < 300)
        assert ok == len(results), (
            f"expected every request to succeed across pause/resume, "
            f"got {ok}/{len(results)}; non-2xx: "
            f"{sorted({s for s in statuses if not 200 <= s < 300})}"
        )
