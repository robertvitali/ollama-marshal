"""Live-pressure admission integration tests (Memory rework M4).

End-to-end coverage for M2's live-aware admission against real Ollama,
driven through the ``_live_available_override`` injection seam (set the
override, call ``sample_live_available``, and every subsequent poll
re-samples the same override — no psutil involved). Unit tests cover
the EWMA math and the refusal control flow; these tests pin the three
behaviors that only show up across the scheduler ↔ memory ↔ lifecycle
↔ real-Ollama boundary:

1. A NEW load under live pressure is refused WITHOUT ever touching
   Ollama (no preload), routes through the Bug C escape valve to a
   fast 503, and the M3 observability (memory.live block + refusal
   counter + last_refusal) reflects it.
2. Gate-new-only: an ALREADY-LOADED model keeps serving under live
   pressure — no refusal, no eviction.
3. Recovery: when pressure eases, the next request loads and serves
   normally (a transient spike is a retry, not a permanent failure).

Every config uses ``live_memory_ewma_alpha=1.0`` so a single sample
pins the EWMA to the override exactly — the poll loop's background
samples (which also read the override) can't drift it, keeping the
assertions deterministic. Smoothing behavior itself is unit-tested.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from asgi_lifespan import LifespanManager

from ollama_marshal.config import (
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
from tests.integration.conftest import (
    DEFAULT_OLLAMA_HOST,
    INTEGRATION_FORWARD_TIMEOUT_S,
    PROGRAM_CRITICAL,
    REQUIRED_MODEL,
    _ollama_reachable,
    make_test_app,
    wait_for,
)


def _required_model_pulled() -> bool:
    """Sync check that REQUIRED_MODEL is pulled (codex P2).

    Without the model, request entry fail-fasts 404 before admission
    ever runs, so the live-pressure paths would never be exercised.
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{DEFAULT_OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            names = {m.get("name") for m in resp.json().get("models", [])}
            return REQUIRED_MODEL in names
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not running on :11434",
    ),
    pytest.mark.skipif(
        not (_ollama_reachable() and _required_model_pulled()),
        reason=f"{REQUIRED_MODEL} not pulled on :11434",
    ),
]

_GB = 1024**3
_HDR_CRIT = {"X-Program-ID": PROGRAM_CRITICAL}

_CHAT_BODY = {
    "model": REQUIRED_MODEL,
    "messages": [{"role": "user", "content": "hi"}],
    "stream": False,
    "options": {"num_predict": 4},
}


def _live_pressure_config(tmp_marshal_paths) -> MarshalConfig:
    """Shared config: roomy static budget, tight escape valve, alpha=1.

    The static budget (16 GB) comfortably admits REQUIRED_MODEL, so any
    refusal in these tests is attributable to the LIVE term alone.
    """
    return MarshalConfig(
        ollama=OllamaConfig(host=DEFAULT_OLLAMA_HOST),
        proxy=ProxyConfig(host="127.0.0.1", port=11437),
        memory=MemoryConfig(
            total_ram="16GB",
            os_overhead="0B",
            safety_margin="0B",
            poll_interval=1,
            live_memory_enabled=True,
            live_memory_ewma_alpha=1.0,
        ),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_marshal_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
            ollama_forward_timeout_s=INTEGRATION_FORWARD_TIMEOUT_S,
            # Bound the escape valve tight so the 503 lands in well under
            # 10s (same rationale as the Bug C cannot-fit test).
            preload_max_consecutive_failures=2,
            preload_backoff_base_s=0.05,
            preload_backoff_max_s=0.2,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(
            mode=ShutdownMode.IMMEDIATE,
            unload_models=True,
            drain_timeout=10,
        ),
    )


async def _unload_required_model() -> None:
    """Unload REQUIRED_MODEL (only) and wait until /api/ps converges.

    Scoped to the one model these tests touch (codex P3) — unloading
    everything resident is broader than needed on a shared daemon —
    and polled to convergence so pre/postconditions can't race the
    async unload.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            f"{DEFAULT_OLLAMA_HOST}/api/generate",
            json={"model": REQUIRED_MODEL, "prompt": "", "keep_alive": "0"},
        )
    await wait_for(
        lambda: _model_absent_on_ollama(REQUIRED_MODEL),
        timeout=15,
        description=f"{REQUIRED_MODEL} unloaded from /api/ps",
    )


async def _model_absent_on_ollama(model: str) -> bool:
    return not await _model_loaded_on_ollama(model)


async def _model_loaded_on_ollama(model: str) -> bool:
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(f"{DEFAULT_OLLAMA_HOST}/api/ps")
        resp.raise_for_status()
        return any(
            (m.get("name") or m.get("model")) == model
            for m in resp.json().get("models", [])
        )


async def test_live_pressure_refuses_new_load_fast_503(tmp_marshal_paths):
    """A new load under live pressure 503s fast and never touches Ollama.

    The static budget (16 GB) would admit REQUIRED_MODEL; the injected
    live reading (200 MB) is the blocker. The refusal must happen
    BEFORE any preload (Ollama never sees the model) and route through
    the bounded backoff to a 503, and the M3 status observability must
    record what bounced and why.
    """
    await _unload_required_model()

    cfg = _live_pressure_config(tmp_marshal_paths)
    app = make_test_app(cfg, tmp_marshal_paths)
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        memory = app.state._marshal_internals.memory
        memory._live_available_override = 200 * 1024**2
        memory.sample_live_available()

        start = asyncio.get_running_loop().time()
        resp = await client.post(
            "/api/chat", json=_CHAT_BODY, headers=_HDR_CRIT, timeout=30
        )
        elapsed = asyncio.get_running_loop().time() - start

        # Escape valve fired: capacity-class 503, fast, no hang.
        assert resp.status_code == 503, resp.text
        body = resp.json()
        assert body["error_type"] == "PreloadFailedError", body
        assert "out of capacity" in body["error"], body
        assert elapsed < 10, f"503 took {elapsed:.1f}s — escape valve too slow"

        # M3 observability reflects the refusal.
        status = (await client.get("/api/marshal/status")).json()
        live = status["memory"]["live"]
        assert live["enabled"] is True
        # alpha=1.0 pins the EWMA to the override exactly.
        assert live["available"] == 200 * 1024**2
        assert live["headroom"] == 200 * 1024**2  # safety_margin is 0
        assert status["metrics"]["live_pressure_refusals"] >= 1
        assert live["last_refusal"] is not None
        assert live["last_refusal"]["model"] == REQUIRED_MODEL

        # The refusal precedes preload: Ollama never saw a load request.
        # Checked while the app is ALIVE (codex P2) — after teardown,
        # shutdown.unload_models=True would make this assertion pass
        # even if a broken implementation had preloaded — and
        # cross-checked against marshal's own in-process books.
        assert not await _model_loaded_on_ollama(REQUIRED_MODEL)
        assert not memory.is_loaded(REQUIRED_MODEL)


async def test_live_pressure_gate_new_only_loaded_model_keeps_serving(
    tmp_marshal_paths,
):
    """Gate-new-only: live pressure never disturbs an already-loaded model.

    Load REQUIRED_MODEL normally, then inject extreme live pressure.
    A follow-up request against the loaded copy must still serve 200 —
    serving consumes no new memory — with zero refusals recorded and
    the model still resident on Ollama afterward.
    """
    await _unload_required_model()

    cfg = _live_pressure_config(tmp_marshal_paths)
    app = make_test_app(cfg, tmp_marshal_paths)
    transport = httpx.ASGITransport(app=app)
    try:
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            # Normal load (no pressure yet).
            resp = await client.post(
                "/api/chat",
                json=_CHAT_BODY,
                headers=_HDR_CRIT,
                timeout=INTEGRATION_FORWARD_TIMEOUT_S,
            )
            assert resp.status_code == 200, resp.text

            # Extreme live pressure lands AFTER the model is resident.
            memory = app.state._marshal_internals.memory
            memory._live_available_override = 100 * 1024**2
            memory.sample_live_available()

            resp = await client.post(
                "/api/chat",
                json=_CHAT_BODY,
                headers=_HDR_CRIT,
                timeout=INTEGRATION_FORWARD_TIMEOUT_S,
            )
            assert resp.status_code == 200, resp.text

            status = (await client.get("/api/marshal/status")).json()
            assert status["metrics"]["live_pressure_refusals"] == 0
            assert status["memory"]["live"]["last_refusal"] is None

            # Still resident — the pressure dip evicted nothing. Checked
            # INSIDE the lifespan (codex P1): shutdown.unload_models=True
            # unloads the marshal-owned copy at teardown, so a
            # post-lifespan residency check would false-fail.
            assert await _model_loaded_on_ollama(REQUIRED_MODEL)
    finally:
        await _unload_required_model()


async def test_live_pressure_recovery_after_spike(tmp_marshal_paths):
    """A transient live-pressure spike is a retry, not a permanent failure.

    Under pressure the load 503s (giveup drains the queue); once the
    injected reading recovers, a fresh request loads and serves 200.
    """
    await _unload_required_model()

    cfg = _live_pressure_config(tmp_marshal_paths)
    app = make_test_app(cfg, tmp_marshal_paths)
    transport = httpx.ASGITransport(app=app)
    try:
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            memory = app.state._marshal_internals.memory
            memory._live_available_override = 200 * 1024**2
            memory.sample_live_available()

            resp = await client.post(
                "/api/chat", json=_CHAT_BODY, headers=_HDR_CRIT, timeout=30
            )
            assert resp.status_code == 503, resp.text

            # Pressure eases: 32 GB live — both terms now admit.
            memory._live_available_override = 32 * _GB
            memory.sample_live_available()

            resp = await client.post(
                "/api/chat",
                json=_CHAT_BODY,
                headers=_HDR_CRIT,
                timeout=INTEGRATION_FORWARD_TIMEOUT_S,
            )
            assert resp.status_code == 200, resp.text
            assert await _model_loaded_on_ollama(REQUIRED_MODEL)
    finally:
        await _unload_required_model()
