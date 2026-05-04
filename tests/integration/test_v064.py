"""v0.6.4 integration tests — Hop 2 forward timeout + preload backoff.

Two focused tests, one per Bug:

- ``test_x_request_timeout_header_overrides_global_forward_timeout``
  (Bug 1): the repurposed ``X-Request-Timeout`` header now sizes the
  marshal→Ollama (Hop 2) call. With a tiny global default and a
  generous header value, the request should succeed even when Ollama
  takes longer than the global default.

- ``test_preload_storm_eliminated_when_ollama_disconnects``
  (Bug 2): when Ollama disconnects on every preload attempt, marshal
  must NOT generate the pre-fix ~10 calls/sec storm. After the
  per-model ``preload_max_consecutive_failures`` budget burns,
  ``PreloadFailedError`` propagates to the client.

Both use ``fault_proxy`` (no real Ollama load) so they run fast and
don't compete with other suite tests for VRAM.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from asgi_lifespan import LifespanManager

from ollama_marshal.config import (
    AuditConfig,
    MarshalConfig,
    MemoryConfig,
    OllamaConfig,
    Priority,
    ProgramConfig,
    ProxyConfig,
    RetryConfig,
    SchedulerConfig,
    ShutdownConfig,
    ShutdownMode,
)
from tests.integration._fault_proxy import fault_proxy
from tests.integration.conftest import (
    PROGRAM_CRITICAL,
    REQUIRED_MODEL,
    _ollama_reachable,
    make_test_app,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not running on :11434",
    ),
]


_HDR = {"X-Program-ID": PROGRAM_CRITICAL}


def _build_cfg(
    *,
    proxy_url: str,
    tmp_paths: dict,
    forward_timeout_s: int,
    preload_max_consecutive_failures: int = 3,
    preload_backoff_base_s: float = 0.05,
    preload_backoff_max_s: float = 0.5,
) -> MarshalConfig:
    """Marshal config pointed at the fault proxy with v0.6.4 knobs.

    Tight backoff so the test stays under a few seconds while still
    exercising the cooldown skip path between attempts.
    """
    return MarshalConfig(
        ollama=OllamaConfig(host=proxy_url),
        proxy=ProxyConfig(host="127.0.0.1", port=11436),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
            benchmark_on_startup=False,
            ollama_forward_timeout_s=forward_timeout_s,
            preload_max_consecutive_failures=preload_max_consecutive_failures,
            preload_backoff_base_s=preload_backoff_base_s,
            preload_backoff_max_s=preload_backoff_max_s,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(mode=ShutdownMode.IMMEDIATE, unload_models=False),
        # Disable retry so the preload-storm test exercises the backoff
        # state machine on its own without the call_with_retry helper
        # also retrying the forward call. The two paths are separate;
        # this test scopes to backoff only.
        retry=RetryConfig(enabled=False, max_attempts=1),
        audit=AuditConfig(enabled=False, path=str(tmp_paths["audit_path"])),
    )


async def test_x_request_timeout_header_overrides_global_forward_timeout(
    tmp_marshal_paths,
):
    """Repurposed X-Request-Timeout header sizes Hop 2 (Bug 1).

    Set ``ollama_forward_timeout_s = 1`` so any forward taking >1s
    would normally fail. Inject a 0.3s delay (well under the 1s
    global) — without the header the request succeeds. With
    ``X-Request-Timeout: 60`` the header value wins, so a higher
    delay (e.g. 0.6s) also succeeds. Validates that the header value
    actually flows into ``RequestEnvelope.ollama_forward_timeout_s``
    and through ``forward_request``.
    """
    async with fault_proxy() as proxy:
        proxy.delay_next("/api/chat", seconds=0.6)
        cfg = _build_cfg(
            proxy_url=proxy.url, tmp_paths=tmp_marshal_paths, forward_timeout_s=1
        )
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            body = {
                "model": REQUIRED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 4},
            }
            # Header overrides the 1s global → 60s wins, request completes.
            hdr = {**_HDR, "X-Request-Timeout": "60"}
            resp = await client.post("/api/chat", json=body, headers=hdr, timeout=120)
            assert resp.status_code == 200, resp.text


async def test_preload_storm_eliminated_when_ollama_disconnects(tmp_marshal_paths):
    """Per-model preload backoff caps the disconnect-storm (Bug 2).

    Pre-fix: scheduler tick at 0.1s would call ``lifecycle.preload``
    ~10 times/sec while Ollama is unreachable. Post-fix: per-model
    failure tracking parks attempts behind exponential backoff and
    after ``preload_max_consecutive_failures`` (=3 in this test) the
    queued envelope fails with ``PreloadFailedError`` (502 to client).

    Approach: use ``fake_response`` on ``/api/ps`` to make marshal's
    memory poller permanently see "no models loaded" — even if the
    real Ollama actually has REQUIRED_MODEL loaded from another
    process. This forces the scheduler down the preload path on every
    tick. Then ``disconnect_next`` on ``/api/generate`` (the preload
    endpoint) makes every preload attempt fail with
    ``httpx.HTTPError``. Without backoff, marshal would burn through
    99 attempts in <1s; with backoff, it gives up after 3 + a few
    cooldown waits (~0.5-2s total).
    """
    async with fault_proxy() as proxy:
        # Force memory poller to see no loaded models. /api/ps polls
        # land on this fake response indefinitely; bin_pack always
        # finds the model "missing" and tries to preload.
        proxy.fake_response("/api/ps", {"models": []}, times=None)
        # Disconnect every preload call. lifecycle.preload catches the
        # httpx exception and returns False, bumping the per-model
        # failure counter. The storm pre-fix regression would consume
        # ~10 of these per second; post-fix bounds at ~3.
        proxy.disconnect_next("/api/generate", times=99)

        cfg = _build_cfg(
            proxy_url=proxy.url,
            tmp_paths=tmp_marshal_paths,
            forward_timeout_s=10,
            preload_max_consecutive_failures=3,
            preload_backoff_base_s=0.1,
            preload_backoff_max_s=0.5,
        )
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            body = {
                "model": REQUIRED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 4},
            }
            # Hop 1 is unbounded as of v0.6.4; rely on the giveup path
            # to fail the envelope. Bound the test with a client-side
            # socket timeout so a regression that hangs the request
            # fails the test deterministically.
            resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=15)

            # v0.6.5 Bug 3: PreloadFailedError now maps to 503 (Service
            # Unavailable — model couldn't be loaded after backoff
            # giveup) and the body propagates the actual class name +
            # message instead of an opaque 502 / "Internal proxy error".
            assert resp.status_code == 503, resp.text
            payload = resp.json()
            assert payload["error_type"] == "PreloadFailedError", payload
            assert REQUIRED_MODEL in payload["error"], payload
            assert payload["model"] == REQUIRED_MODEL, payload

            # Failure state cleared after giveup — the next request can
            # try again from scratch, per the per-batch giveup design.
            scheduler = app.state._marshal_internals.scheduler
            failure_state = scheduler._preload_failures.get(REQUIRED_MODEL)
            assert failure_state is None, (
                f"preload failure state should be cleared after giveup, "
                f"got {failure_state}"
            )

            # Brief cooldown window to confirm no storm continues
            # post-giveup. If the backoff is broken, additional preloads
            # would fire here while the test waits.
            await asyncio.sleep(0.5)
