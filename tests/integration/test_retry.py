"""Surface A integration tests — retry on transient Ollama failures.

Uses the fault-injection proxy to simulate Ollama failures without
restarting the real daemon. Tests:
- ConnectError on first attempt → retry succeeds
- 503 exhausted across all retries → retries_succeeded does NOT bump
  (the metric-honesty regression caught by /review on PR #6)
- ``X-Marshal-Retry-Max: 0`` header disables retry per-request
"""

from __future__ import annotations

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


def _build_marshal_config(
    *,
    proxy_url: str,
    tmp_paths: dict,
    max_attempts: int = 3,
    read_timeouts: bool = True,
) -> MarshalConfig:
    """Build a MarshalConfig pointed at the fault proxy.

    ``read_timeouts=True`` by default for retry tests because the
    fault proxy's ``disconnect_next`` surfaces as
    ``httpx.RemoteProtocolError`` (server disconnected mid-response)
    on the marshal side, which is in UNSAFE_RETRY_EXCEPTIONS. The
    real-world parallel: an Ollama daemon recycle mid-flight produces
    the same error class, and operators running into it would set
    this flag too. Production default stays False.
    """
    return MarshalConfig(
        ollama=OllamaConfig(host=proxy_url),
        proxy=ProxyConfig(host="127.0.0.1", port=11436),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
            ollama_forward_timeout_s=60,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(mode=ShutdownMode.IMMEDIATE, unload_models=False),
        retry=RetryConfig(
            enabled=True,
            max_attempts=max_attempts,
            base_delay_s=0.05,  # tight backoff for fast tests
            max_delay_s=0.5,
            read_timeouts=read_timeouts,
        ),
        audit=AuditConfig(enabled=False, path=str(tmp_paths["audit_path"])),
    )


async def test_connect_error_retry_succeeds(tmp_marshal_paths):
    """First attempt disconnects → retry succeeds → metric counts the retry.

    Verifies retries_attempted and retries_succeeded both bump together
    when retry actually saves the request.
    """
    async with fault_proxy() as proxy:
        # First /api/chat hit disconnects (ConnectError on client side);
        # second pass-through succeeds.
        proxy.disconnect_next("/api/chat", times=1)
        cfg = _build_marshal_config(
            proxy_url=proxy.url, tmp_paths=tmp_marshal_paths, max_attempts=3
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
            resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=120)
            assert resp.status_code == 200, resp.text
            metrics = app.state._marshal_internals.scheduler.metrics
            # At least 1 retry attempted (the disconnect). Could be more
            # if Ollama itself glitched, but ≥1 is the assertion.
            assert metrics.retries_attempted >= 1, (
                f"expected ≥1 retry attempted, got {metrics.retries_attempted}"
            )
            assert metrics.retries_succeeded >= 1, (
                f"expected ≥1 retry succeeded, got {metrics.retries_succeeded}"
            )


async def test_503_exhausted_not_counted_as_success(tmp_marshal_paths):
    """503 across all retry attempts → caller gets 503; retries_succeeded stays 0.

    The metric-honesty regression caught by local /review on PR #6:
    when call_with_retry exhausts on a retryable status code, marshal
    returns the failed response (not an exception) but it must NOT
    bump retries_succeeded — operators reading "100 retries attempted,
    100 succeeded" would think Ollama is healthy when actually every
    retry exhausted and the client got 503.
    """
    async with fault_proxy() as proxy:
        # Fail every /api/chat with 503. With max_attempts=3, marshal
        # tries 3 times then returns the last 503 to the client.
        proxy.fail_next("/api/chat", times=99, status=503)
        cfg = _build_marshal_config(
            proxy_url=proxy.url, tmp_paths=tmp_marshal_paths, max_attempts=3
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
            resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=120)
            # Marshal forwards the 503 to the client unchanged.
            assert resp.status_code == 503
            metrics = app.state._marshal_internals.scheduler.metrics
            # 2 retries attempted (max_attempts - 1).
            assert metrics.retries_attempted == 2, (
                f"expected 2 retries attempted, got {metrics.retries_attempted}"
            )
            # CRUCIAL: retries_succeeded stays 0. The metric must
            # distinguish retry-saved-us from retry-exhausted-on-failure.
            assert metrics.retries_succeeded == 0, (
                f"retries_succeeded={metrics.retries_succeeded} but request "
                f"actually returned 503 — metric is dishonest"
            )


async def test_x_marshal_retry_max_zero_disables_retry(tmp_marshal_paths):
    """``X-Marshal-Retry-Max: 0`` header → no retries even on transient failure.

    Per-request opt-out. Tool-calling agents that prefer fail-fast
    can disable retry on a single call without changing config.

    Uses 503 (a SAFE retryable status) so without the header the
    request WOULD have retried; with the header set to 0, it must NOT.
    """
    async with fault_proxy() as proxy:
        # 503 across all attempts. Without retry override, marshal
        # would try 3 times. With X-Marshal-Retry-Max: 0, marshal must
        # try ONCE and surface the 503 to the client.
        proxy.fail_next("/api/chat", times=99, status=503)
        cfg = _build_marshal_config(
            proxy_url=proxy.url, tmp_paths=tmp_marshal_paths, max_attempts=3
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
            headers = {**_HDR, "X-Marshal-Retry-Max": "0"}
            resp = await client.post(
                "/api/chat", json=body, headers=headers, timeout=10
            )
            # Marshal forwards the 503 with no retry.
            assert resp.status_code == 503, (
                f"expected 503 surfaced unchanged; got {resp.status_code}: {resp.text}"
            )
            metrics = app.state._marshal_internals.scheduler.metrics
            # No retries attempted — header overrode the config default.
            assert metrics.retries_attempted == 0, (
                f"expected 0 retries with header override, "
                f"got {metrics.retries_attempted}"
            )
