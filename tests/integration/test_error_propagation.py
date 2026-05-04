"""End-to-end coverage for v0.6.5 error-body propagation.

v0.6.5's Bug 3 fix replaced marshal's old generic 502 with a status code
derived from the underlying ``httpx`` exception class (504 for
``TimeoutException``, 503 for ``NetworkError`` / ``PreloadFailedError``,
502 for everything else) and a body that carries the actual exception
class name. The unit suite covers ``_http_status_for_error`` and
``_build_error_response`` in isolation, but until v0.6.6 there was no
integration test asserting that a real wire-level Ollama failure
surfaces with the new shape end-to-end.

Each test injects the failure via ``tests/integration/_fault_proxy``
(which sits between marshal and real Ollama) and asserts the response
status + body. Status mapping verified per surface:

- ``ReadTimeout`` → 504 (Ollama-native flat shape)
- ``ConnectError`` → 503 (Ollama-native flat shape)
- OpenAI-compat envelope on ``/v1/chat/completions`` failure
- ``PreloadFailedError`` → 503 (validates v0.6.4 preload-giveup +
  v0.6.5 error-shape interaction)
- ``RemoteProtocolError`` → 502 (default class for unmapped errors)
"""

from __future__ import annotations

from pathlib import Path

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
from tests.integration._fault_proxy import FaultProxy, fault_proxy
from tests.integration.conftest import (
    DEFAULT_OLLAMA_HOST,
    PROGRAM_CRITICAL,
    REQUIRED_MODEL,
    _ollama_reachable,
    make_test_app,
    wait_for,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not running on :11434",
    ),
]

_HDR = {"X-Program-ID": PROGRAM_CRITICAL}
_CHAT_BODY = {
    "model": REQUIRED_MODEL,
    "messages": [{"role": "user", "content": "hi"}],
    "stream": False,
    "options": {"num_predict": 4},
}


def _build_marshal_config(
    *,
    proxy_url: str,
    tmp_paths: dict[str, Path],
    ollama_forward_timeout_s: int = 60,
    preload_max_consecutive_failures: int = 5,
    preload_backoff_base_s: float = 1.0,
    retry_max_attempts: int = 1,
) -> MarshalConfig:
    """Build a MarshalConfig pointed at the fault proxy.

    Defaults disable retry (``retry_max_attempts=1``) so each test sees
    the underlying exception class on the very first attempt — without
    this, ``ConnectError`` would be retried by the SAFE class set and
    might race with proxy-state changes mid-test. ``retry.enabled``
    stays True so the surrounding ``call_with_retry`` plumbing matches
    the production path; setting attempts to 1 short-circuits the loop.
    """
    return MarshalConfig(
        ollama=OllamaConfig(host=proxy_url),
        proxy=ProxyConfig(host="127.0.0.1", port=11436),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
            benchmark_on_startup=False,
            ollama_forward_timeout_s=ollama_forward_timeout_s,
            preload_max_consecutive_failures=preload_max_consecutive_failures,
            preload_backoff_base_s=preload_backoff_base_s,
            preload_backoff_max_s=0.5,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(
            mode=ShutdownMode.IMMEDIATE,
            drain_timeout=5,
            unload_models=False,
        ),
        retry=RetryConfig(
            enabled=True,
            max_attempts=retry_max_attempts,
            base_delay_s=0.05,
            max_delay_s=0.5,
            read_timeouts=False,
        ),
        audit=AuditConfig(enabled=False, path=str(tmp_paths["audit_path"])),
    )


async def test_native_read_timeout_returns_504(tmp_marshal_paths):
    """Forward call exceeds Hop 2 timeout → 504 + ``error_type: "ReadTimeout"``.

    Proxy delays /api/chat past the per-request ``X-Request-Timeout``
    budget (2s). Preload still uses the config-default 60s budget so
    the upstream load is unaffected. ``read_timeouts=False`` keeps the
    ReadTimeout from being retried, so the very first failure surfaces.
    """
    async with fault_proxy() as proxy:
        proxy.delay_next("/api/chat", seconds=10)
        cfg = _build_marshal_config(proxy_url=proxy.url, tmp_paths=tmp_marshal_paths)
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            resp = await client.post(
                "/api/chat",
                json=_CHAT_BODY,
                headers={**_HDR, "X-Request-Timeout": "2"},
                timeout=30,
            )
    assert resp.status_code == 504, resp.text
    body = resp.json()
    assert body["error_type"] == "ReadTimeout", body
    assert body["model"] == REQUIRED_MODEL, body
    assert body["error"], "expected non-empty error message"


async def test_native_connect_error_returns_503(tmp_marshal_paths):
    """Forward connect refused → 503 + ``error_type: "ConnectError"``.

    Setup is delicate: ``ConnectError`` only surfaces when the upstream
    TCP port is CLOSED at forward time. If we point marshal at a
    permanently-closed port from startup, the scheduler's preload step
    fails first and surfaces as ``PreloadFailedError`` instead. So the
    test stages the failure: preload the model on real Ollama directly,
    let marshal's memory poller observe it via the live fault-proxy
    passthrough (so the scheduler skips preload), then stop the proxy
    so the next /api/chat forward connects to a closed port. The proxy
    is started/stopped manually rather than via ``async with`` so we
    can shut it down mid-test without leaving the lifespan context.
    """
    # Ensure REQUIRED_MODEL is loaded on real Ollama before the test
    # starts so the proxy passthrough surfaces it on /api/ps. Marshal's
    # memory poller picks that up and the scheduler short-circuits
    # preload. Keep-alive is short ("30s") so the model unloads soon
    # after the test — bypassing marshal means marshal can't track this
    # load, so a long keep_alive would leak warm state to subsequent
    # tests for the rest of the integration session.
    async with httpx.AsyncClient(timeout=120) as direct:
        await direct.post(
            f"{DEFAULT_OLLAMA_HOST}/api/generate",
            json={
                "model": REQUIRED_MODEL,
                "prompt": "",
                "keep_alive": "30s",
            },
        )

    proxy = FaultProxy()
    await proxy._start()
    proxy_stopped = False
    try:
        cfg = _build_marshal_config(
            proxy_url=proxy.url,
            tmp_paths=tmp_marshal_paths,
            # Tight forward budget so the ConnectError surfaces fast.
            ollama_forward_timeout_s=5,
        )
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            memory = app.state._marshal_internals.memory
            await wait_for(
                lambda: REQUIRED_MODEL in memory.get_loaded_models(),
                timeout=15,
                description="memory poller to observe REQUIRED_MODEL via proxy",
            )

            await proxy._stop()
            proxy_stopped = True

            resp = await client.post(
                "/api/chat",
                json=_CHAT_BODY,
                headers=_HDR,
                timeout=30,
            )
    finally:
        if not proxy_stopped:
            await proxy._stop()
        # Explicit unload so the model doesn't sit warm in real Ollama
        # if the in-test ``keep_alive: "30s"`` window hasn't elapsed by
        # the time the next test runs. Best-effort — failures here
        # don't fail the test.
        try:
            async with httpx.AsyncClient(timeout=10) as direct:
                await direct.post(
                    f"{DEFAULT_OLLAMA_HOST}/api/generate",
                    json={
                        "model": REQUIRED_MODEL,
                        "prompt": "",
                        "keep_alive": "0",
                    },
                )
        except (httpx.HTTPError, OSError):
            pass

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["error_type"] == "ConnectError", body
    assert body["model"] == REQUIRED_MODEL, body
    assert body["error"], "expected non-empty error message"


async def test_openai_compat_error_envelope_on_v1_chat(tmp_marshal_paths):
    """OpenAI-compat /v1/chat/completions failure → spec error envelope.

    The v0.6.5 fix preserved the OpenAI envelope shape
    (``{error: {message, type, code}}``) and added an
    ``error.exception_type`` field so clients can match on the actual
    Python class name without breaking compatibility with clients that
    only read ``error.type``. ``type`` and ``code`` are pinned to
    ``"proxy_error"`` matching OpenAI's
    ``invalid_request_error``/``rate_limit_error`` slug convention.
    """
    async with fault_proxy() as proxy:
        proxy.disconnect_next("/api/chat", times=99)
        cfg = _build_marshal_config(proxy_url=proxy.url, tmp_paths=tmp_marshal_paths)
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": REQUIRED_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "max_tokens": 4,
                },
                headers=_HDR,
                timeout=30,
            )
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert isinstance(body.get("error"), dict), (
        f"expected nested error envelope, got: {body}"
    )
    err = body["error"]
    assert err["type"] == "proxy_error", err
    assert err["code"] == "proxy_error", err
    assert err["exception_type"] == "RemoteProtocolError", err
    assert err.get("message"), "expected non-empty error.message"


async def test_preload_failed_error_returns_503(tmp_marshal_paths):
    """Preload exhaustion → 503 + ``error_type: "PreloadFailedError"``.

    Validates the v0.6.4 → v0.6.5 interaction: the scheduler hits
    ``preload_max_consecutive_failures`` consecutive preload failures
    against /api/generate, drains the queued envelope with
    ``PreloadFailedError`` (per ``Scheduler._give_up_on_preload``),
    and the server's ``_http_status_for_error`` maps that class to
    503. Tight backoff + max=2 keeps the test under a second.

    /api/ps is faked as empty so the memory poller cannot short-circuit
    preload by reporting the model as already loaded (which would
    happen if a prior test in the suite left it warm).
    """
    async with fault_proxy() as proxy:
        proxy.fake_response("/api/ps", {"models": []}, times=None)
        proxy.disconnect_next("/api/generate", times=99)
        cfg = _build_marshal_config(
            proxy_url=proxy.url,
            tmp_paths=tmp_marshal_paths,
            preload_max_consecutive_failures=2,
            preload_backoff_base_s=0.05,
        )
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            resp = await client.post(
                "/api/chat",
                json=_CHAT_BODY,
                headers=_HDR,
                timeout=30,
            )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["error_type"] == "PreloadFailedError", body
    assert body["model"] == REQUIRED_MODEL, body
    assert body["error"], "expected non-empty error message"


async def test_remote_protocol_error_returns_502(tmp_marshal_paths):
    """Forward disconnect → 502 + ``error_type: "RemoteProtocolError"``.

    The fault proxy's ``disconnect_next`` closes the socket without
    writing any response; httpx surfaces this as ``RemoteProtocolError``
    on marshal's side. Since RemoteProtocolError isn't matched by
    ``_http_status_for_error``'s timeout/network branches, it falls
    through to the default 502 — confirming the v0.6.5 default-case
    arm still works.
    """
    async with fault_proxy() as proxy:
        proxy.disconnect_next("/api/chat", times=99)
        cfg = _build_marshal_config(proxy_url=proxy.url, tmp_paths=tmp_marshal_paths)
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            resp = await client.post(
                "/api/chat",
                json=_CHAT_BODY,
                headers=_HDR,
                timeout=30,
            )
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error_type"] == "RemoteProtocolError", body
    assert body["model"] == REQUIRED_MODEL, body
    assert body["error"], "expected non-empty error message"
