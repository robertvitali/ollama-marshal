"""End-to-end coverage for v0.6.6 periodic /api/tags pruning.

v0.6.5 and earlier left ``ModelRegistry._known_models`` frozen between
process startup and the first opportunistic re-sync triggered by an
unknown-model request. If a user ran ``ollama rm <model>`` while
marshal was running, subsequent inference requests for the removed
model would slip past the server's fail-fast 404 check and ride the
preload retry budget into a 502 ``PreloadFailedError`` after several
seconds — masking what's really a "model not installed" condition
from the operator.

v0.6.6 wires a periodic ``_sync_with_ollama`` poll task driven by
``scheduler.model_detect_interval``, so a removed model shows up in
``_known_models`` (or rather, disappears from it) within one poll
interval and the existing 404 fail-fast covers the registry-stale
case.

This test exercises that path against real Ollama via the
``FaultProxy`` (which can ``fake_response`` /api/tags to simulate the
post-removal world without needing to actually ``ollama rm`` an
installed model). All other endpoints pass through.
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
    SchedulerConfig,
    ShutdownConfig,
    ShutdownMode,
)
from tests.integration._fault_proxy import fault_proxy
from tests.integration.conftest import (
    INTEGRATION_FORWARD_TIMEOUT_S,
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


def _build_config(
    *,
    proxy_url: str,
    tmp_paths: dict[str, Path],
    model_detect_interval: int = 1,
) -> MarshalConfig:
    """MarshalConfig pointed at the fault proxy with a tight detect interval.

    Default ``model_detect_interval`` of 1 second keeps the test wall-
    clock low — production default is 30s. Memory poll stays at 1 to
    match the rest of the integration suite.
    """
    return MarshalConfig(
        ollama=OllamaConfig(host=proxy_url),
        proxy=ProxyConfig(host="127.0.0.1", port=11436),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
            benchmark_on_startup=False,
            model_detect_interval=model_detect_interval,
            ollama_forward_timeout_s=INTEGRATION_FORWARD_TIMEOUT_S,
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
        audit=AuditConfig(enabled=False, path=str(tmp_paths["audit_path"])),
    )


async def test_registry_prunes_known_model_after_external_removal(
    tmp_marshal_paths,
):
    """`ollama rm <model>` simulated via fake /api/tags → next request 404s.

    Setup:
    - Marshal initialized; initial /api/tags sync pulls the live list
      (which includes REQUIRED_MODEL) through the proxy. Sanity check
      that REQUIRED_MODEL is in ``_known_models``.
    - Test queues a permanent fake_response on /api/tags returning an
      empty model list — every subsequent poll sees REQUIRED_MODEL gone.
    - Wait for the poll task to apply the new state.
    - Inference request for REQUIRED_MODEL must fail-fast with 404,
      not 502 (which is what the pre-fix preload-retry path produced).
    """
    async with fault_proxy() as proxy:
        cfg = _build_config(proxy_url=proxy.url, tmp_paths=tmp_marshal_paths)
        app = make_test_app(cfg, tmp_marshal_paths)
        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            registry = app.state._marshal_internals.registry

            # Initial sync ran inside lifespan startup against the real
            # /api/tags (passing through the proxy). Sanity-check that
            # the test model is present so the subsequent removal
            # actually exercises the prune path.
            assert REQUIRED_MODEL in registry._known_models, (
                f"sanity: {REQUIRED_MODEL!r} expected in registry._known_models "
                f"after initial sync; got {sorted(registry._known_models)}"
            )

            # Simulate `ollama rm REQUIRED_MODEL` — every subsequent
            # /api/tags poll returns an empty list. ``times=None`` keeps
            # the fake in place for the rest of the test.
            proxy.fake_response("/api/tags", {"models": []}, times=None)

            # Periodic poll should pick up the change within one
            # interval (configured to 1s above; 20s timeout for CI
            # variability — initial sync runs in lifespan startup and
            # eats some of the budget, plus cold-start GC pauses can
            # add seconds before the first poll fires).
            await wait_for(
                lambda: REQUIRED_MODEL not in registry._known_models,
                timeout=20.0,
                description=f"{REQUIRED_MODEL} pruned from registry",
            )

            # Subsequent inference request must fail-fast with 404, not
            # ride the preload retry budget into a 502. The error body
            # carries the operator-actionable hint to run `ollama pull`.
            resp = await client.post(
                "/api/chat",
                json={
                    "model": REQUIRED_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"X-Program-ID": PROGRAM_CRITICAL},
                timeout=10,
            )

    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert "error" in body, body
    err = body["error"]
    assert REQUIRED_MODEL in err, err
    assert "ollama pull" in err.lower(), err
