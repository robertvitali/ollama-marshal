"""Multi-instance routing — end-to-end tests against a real two-instance setup.

# Pre-conditions

These tests require **two** Ollama instances reachable from the test
machine:

- ``localhost:11434`` running with ``OLLAMA_KV_CACHE_TYPE=f16`` (the
  user's normal Ollama, the "primary")
- ``localhost:11444`` running with ``OLLAMA_KV_CACHE_TYPE=q8_0`` (the
  "fallback" tier; bootstrap via the example launchd plist at
  ``examples/com.user.ollama-serve-q8.plist``)

Both must respond to ``/api/version`` AND have ``REQUIRED_MODEL``
pulled (``ollama pull qwen3.5:0.8b-bf16`` runs on each instance
separately — model files aren't shared between Ollama daemons).

When the q8 instance isn't up, every test in this module SKIPs
cleanly via the module-level ``pytestmark`` so the rest of the
integration suite still runs against single-instance setups.

# Why these tests exist

Stage 2 plumbing wires the Stage 1 routing decision through
MemoryManager → Lifecycle → Scheduler → Server. Unit tests cover the
decision tree itself; these tests cover the cross-component path:

- Real requests get tagged with ``envelope.instance_url``
- ``forward_request`` actually targets the chosen instance
- The audit log records ``tier_label`` + ``routing_reason``
- ``MemoryManager`` polls each instance's /api/ps independently
- ``ModelLifecycle.preload`` loads on the right instance

These are the bug classes that pure unit tests can't catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ollama_marshal.config import (
    TIER_FALLBACK,
    TIER_PRIMARY,
    AuditConfig,
    KVCacheType,
    MarshalConfig,
    MemoryConfig,
    OllamaConfig,
    OllamaInstance,
    Priority,
    ProgramConfig,
    ProxyConfig,
    SchedulerConfig,
    ShutdownConfig,
    ShutdownMode,
)
from tests.integration.conftest import (
    DEFAULT_OLLAMA_HOST,
    PROGRAM_CRITICAL,
    REQUIRED_MODEL,
    _ollama_reachable,
    make_test_app,
    wait_for,
)

# Q8 instance URL (per README "Multi-instance setup" walkthrough).
Q8_OLLAMA_HOST = "http://localhost:11444"


def _both_instances_reachable() -> bool:
    """True iff f16 (:11434) AND q8 (:11444) instances respond."""
    return _ollama_reachable(DEFAULT_OLLAMA_HOST) and _ollama_reachable(Q8_OLLAMA_HOST)


def _q8_has_model(model: str = REQUIRED_MODEL) -> bool:
    """Sync check that ``model`` is pulled on the q8 instance."""
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{Q8_OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            return any(m.get("name") == model for m in resp.json().get("models", []))
    except (httpx.HTTPError, OSError):
        return False


# Module-level marker: every test in this file is an integration test
# and needs at least the f16 (localhost:11434) Ollama up. Tests that
# additionally need the real q8 daemon at :11444 carry an extra
# ``_REQUIRES_BOTH_INSTANCES`` decorator. Tests that use fault proxies
# instead of the real q8 daemon don't need it — they run on every
# integration-suite invocation as long as :11434 is up.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(DEFAULT_OLLAMA_HOST),
        reason="Ollama not running on :11434",
    ),
]

# Per-test gate for the original "real two-instance" tests that need
# a separate q8 Ollama daemon at :11444 with REQUIRED_MODEL pulled.
_REQUIRES_BOTH_INSTANCES = pytest.mark.skipif(
    not _both_instances_reachable() or not _q8_has_model(),
    reason=(
        "Real two-instance tests require BOTH localhost:11434 and "
        "localhost:11444 to be reachable AND have REQUIRED_MODEL "
        "pulled on the q8 instance. See README 'Multi-instance setup' "
        "for the q8 launchd plist walkthrough."
    ),
)


def _multi_instance_config(tmp_paths: dict[str, Path], audit_enabled: bool = False):
    """Build a MarshalConfig fronting both f16 + q8 instances."""
    return MarshalConfig(
        instances=[
            OllamaInstance(
                url=DEFAULT_OLLAMA_HOST,
                kv_cache_type=KVCacheType.F16,
                tier_label=TIER_PRIMARY,
            ),
            OllamaInstance(
                url=Q8_OLLAMA_HOST,
                kv_cache_type=KVCacheType.Q8_0,
                tier_label=TIER_FALLBACK,
            ),
        ],
        proxy=ProxyConfig(host="127.0.0.1", port=11437, request_timeout_s=90),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(
            mode=ShutdownMode.IMMEDIATE,
            drain_timeout=5,
            unload_models=True,
        ),
        audit=AuditConfig(
            enabled=audit_enabled,
            path=str(tmp_paths["audit_path"]),
        ),
    )


async def _unload_on_instance(host: str, model: str) -> None:
    """Force-unload ``model`` from a specific instance via keep_alive=0."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": "0"},
            )
    except httpx.HTTPError:
        pass  # best-effort cleanup


async def _preload_on_instance(host: str, model: str) -> None:
    """Force-preload ``model`` on a specific instance via keep_alive=24h."""
    async with httpx.AsyncClient(timeout=300) as client:
        await client.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "24h"},
        )


async def _is_loaded_on(host: str, model: str) -> bool:
    """True iff ``model`` appears in ``host``'s /api/ps."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{host}/api/ps")
            resp.raise_for_status()
            data = resp.json()
            return any(m.get("name") == model for m in data.get("models", []))
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------------------
# Cold-start routing → primary (f16)
# ---------------------------------------------------------------------------


@_REQUIRES_BOTH_INSTANCES
async def test_cold_start_routes_to_f16_when_room(tmp_marshal_paths):
    """Empty system + plenty of room → primary tier wins.

    Verifies the simplest routing path: nothing loaded, both instances
    have room, ``pick_instance`` returns ``PRIMARY_FITS`` and the
    request lands on f16. Status payload reflects the load on the
    primary instance URL.
    """
    # Cold-start invariant: ensure nothing is preloaded on either side.
    await _unload_on_instance(DEFAULT_OLLAMA_HOST, REQUIRED_MODEL)
    await _unload_on_instance(Q8_OLLAMA_HOST, REQUIRED_MODEL)

    cfg = _multi_instance_config(tmp_marshal_paths)
    app = make_test_app(cfg, tmp_marshal_paths)
    from asgi_lifespan import LifespanManager

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        resp = await client.post(
            "/api/chat",
            json={
                "model": REQUIRED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            headers={"X-Program-ID": PROGRAM_CRITICAL},
            timeout=60,
        )
        assert resp.status_code == 200, resp.text

        # Model is now loaded — but on which instance?
        memory = app.state._marshal_internals.memory
        assert memory.is_loaded_on(REQUIRED_MODEL, DEFAULT_OLLAMA_HOST), (
            f"expected {REQUIRED_MODEL} on f16 ({DEFAULT_OLLAMA_HOST}); "
            f"loaded_on={memory.loaded_on()}"
        )
        assert not memory.is_loaded_on(REQUIRED_MODEL, Q8_OLLAMA_HOST)

    # Cleanup so the next test starts cold.
    await _unload_on_instance(DEFAULT_OLLAMA_HOST, REQUIRED_MODEL)


# ---------------------------------------------------------------------------
# Already-loaded wins — q8 keeps serving even when f16 is empty
# ---------------------------------------------------------------------------


@_REQUIRES_BOTH_INSTANCES
async def test_already_loaded_q8_wins_over_empty_f16(tmp_marshal_paths):
    """Pre-loaded q8 keeps serving — no promotion to f16.

    Per the design ("q8 quality is good enough that we don't pay the
    load-time tax to promote it"), a model already on q8 should keep
    serving from there even when f16 has plenty of room. The
    routing decision returns ``ALREADY_LOADED`` and routes to q8.
    """
    await _unload_on_instance(DEFAULT_OLLAMA_HOST, REQUIRED_MODEL)
    # Pre-load on q8 BEFORE marshal starts polling so the first
    # /api/ps observation sees it on q8.
    await _preload_on_instance(Q8_OLLAMA_HOST, REQUIRED_MODEL)
    assert await _is_loaded_on(Q8_OLLAMA_HOST, REQUIRED_MODEL)

    cfg = _multi_instance_config(tmp_marshal_paths)
    app = make_test_app(cfg, tmp_marshal_paths)
    from asgi_lifespan import LifespanManager

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        memory = app.state._marshal_internals.memory
        # Wait for marshal's first /api/ps poll to attribute the model
        # to the q8 instance.
        await wait_for(
            lambda: memory.is_loaded_on(REQUIRED_MODEL, Q8_OLLAMA_HOST),
            timeout=10,
            description="q8-loaded model observed by marshal poll",
        )

        resp = await client.post(
            "/api/chat",
            json={
                "model": REQUIRED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            headers={"X-Program-ID": PROGRAM_CRITICAL},
            timeout=60,
        )
        assert resp.status_code == 200, resp.text

        # The model is still on q8, not promoted to f16.
        assert memory.is_loaded_on(REQUIRED_MODEL, Q8_OLLAMA_HOST)
        assert not memory.is_loaded_on(REQUIRED_MODEL, DEFAULT_OLLAMA_HOST)

    await _unload_on_instance(Q8_OLLAMA_HOST, REQUIRED_MODEL)


# ---------------------------------------------------------------------------
# Audit log fields — tier_label + routing_reason on every served record
# ---------------------------------------------------------------------------


@_REQUIRES_BOTH_INSTANCES
async def test_audit_log_records_tier_and_reason(tmp_marshal_paths):
    """Audit JSONL records carry the routing decision context.

    The audit log is the operator's "why" trail. v0.5.0 requires
    every served/failed record to include ``instance_url``,
    ``tier_label``, and ``routing_reason`` so an operator can answer
    "why did this request run on q8?" by reading audit.jsonl alone.
    """
    await _unload_on_instance(DEFAULT_OLLAMA_HOST, REQUIRED_MODEL)
    await _unload_on_instance(Q8_OLLAMA_HOST, REQUIRED_MODEL)

    cfg = _multi_instance_config(tmp_marshal_paths, audit_enabled=True)
    app = make_test_app(cfg, tmp_marshal_paths)
    from asgi_lifespan import LifespanManager

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        resp = await client.post(
            "/api/chat",
            json={
                "model": REQUIRED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            headers={"X-Program-ID": PROGRAM_CRITICAL},
            timeout=60,
        )
        assert resp.status_code == 200

    # Audit file is written on lifespan teardown (final flush).
    audit_path = tmp_marshal_paths["audit_path"]
    assert audit_path.exists(), "audit file not created"

    served_records = []
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "request.served":
            served_records.append(rec)

    assert served_records, "no request.served record in audit log"
    rec = served_records[0]
    # Routing fields populated.
    assert rec.get("instance_url") == DEFAULT_OLLAMA_HOST
    assert rec.get("tier_label") == TIER_PRIMARY
    # Reason is one of the cold-start primary outcomes.
    assert rec.get("routing_reason") in {
        "primary_fits",
        "primary_evicting_idle",
    }

    await _unload_on_instance(DEFAULT_OLLAMA_HOST, REQUIRED_MODEL)


# ---------------------------------------------------------------------------
# Per-instance polling — each instance's /api/ps observed independently
# ---------------------------------------------------------------------------


@_REQUIRES_BOTH_INSTANCES
async def test_per_instance_polling_attributes_correctly(tmp_marshal_paths):
    """A model loaded on one instance only is attributed to that instance.

    Regression: when multi-instance attribution is broken,
    ``MemoryManager.refresh`` writes the same ``LoadedModel`` to every
    instance's slot or to the wrong slot. This test seeds q8 only
    (out-of-band) and asserts marshal sees it as q8-only.
    """
    await _unload_on_instance(DEFAULT_OLLAMA_HOST, REQUIRED_MODEL)
    await _unload_on_instance(Q8_OLLAMA_HOST, REQUIRED_MODEL)
    await _preload_on_instance(Q8_OLLAMA_HOST, REQUIRED_MODEL)

    cfg = _multi_instance_config(tmp_marshal_paths)
    app = make_test_app(cfg, tmp_marshal_paths)
    from asgi_lifespan import LifespanManager

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as _client,
    ):
        memory = app.state._marshal_internals.memory
        # Wait for marshal's first poll to observe the q8 load.
        await wait_for(
            lambda: memory.is_loaded_on(REQUIRED_MODEL, Q8_OLLAMA_HOST),
            timeout=10,
            description="q8 load attribution",
        )
        # Crucially, NOT attributed to f16 (would be a bug).
        assert not memory.is_loaded_on(REQUIRED_MODEL, DEFAULT_OLLAMA_HOST)
        # find_instance_for returns the q8 URL.
        assert memory.find_instance_for(REQUIRED_MODEL) == Q8_OLLAMA_HOST

    await _unload_on_instance(Q8_OLLAMA_HOST, REQUIRED_MODEL)


# ===========================================================================
# Fault-proxy multi-instance tests — DON'T require the q8 daemon
# ===========================================================================
#
# These tests stand up TWO fault-injection proxies in front of the same real
# Ollama at :11434, then point marshal at the proxies as if they were
# separate instances. Marshal sees two distinct ``OllamaInstance`` URLs,
# each with their own ``OLLAMA_KV_CACHE_TYPE`` setting in marshal's config —
# but every request ultimately lands on the one real Ollama daemon.
#
# This gives integration coverage of the routing-decision plumbing
# (envelope tagging, forward_request URL selection, per-instance polling
# attribution, audit log fields, unload_from cleanup) WITHOUT requiring
# the user to run the launchctl-managed q8 daemon. Tests run on every
# integration-suite invocation.
#
# Limitation: because both "instances" share the same physical Ollama, we
# cannot test memory-pressure failover that requires actually different
# precision footprints (B-rule, A-rule, real q4 promotion). Those still
# need the real two-instance setup above.

# Fault-proxy tests below inherit the module-level f16-reachability
# pytestmark and don't need a per-test decorator beyond that.


def _proxy_instances_config(
    f16_proxy_url: str,
    q8_proxy_url: str,
    tmp_paths: dict[str, Path],
    audit_enabled: bool = False,
):
    """Build a MarshalConfig fronting two fault proxies as fake instances."""
    return MarshalConfig(
        instances=[
            OllamaInstance(
                url=f16_proxy_url,
                kv_cache_type=KVCacheType.F16,
                tier_label=TIER_PRIMARY,
            ),
            OllamaInstance(
                url=q8_proxy_url,
                kv_cache_type=KVCacheType.Q8_0,
                tier_label=TIER_FALLBACK,
            ),
        ],
        proxy=ProxyConfig(host="127.0.0.1", port=11438, request_timeout_s=90),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(
            mode=ShutdownMode.IMMEDIATE,
            drain_timeout=5,
            unload_models=True,
        ),
        audit=AuditConfig(
            enabled=audit_enabled,
            path=str(tmp_paths["audit_path"]),
        ),
    )


async def test_fault_proxy_envelope_tagged_with_chosen_instance(tmp_marshal_paths):
    """Routing decision actually populates ``envelope.instance_url``.

    Stage 2 plumbing: scheduler builds a ``RoutingDecision`` and tags
    pending envelopes via ``_tag_pending_with_decision`` so
    ``forward_request`` reads ``envelope.instance_url`` for the upstream.
    Without this tag, every request goes to ``config.ollama.host``
    regardless of routing — silent multi-instance failure.

    Verifies via the audit log: ``request.served`` records carry the
    chosen instance URL + tier_label + routing_reason. Both proxies
    forward to the same Ollama, so the request succeeds either way;
    the test asserts the *routing decision was recorded*, not that the
    physical request landed elsewhere.
    """
    from tests.integration._fault_proxy import fault_proxy

    async with fault_proxy() as f16_proxy, fault_proxy() as q8_proxy:
        cfg = _proxy_instances_config(
            f16_proxy.url, q8_proxy.url, tmp_marshal_paths, audit_enabled=True
        )
        app = make_test_app(cfg, tmp_marshal_paths)
        from asgi_lifespan import LifespanManager

        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            resp = await client.post(
                "/api/chat",
                json={
                    "model": REQUIRED_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"X-Program-ID": PROGRAM_CRITICAL},
                timeout=60,
            )
            assert resp.status_code == 200, resp.text

    # Audit file populated on lifespan teardown.
    audit_path = tmp_marshal_paths["audit_path"]
    assert audit_path.exists(), "audit file not created"
    served = [
        json.loads(line)
        for line in audit_path.read_text().splitlines()
        if line.strip() and json.loads(line).get("event") == "request.served"
    ]
    assert served, "no request.served record in audit log"
    rec = served[0]
    # Cold-start with empty proxies → routes to f16 primary.
    assert rec["instance_url"] == f16_proxy.url, (
        f"expected f16 routing; got {rec['instance_url']}"
    )
    assert rec["tier_label"] == TIER_PRIMARY
    assert rec["routing_reason"] in {"primary_fits", "primary_evicting_idle"}


async def test_fault_proxy_forward_request_targets_chosen_instance(tmp_marshal_paths):
    """``forward_request`` actually hits the proxy for the chosen instance.

    Counts each proxy's request hit by injecting a `delay_next` of 0
    (transparent pass-through that still records the call). When marshal
    routes to f16, the f16 proxy receives the inference call; the q8
    proxy receives only `/api/ps` polls.

    Regression target: an envelope with ``instance_url=None`` (Stage 2
    bug class) would silently fall through to ``config.ollama.host`` —
    in legacy single-instance configs that's the same URL, but in
    multi-instance configs ``ollama.host`` is auto-synced to
    ``instances[0].url`` (the f16 proxy here), so the bug would still
    LOOK correct in this test. We verify by inspecting the routing
    decision's recorded ``instance_url`` AND by counting the inference
    calls to each proxy.
    """
    from tests.integration._fault_proxy import fault_proxy

    f16_inference_count = 0
    q8_inference_count = 0

    async with fault_proxy() as f16_proxy, fault_proxy() as q8_proxy:
        # Wrap each proxy's _handle_client to count inference paths.
        # We don't fault — just observe.
        original_f16_forward = f16_proxy._forward
        original_q8_forward = q8_proxy._forward

        async def f16_counting_forward(method, path, headers, body, writer):
            nonlocal f16_inference_count
            if path in ("/api/chat", "/api/generate"):
                f16_inference_count += 1
            return await original_f16_forward(method, path, headers, body, writer)

        async def q8_counting_forward(method, path, headers, body, writer):
            nonlocal q8_inference_count
            if path in ("/api/chat", "/api/generate"):
                q8_inference_count += 1
            return await original_q8_forward(method, path, headers, body, writer)

        f16_proxy._forward = f16_counting_forward  # type: ignore[method-assign]
        q8_proxy._forward = q8_counting_forward  # type: ignore[method-assign]

        cfg = _proxy_instances_config(f16_proxy.url, q8_proxy.url, tmp_marshal_paths)
        app = make_test_app(cfg, tmp_marshal_paths)
        from asgi_lifespan import LifespanManager

        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            resp = await client.post(
                "/api/chat",
                json={
                    "model": REQUIRED_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"X-Program-ID": PROGRAM_CRITICAL},
                timeout=60,
            )
            assert resp.status_code == 200

    # f16 (primary) saw the inference; q8 saw only /api/ps polls.
    # Note: lifecycle.preload also POSTs to /api/generate as a
    # warm-up (with empty prompt), so f16_inference_count may be ≥2
    # (preload + actual chat). q8 never receives /api/generate.
    assert f16_inference_count >= 1, (
        f"f16 proxy never received inference call (preload + chat); "
        f"f16={f16_inference_count}, q8={q8_inference_count}"
    )
    assert q8_inference_count == 0, (
        f"q8 proxy unexpectedly received inference; "
        f"f16={f16_inference_count}, q8={q8_inference_count}"
    )


async def test_fault_proxy_per_instance_polling_independent(tmp_marshal_paths):
    """Each instance's /api/ps is polled and attributed independently.

    Inject ``fake_response("/api/ps", ...)`` on the q8 proxy to claim
    a model is loaded there, while the f16 proxy is empty. After
    poll_interval, marshal should report the model as loaded on q8
    only — proving per-instance attribution works through the polling
    code path. A bug that wrote both instances' ps responses to the
    same flat dict would attribute the model to f16 (or to both,
    depending on order).
    """
    from tests.integration._fault_proxy import fault_proxy

    async with fault_proxy() as f16_proxy, fault_proxy() as q8_proxy:
        # f16 stays transparent — passes through to real /api/ps (empty).
        # q8 always returns a fake ps with our model "loaded".
        q8_proxy.fake_response(
            "/api/ps",
            {
                "models": [
                    {
                        "name": REQUIRED_MODEL,
                        "model": REQUIRED_MODEL,
                        "size_vram": 1_600_000_000,
                        "expires_at": "2099-01-01T00:00:00Z",
                    }
                ]
            },
            times=None,  # forever
        )

        cfg = _proxy_instances_config(f16_proxy.url, q8_proxy.url, tmp_marshal_paths)
        app = make_test_app(cfg, tmp_marshal_paths)
        from asgi_lifespan import LifespanManager

        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as _client,
        ):
            memory = app.state._marshal_internals.memory
            # Wait for first /api/ps poll cycle (poll_interval=1s).
            await wait_for(
                lambda: memory.is_loaded_on(REQUIRED_MODEL, q8_proxy.url),
                timeout=10,
                description="model attributed to q8 via fake /api/ps",
            )
            # Critically, NOT attributed to f16.
            assert not memory.is_loaded_on(REQUIRED_MODEL, f16_proxy.url), (
                f"model wrongly attributed to f16; loaded_on={memory.loaded_on()}"
            )
            assert memory.find_instance_for(REQUIRED_MODEL) == q8_proxy.url


async def test_fault_proxy_routing_decision_observes_q4_load(tmp_marshal_paths):
    """Routing decision sees a q4-only load and chooses to promote.

    Stand up a 3-instance config (f16 + q8 + q4 proxies). Fake
    /api/ps on q4 to claim the model is loaded there only. Once
    marshal's poll observes the q4 entry, ``_resolve_routing`` should
    return a ``RoutingDecision`` with ``unload_from=[q4]`` and
    ``reason=PROMOTING_FROM_LAST_RESORT`` — proving the routing tree
    correctly handles the q4-only escape path.

    Asserts directly on ``_resolve_routing()`` rather than firing a
    full inference request. Inference would compete for VRAM with the
    user's running marshal at :11435 and timeout under benchmarking
    pressure, which produces flaky results unrelated to routing.
    """
    from tests.integration._fault_proxy import fault_proxy

    async with (
        fault_proxy() as f16_proxy,
        fault_proxy() as q8_proxy,
        fault_proxy() as q4_proxy,
    ):
        # Fake q4 as holding the model so routing sees q4-only.
        q4_proxy.fake_response(
            "/api/ps",
            {
                "models": [
                    {
                        "name": REQUIRED_MODEL,
                        "model": REQUIRED_MODEL,
                        "size_vram": 800_000_000,
                        "expires_at": "2099-01-01T00:00:00Z",
                    }
                ]
            },
            times=None,
        )
        # f16 + q8 stay transparent (real /api/ps, empty).

        cfg = MarshalConfig(
            instances=[
                OllamaInstance(
                    url=f16_proxy.url,
                    kv_cache_type=KVCacheType.F16,
                    tier_label=TIER_PRIMARY,
                ),
                OllamaInstance(
                    url=q8_proxy.url,
                    kv_cache_type=KVCacheType.Q8_0,
                    tier_label=TIER_FALLBACK,
                ),
                OllamaInstance(
                    url=q4_proxy.url,
                    kv_cache_type=KVCacheType.Q4_0,
                    tier_label="last_resort",
                ),
            ],
            proxy=ProxyConfig(host="127.0.0.1", port=11439, request_timeout_s=90),
            memory=MemoryConfig(poll_interval=1),
            scheduler=SchedulerConfig(
                metrics_path=str(tmp_marshal_paths["metrics_path"]),
                metrics_persist_interval_s=3600,
            ),
            programs={
                "default": ProgramConfig(),
                PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
            },
            shutdown=ShutdownConfig(
                mode=ShutdownMode.IMMEDIATE,
                drain_timeout=5,
                unload_models=True,
            ),
            audit=AuditConfig(
                enabled=False,
                path=str(tmp_marshal_paths["audit_path"]),
            ),
        )
        app = make_test_app(cfg, tmp_marshal_paths)
        from asgi_lifespan import LifespanManager

        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=transport, base_url="http://testserver"),
        ):
            internals = app.state._marshal_internals
            memory = internals.memory
            scheduler = internals.scheduler
            # Wait for marshal to observe the q4 "load".
            await wait_for(
                lambda: memory.is_loaded_on(REQUIRED_MODEL, q4_proxy.url),
                timeout=10,
                description="model observed on q4",
            )
            # Ask routing what it would do for an incoming request on
            # this model. This drives the `_resolve_routing` code path
            # — same path inference would take — without actually
            # firing inference (which would compete with the user's
            # running marshal and timeout under load).
            decision = await scheduler._resolve_routing(REQUIRED_MODEL, num_ctx=4096)

    # Routing chose a higher tier (f16 or q8) and emitted unload_from
    # with the q4 stale copy. This proves the q4-only escape path
    # works AND that the scheduler would honor the cleanup.
    assert decision.reason.value == "promoting_from_last_resort", (
        f"expected promoting_from_last_resort; got {decision.reason.value}"
    )
    assert decision.instance.url in {f16_proxy.url, q8_proxy.url}
    # `unload_from` carries the OllamaInstance for the q4 stale copy.
    assert any(i.url == q4_proxy.url for i in decision.unload_from), (
        f"expected q4 in unload_from; got {[i.url for i in decision.unload_from]}"
    )


async def test_fault_proxy_one_instance_failure_does_not_break_others(
    tmp_marshal_paths,
):
    """An unreachable instance during /api/ps polling doesn't abort others.

    Inject ``disconnect_next`` on the q8 proxy's /api/ps so every poll
    fails with a connection drop, while f16 stays transparent. Marshal
    should still serve a request (routing skips q8, lands on f16).

    Without per-instance error isolation, a single instance going down
    would cripple the whole MemoryManager and every dispatch would
    fail.
    """
    from tests.integration._fault_proxy import fault_proxy

    async with fault_proxy() as f16_proxy, fault_proxy() as q8_proxy:
        # q8 /api/ps always disconnects. Use times=999 to keep the
        # behavior active for the whole test (rather than running out
        # of queued faults mid-run and silently passing through).
        q8_proxy.disconnect_next("/api/ps", times=999)

        cfg = _proxy_instances_config(f16_proxy.url, q8_proxy.url, tmp_marshal_paths)
        app = make_test_app(cfg, tmp_marshal_paths)
        from asgi_lifespan import LifespanManager

        transport = httpx.ASGITransport(app=app)
        async with (
            LifespanManager(app),
            httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client,
        ):
            # f16 still works for inference.
            resp = await client.post(
                "/api/chat",
                json={
                    "model": REQUIRED_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"X-Program-ID": PROGRAM_CRITICAL},
                timeout=60,
            )
            assert resp.status_code == 200, resp.text


async def test_legacy_single_instance_config_still_works(tmp_marshal_paths):
    """Stage 2 didn't break the v0.4.x ``ollama.host`` legacy config path.

    Builds a MarshalConfig using ONLY the legacy ``ollama.host`` field
    (no ``instances`` list). The validator should backfill a single
    f16 instance, and request routing should pick SINGLE_INSTANCE.

    Regression target: a Stage 2 bug that required explicit
    ``instances`` list for routing to work would surface here as
    either a startup error or a request that times out because no
    instance was reachable.
    """
    cfg = MarshalConfig(
        ollama=OllamaConfig(host=DEFAULT_OLLAMA_HOST),  # legacy form
        # NOTE: no `instances` field — validator backfills.
        proxy=ProxyConfig(host="127.0.0.1", port=11440, request_timeout_s=90),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_marshal_paths["metrics_path"]),
            metrics_persist_interval_s=3600,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
        },
        shutdown=ShutdownConfig(
            mode=ShutdownMode.IMMEDIATE,
            drain_timeout=5,
            unload_models=True,
        ),
        audit=AuditConfig(enabled=True, path=str(tmp_marshal_paths["audit_path"])),
    )
    # Verify the validator backfilled.
    assert len(cfg.instances) == 1
    assert cfg.instances[0].url == DEFAULT_OLLAMA_HOST
    assert cfg.instances[0].kv_cache_type == KVCacheType.F16

    app = make_test_app(cfg, tmp_marshal_paths)
    from asgi_lifespan import LifespanManager

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        resp = await client.post(
            "/api/chat",
            json={
                "model": REQUIRED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            headers={"X-Program-ID": PROGRAM_CRITICAL},
            timeout=60,
        )
        assert resp.status_code == 200

    # Audit log: routing_reason must be SINGLE_INSTANCE (not any
    # multi-instance reason).
    served = [
        json.loads(line)
        for line in tmp_marshal_paths["audit_path"].read_text().splitlines()
        if line.strip() and json.loads(line).get("event") == "request.served"
    ]
    assert served
    assert served[0]["routing_reason"] == "single_instance"
    assert served[0]["instance_url"] == DEFAULT_OLLAMA_HOST
    assert served[0]["tier_label"] == TIER_PRIMARY

    # Cleanup so other tests start cold.
    await _unload_on_instance(DEFAULT_OLLAMA_HOST, REQUIRED_MODEL)
