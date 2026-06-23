"""Shared fixtures for the integration test suite.

Integration tests run a real marshal app in-process via FastAPI's
``httpx.ASGITransport`` and ``asgi-lifespan``'s ``LifespanManager``. The
app talks to the user's actual Ollama at ``localhost:11434``. No
uvicorn subprocess. No mocks at the HTTP boundary.

Each test gets isolated state via per-test temp directories for
registry/audit/metrics paths. Test envelopes ride at CRITICAL priority
by default (program_id ``integration-test``); tests that specifically
need normal-priority behavior (e.g. drain-before-evict) opt in to
``integration-test-normal``.

If Ollama isn't reachable on :11434, every test in this directory
SKIPs cleanly via the module-level ``pytestmark`` set in each test
file. See ``CLAUDE.md`` Testing Rules for the integration-suite
conventions.

Constrained-budget sizing (Bug B, v0.6.7): tests that tighten the
memory budget to force eviction (e.g.
``test_marshal_eviction_drains_then_unloads``) rely on the
quantization-aware size estimator (Bug A) for accuracy on a cold
registry cache, NOT on a pre-seeded size file. Each test gets a
fresh ``registry_path`` (empty ``_sizes``), so first-touch sizing
falls through to the estimator; Bug A made that fallback typically
accurate within roughly 10-20%, enough for the constrained-budget
tests to pass deterministically without per-test benchmarking. We
deliberately did NOT pre-seed sizes from the prod cache (dev/CI
coupling) or ship a known-sizes JSON (drift risk). If the eviction
test ever flakes because the estimator margin matters at a tight
budget, graduate to a shipped known-sizes fixture (Bug B Option 2).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio
import structlog
from asgi_lifespan import LifespanManager

from ollama_marshal import __version__
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
from ollama_marshal.server import create_app

from ._admin_token import discover_admin_token, discover_bypass_token
from ._version_skew import version_skew_reason

if TYPE_CHECKING:
    from fastapi import FastAPI

# The smallest model used across the integration suite. ~1.6 GB.
# Tests that need a different model parametrize the model name
# explicitly; this is the default for "I just need any model".
REQUIRED_MODEL = "qwen3.5:0.8b-bf16"

# Default Ollama host the suite expects to be reachable.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Forward-timeout used by every integration test config (Hop 2: marshal
# → Ollama). v0.6.6 bumped from 120s to 900s (15 min) to absorb the
# inherent VRAM-contention window opened by ``PAUSE_DRAIN_TIMEOUT_S=0``
# below.
#
# Why 15 minutes:
# - The autouse ``pause_local_prod_marshal`` fixture stops NEW prod
#   dispatches but lets in-flight Ollama inferences finish naturally
#   (the v0.6.3 trade-off — see ``PAUSE_DRAIN_TIMEOUT_S`` rationale).
# - When prod has heavy models (gpt-oss:120b, qwen3:235b) actively
#   serving, those inferences hold Ollama VRAM until they complete
#   (~5-30s typically, longer for big context).
# - Test marshal preloads land fine but the chat call to Ollama then
#   sits behind that contention. With the old 120s budget, even a
#   tiny qwen3.5:0.8b chat with num_predict=4 would ReadTimeout
#   sporadically and fail multi_instance tests.
# - 900s is generous enough that any genuine progress (Ollama eventually
#   freeing VRAM and serving the request) completes within budget;
#   anything longer signals a real bug worth surfacing.
#
# Why not a "progressive" no-progress timeout:
# - HTTP doesn't expose mid-request progress for ``stream: False``
#   chats — we send the prompt, then wait for the full response.
# - For streaming, marshal already proxies chunks as they arrive.
# - A true progress-aware timeout would require marshal to observe
#   /api/ps for evidence of activity (size_vram, expires_at advancing).
#   That's v0.7+ work; bumping the wall-clock budget gets us reliable
#   pre-push hooks in the meantime.
#
# Single source of truth — every fixture and inline test client
# reads from this constant.
INTEGRATION_FORWARD_TIMEOUT_S = 900

# Program ID conventions — tests use these via the X-Program-ID header.
# integration-test (critical) is the default; integration-test-normal
# is for tests of normal-priority paths (drain-before-evict, etc.).
PROGRAM_CRITICAL = "integration-test"
PROGRAM_NORMAL = "integration-test-normal"


def _ollama_reachable(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 1.0) -> bool:
    """Return True if Ollama responds to /api/version at ``host``.

    Used as the ``skipif`` condition on every integration test file.
    Synchronous on purpose — pytestmark needs a value at collection
    time, before any event loop exists.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{host}/api/version")
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


@pytest.fixture
def tmp_marshal_paths(tmp_path: Path) -> dict[str, Path]:
    """Per-test temp paths for registry/audit/metrics state.

    Every fixture that builds a ``MarshalConfig`` reads from this so
    nothing leaks between tests or to the user's home directory.
    """
    return {
        "registry_path": tmp_path / "model_sizes.json",
        "metadata_path": tmp_path / "model_metadata.json",
        "vram_path": tmp_path / "model_vram.json",
        "audit_path": tmp_path / "audit.jsonl",
        "metrics_path": tmp_path / "metrics.json",
    }


@pytest.fixture
def marshal_config(tmp_marshal_paths: dict[str, Path]) -> MarshalConfig:
    """Test-tuned MarshalConfig wired to the per-test temp paths.

    Test defaults that differ from production:

    - ``scheduler.ollama_forward_timeout_s = INTEGRATION_FORWARD_TIMEOUT_S``
      (currently 900s / 15 min) — see the constant's docstring at
      module top for the rationale and tuning history.
    - ``memory.poll_interval = 1`` — speed up unexpected-unload tests.
      Real production uses 5s.
    - ``shutdown.mode = IMMEDIATE``, ``unload_models = True`` — when
      the lifespan tears down at end of test, every loaded model gets
      unloaded so the next test starts cold.
    - ``scheduler.benchmark_on_startup = False`` — production marshal
      benchmarks every uncached model on startup. With per-test temp
      registry paths the cache is always empty, so the benchmark task
      would load every installed model through the test fault-proxy
      (or real Ollama) on every test, saturating the upstream and
      starving the test's own request behind ~10s+ per model load.
    - ``programs`` pre-populated with two profiles:
      - ``integration-test`` → CRITICAL (default for all tests)
      - ``integration-test-normal`` → NORMAL (opt-in for tests of
        normal-priority paths like drain-before-evict)
    """
    return MarshalConfig(
        ollama=OllamaConfig(host=DEFAULT_OLLAMA_HOST),
        proxy=ProxyConfig(host="127.0.0.1", port=11436),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_marshal_paths["metrics_path"]),
            metrics_persist_interval_s=3600,  # don't write during tests
            benchmark_on_startup=False,
            ollama_forward_timeout_s=INTEGRATION_FORWARD_TIMEOUT_S,
        ),
        programs={
            "default": ProgramConfig(),
            PROGRAM_CRITICAL: ProgramConfig(priority=Priority.CRITICAL),
            PROGRAM_NORMAL: ProgramConfig(priority=Priority.NORMAL),
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


def make_test_app(cfg: MarshalConfig, tmp_marshal_paths: dict[str, Path]) -> FastAPI:
    """Build a marshal FastAPI app with test-isolated registry paths.

    Tests that need custom MarshalConfig (e.g. constrained memory budget,
    audit enabled, idle eviction interval) build their own ``cfg`` and
    pass it here. Without this helper, an inline ``create_app(cfg)``
    call would skip the ``app.state.registry_path/metadata_path``
    overrides and the ModelRegistry would fall back to its default
    on-disk cache at ``~/.ollama-marshal/`` — clobbering the user's
    production-marshal registry.

    Sets the same per-test path overrides the ``marshal_app`` fixture
    sets, so tests using either path get identical isolation.

    Also force-disables ``scheduler.benchmark_on_startup``. Per-test
    registry paths start with an empty cache, and the production
    benchmark sweep would otherwise load every installed model
    through the test fault-proxy on every startup — saturating the
    upstream and starving the test's own request behind ~10s+ per
    model load. This belt-and-suspenders override catches inline
    SchedulerConfig builders that forget to set the flag.
    """
    # model_copy bypasses root validators; relies on cfg already being
    # a validated MarshalConfig instance (callers always pass one).
    cfg = cfg.model_copy(
        update={
            "scheduler": cfg.scheduler.model_copy(
                update={"benchmark_on_startup": False}
            )
        }
    )
    # Regression guard: if a future Pydantic version or refactor changes
    # the model_copy semantics so the override silently fails, this
    # assertion fires loudly during test setup instead of letting the
    # benchmark sweep run through the test fault_proxy.
    assert cfg.scheduler.benchmark_on_startup is False, (
        "make_test_app override failed — benchmark_on_startup is still True"
    )
    app = create_app(cfg)
    app.state.metrics_path = tmp_marshal_paths["metrics_path"]
    app.state.registry_path = tmp_marshal_paths["registry_path"]
    app.state.metadata_path = tmp_marshal_paths["metadata_path"]
    app.state.vram_path = tmp_marshal_paths["vram_path"]
    return app


@pytest.fixture
async def marshal_app(
    marshal_config: MarshalConfig, tmp_marshal_paths: dict[str, Path]
) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    """Run a marshal FastAPI app in-process and yield (client, app).

    The app's lifespan starts the scheduler, memory poller, and
    registry; on teardown it stops them and unloads any models marshal
    owns (because shutdown.unload_models=True in the test config).

    The yielded client is an httpx.AsyncClient bound to the app via
    ASGITransport — no real socket. Test code uses it the same way it
    would use a client against a live marshal:

        async def test_x(marshal_app):
            client, app = marshal_app
            r = await client.post("/api/chat", json={...},
                                  headers={"X-Program-ID": PROGRAM_CRITICAL})

    To inspect marshal's internal state (e.g. ``_allocated_num_ctx``),
    tests read ``app.state._marshal_internals.scheduler``,
    ``.memory``, ``.registry``, ``.lifecycle``, ``.queues`` —
    a SimpleNamespace stashed there at the end of lifespan startup
    (small additive production change in server.py). The underscore
    prefix signals "test-only — not part of the public app surface";
    production request handlers continue using module globals.

    Registry/metrics paths are wired through ``app.state.*`` attributes
    that the lifespan reads at startup. Without this, the test marshal
    would write to ``~/.ollama-marshal/model_sizes.json`` and clobber
    the user's production-marshal registry cache.
    """
    app = make_test_app(marshal_config, tmp_marshal_paths)

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client, app


async def wait_for(
    condition: Any,
    *,
    timeout: float = 10.0,
    interval: float = 0.1,
    description: str = "condition",
) -> None:
    """Poll ``condition`` (a sync or async callable) until True or timeout.

    Used by tests waiting on async side-effects (model loaded,
    metric incremented, audit record written). Fails the test with a
    clear message if the condition doesn't hold within ``timeout``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        result = condition()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        await asyncio.sleep(interval)
    pytest.fail(f"timed out after {timeout}s waiting for {description}")


# ---------------------------------------------------------------------------
# Subprocess fixture infrastructure (v0.6.0+)
# ---------------------------------------------------------------------------
#
# Tests marked ``@pytest.mark.marshal_subprocess`` spawn a real
# ``ollama-marshal start`` subprocess on an OS-assigned ephemeral port,
# letting tests exercise the same wire format prod operators see (real
# socket I/O, real header parsing, real audit log writes) without
# touching the user's prod marshal at :11435.
#
# Tests marked ``@pytest.mark.marshal_prod`` hit the live prod marshal
# at :11435 directly via ``prod_marshal_client``. The session-scoped
# ``pause_local_prod_marshal`` fixture (autouse, v0.6.3+) handles the
# pause via ``POST /api/marshal/admin/pause`` so EVERY integration
# test runs without competing for Ollama VRAM with prod marshal.
# Requires admin/bypass tokens — discovered from the
# ``MARSHAL_TEST_ADMIN_TOKEN`` / ``MARSHAL_TEST_BYPASS_TOKEN`` env
# vars, or read directly from ``~/.ollama-marshal/admin-tokens.env``
# if the operator hasn't sourced them. Set
# ``MARSHAL_INTEGRATION_SKIP_PROD_PAUSE=1`` to opt out of the autouse
# pause (e.g. when intentionally exercising contention).

# Test bypass token used by both subprocess fixture and prod-pause
# fixture. Subprocess marshals get this hardcoded into their per-test
# config so test traffic flows even when the subprocess itself is
# paused (rare but possible during fixture teardown races).
SUBPROCESS_BYPASS_TOKEN = "subprocess-test-bypass-token"  # noqa: S105 — test fixture
SUBPROCESS_ADMIN_TOKEN = "subprocess-test-admin-token"  # noqa: S105 — test fixture


def _reserve_ephemeral_port() -> int:
    """Bind to port 0 and read back the OS-assigned port number.

    Brief race window between socket close and subprocess bind is
    acceptable for local-dev test execution. macOS/Linux ephemeral
    range avoids well-known service ports.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


async def _wait_for_marshal_ready(base_url: str, timeout: float = 20.0) -> None:
    """Poll ``GET <base_url>/api/marshal/status`` until 200 or timeout.

    Subprocess startup involves loading the config file, importing the
    package, and binding the proxy port — typically 1-3s but can be
    longer on first run when uv resolves the venv. ``timeout`` defaults
    to 20s for cold-start headroom.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_error: Exception | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"{base_url}/api/marshal/status")
                if resp.status_code == 200:
                    return
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
        await asyncio.sleep(0.1)
    msg = (
        f"marshal subprocess at {base_url} not ready within {timeout}s "
        f"(last error: {last_error!r})"
    )
    raise TimeoutError(msg)


def build_test_config_yaml(
    *,
    port: int,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    audit_path: Path | None = None,
    registry_path: Path | None = None,
    metadata_path: Path | None = None,
    metrics_path: Path | None = None,
    audit_enabled: bool = True,
    idle_eviction_minutes: int = 0,
    extra_programs: dict[str, str] | None = None,
) -> str:
    """Build a marshal.yaml string for a subprocess marshal.

    Pre-populates the test programs (integration-test CRITICAL,
    integration-test-normal NORMAL), the admin endpoints (so test
    fixtures can pause the subprocess if needed), and the debug
    endpoint (for tests asserting on internal state via
    /api/marshal/debug). Runs with ``benchmark_on_startup=false`` so
    per-test temp registry paths don't trigger the model-size sweep.

    Args:
        port: Ephemeral port the subprocess should bind to.
        ollama_host: Upstream Ollama URL.
        audit_path / registry_path / metadata_path / metrics_path:
            Per-test temp paths so the subprocess doesn't clobber
            the user's prod marshal state.
        audit_enabled: Default True for audit-log assertion tests;
            tests that don't read audit can pass False to skip the
            file writes.
        idle_eviction_minutes: Default 0 disables idle eviction.
            Tests of idle eviction set this to 1.
        extra_programs: Extra ``program_id -> priority`` entries
            beyond the default integration-test programs.

    Returns:
        YAML config string ready to be written to a tmp_path file
        and passed to ``ollama-marshal start --config <path>``.
    """
    lines = [
        "ollama:",
        f"  host: {ollama_host}",
        "proxy:",
        '  host: "127.0.0.1"',
        f"  port: {port}",
        "memory:",
        "  poll_interval: 1",
        "scheduler:",
        "  benchmark_on_startup: false",
        "  metrics_persist_interval_s: 3600",
        f"  ollama_forward_timeout_s: {INTEGRATION_FORWARD_TIMEOUT_S}",
        f"  idle_eviction_minutes: {idle_eviction_minutes}",
        # Keep the subprocess marshal's metrics off the user's real
        # ~/.ollama-marshal. MARSHAL_STATE_DIR isolates the registry files
        # but metrics_path is its own config key, saved unconditionally on
        # shutdown, so it must be redirected here too.
        *(
            [f"  metrics_path: {str(metrics_path)!r}"]
            if metrics_path is not None
            else []
        ),
        "shutdown:",
        '  mode: "immediate"',
        "  drain_timeout: 5",
        "  unload_models: true",
        "audit:",
        f"  enabled: {str(audit_enabled).lower()}",
    ]
    if audit_path is not None:
        lines.append(f"  path: {str(audit_path)!r}")
    lines.append("admin:")
    lines.append("  pause_endpoints_enabled: true")
    lines.append(f"  admin_token: {SUBPROCESS_ADMIN_TOKEN!r}")
    lines.append(f"  test_bypass_token: {SUBPROCESS_BYPASS_TOKEN!r}")
    lines.append("debug:")
    lines.append("  endpoint_enabled: true")
    lines.append("programs:")
    lines.append("  default:")
    lines.append('    priority: "normal"')
    lines.append(f"  {PROGRAM_CRITICAL}:")
    lines.append('    priority: "critical"')
    lines.append(f"  {PROGRAM_NORMAL}:")
    lines.append('    priority: "normal"')
    if extra_programs:
        for prog_id, priority in extra_programs.items():
            lines.append(f"  {prog_id}:")
            lines.append(f"    priority: {priority!r}")
    return "\n".join(lines) + "\n"


@pytest.fixture
async def marshal_subprocess(
    tmp_path: Path,
) -> AsyncIterator[tuple[str, Path]]:
    """Spawn a real ``ollama-marshal start`` subprocess on an ephemeral port.

    Yields ``(base_url, audit_path)`` so tests can hit the subprocess
    via httpx AND read the audit file directly to assert on what was
    recorded. Per-test temp paths isolate state from the user's prod
    marshal at ``~/.ollama-marshal/``.

    Subprocess teardown sends SIGTERM with a 5s grace period, then
    SIGKILL fallback. This guarantees the subprocess never leaks past
    test exit, even if the test crashed mid-assertion.
    """
    port = _reserve_ephemeral_port()
    base_url = f"http://127.0.0.1:{port}"
    audit_path = tmp_path / "audit.jsonl"
    config_path = tmp_path / "marshal.yaml"
    config_path.write_text(
        build_test_config_yaml(
            port=port,
            audit_path=audit_path,
            registry_path=tmp_path / "registry.json",
            metadata_path=tmp_path / "metadata.json",
            metrics_path=tmp_path / "metrics.json",
        )
    )

    proc = subprocess.Popen(  # noqa: S603 — args are hardcoded + tmp path
        [
            sys.executable,
            "-m",
            "ollama_marshal",
            "start",
            "--config",
            str(config_path),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        # MARSHAL_STATE_DIR redirects the registry's on-disk state
        # (model_sizes/metadata/vram.json) into the per-test tmp dir. The
        # subprocess can't take the in-process ``app.state.*_path``
        # overrides, so without this the subprocess marshal — and the M1
        # self-learning feedback loop in particular — would write measured
        # VRAM into the user's real ``~/.ollama-marshal``.
        env={**os.environ, "MARSHAL_STATE_DIR": str(tmp_path)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,  # avoid PIPE deadlock if marshal logs >64KB
    )

    try:
        await _wait_for_marshal_ready(base_url, timeout=20.0)
        yield base_url, audit_path
    finally:
        proc.terminate()
        # Hoist the blocking proc.wait off the event loop per CLAUDE.md
        # async correctness rules — without this the fixture's finally
        # block stalls the loop for up to 5s while the subprocess shuts
        # down (blocks any concurrent fixture cleanup).
        try:
            await asyncio.to_thread(proc.wait, 5)
        except subprocess.TimeoutExpired:
            proc.kill()
            await asyncio.to_thread(proc.wait)


@pytest.fixture
async def marshal_subprocess_client(
    marshal_subprocess: tuple[str, Path],
) -> AsyncIterator[tuple[httpx.AsyncClient, Path]]:
    """``httpx.AsyncClient`` pointing at the subprocess marshal.

    Yields ``(client, audit_path)`` matching the
    ``marshal_subprocess`` fixture's tuple. The client carries the
    subprocess bypass token by default so requests dispatch even
    if a prior test left the subprocess paused (defense in depth).
    """
    base_url, audit_path = marshal_subprocess
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=900,
        headers={
            "X-Marshal-Test-Bypass": SUBPROCESS_BYPASS_TOKEN,
        },
    ) as client:
        yield client, audit_path


# ---------------------------------------------------------------------------
# Prod-marshal pause fixtures
# ---------------------------------------------------------------------------
#
# Two layers, both targeting the same v0.6.0 admin/pause endpoint:
#
# 1. ``pause_local_prod_marshal`` — session-scoped autouse, fires for
#    EVERY integration test. Pauses local prod marshal at :11435 (if
#    reachable, with a discoverable admin token) for the duration of
#    the suite, then resumes. Prevents cross-suite contamination
#    where prod marshal's loaded models compete with test marshals
#    for Ollama VRAM, causing flaky integration test failures.
#    Degrades to a no-op (does NOT skip tests) when prod isn't
#    reachable, no token is discoverable, or the pause call fails.
#    Honor the ``MARSHAL_INTEGRATION_SKIP_PROD_PAUSE=1`` env var to
#    opt out (e.g. when intentionally exercising contention).
#
# 2. ``prod_marshal_pause`` — opt-in fixture for Path A tests that
#    hit prod marshal directly via ``prod_marshal_client``. Skips
#    the dependent tests when pause didn't take effect.


# Default prod marshal URL — both fixtures target the user's running
# marshal directly. Override via env var if the operator runs marshal
# on a non-default port.
PROD_MARSHAL_URL = os.environ.get("MARSHAL_TEST_PROD_URL", "http://localhost:11435")

# Manual escape hatch — set to ``1`` to skip the autouse pause (e.g.
# when intentionally exercising contention with prod, or running on
# a machine where the operator hasn't configured an admin token).
SKIP_PAUSE_ENV = "MARSHAL_INTEGRATION_SKIP_PROD_PAUSE"

# Auto-resume failsafe — set to one hour so a long integration run
# (cold-start model loads + memory_behavior tests) doesn't trip the
# failsafe mid-suite and re-introduce the v0.5.0 contamination bug
# this fixture exists to prevent. Operators relying on a tighter
# value can set MARSHAL_TEST_PAUSE_TIMEOUT_S explicitly.
PAUSE_AUTO_RESUME_S = int(os.environ.get("MARSHAL_TEST_PAUSE_TIMEOUT_S", "3600"))

# Drain timeout — DEFAULT 0 (don't wait for in-flight to drain).
#
# DESIGN DECISION (v0.6.3, reaffirmed v0.6.6 — do not re-litigate without
# evidence). Pause flag is set server-side immediately; in-flight
# inferences complete naturally while the suite runs. Empirically:
# a non-zero drain timeout (e.g. 60s) blocks fixture setup waiting
# for prod's in-flight inferences to finish, which on a busy machine
# adds 60s of cold start AND introduced new flakes in multi-instance
# routing tests. The zero-drain path is faster (suite went from 6:13
# to 2:34) AND eliminated 4 multi-instance test failures.
#
# Trade-off this opens: in-flight prod inferences keep using Ollama
# VRAM until they naturally finish (~5-30s, longer for big context),
# competing with test marshal's preload + chat dispatches. We absorb
# this in the integration tests by setting a 15-minute Hop 2 budget
# (``INTEGRATION_FORWARD_TIMEOUT_S = 900`` at module top) so VRAM-
# contended chats complete within budget instead of ReadTimeout-ing
# at 120s.
#
# Operators who want a stricter drain (e.g. before destructive admin
# work) can set MARSHAL_TEST_PAUSE_DRAIN_S explicitly.
PAUSE_DRAIN_TIMEOUT_S = int(os.environ.get("MARSHAL_TEST_PAUSE_DRAIN_S", "0"))

# Auto-restart prod marshal (v0.6.7) — see CLAUDE.md "Integration test
# infrastructure design decisions". Two layers:
#
# - Option B (always-on floor): ``pause_local_prod_marshal`` reads prod
#   marshal's ``version`` (now on /api/marshal/status) and WARNS when it
#   differs from the test marshal's ``ollama_marshal.__version__``. Zero
#   outage — catches the stale-prod-code skew that silently masked the
#   ``paused``-field verification during Bug 12 (2026-05-15).
# - Option A (opt-in via MARSHAL_TEST_RESTART_PROD=1): launchctl-restart
#   prod marshal pre-suite so it runs freshly-installed code. Graceful
#   fallback (no-op + warn) on non-launchd hosts or unregistered labels.
#
# Option C (fail-fast / skip pause on mismatch) was rejected — it would
# surrender the VRAM-contention protection the autouse pause exists for.
PROD_MARSHAL_RESTART_ENV = "MARSHAL_TEST_RESTART_PROD"
# launchd label for the prod marshal service; restart target is
# gui/<uid>/<label>. Override for non-default service names.
PROD_MARSHAL_LAUNCHD_LABEL = os.environ.get(
    "MARSHAL_TEST_LAUNCHD_LABEL", "com.ollama-marshal"
)
# Drain window (seconds) before the abrupt ``launchctl kickstart -k``.
PROD_RESTART_DRAIN_S = int(os.environ.get("MARSHAL_TEST_RESTART_DRAIN_S", "10"))

_log = structlog.get_logger("integration.prod_pause")


def _prod_marshal_reachable() -> bool:
    """Sync probe — does prod marshal respond at PROD_MARSHAL_URL?"""
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{PROD_MARSHAL_URL}/api/marshal/status")
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def _prod_pause(token: str) -> bool:
    """Send pause request to prod marshal; True if pause flag is in effect.

    Server returns:
    - 200: drain completed AND pause flag set
    - 409: drain timed out BUT pause flag is still set (one or more
      inferences are mid-stream, but no NEW dispatch happens). Per
      ``Scheduler.pause()`` docstring: "The pause flag is set in
      either case — a False return is informational, not a failure
      to pause." So 409 is ALSO success from the fixture's
      perspective — pause is in effect, resume is required on
      teardown.
    - any other: bypass disabled, auth fail, or other server error
      — treat as no-pause and skip.

    After the pause call returns "paused per server", v0.6.6+ does an
    independent verification by polling ``/api/marshal/status`` for
    ``paused: True``. Without this verify-step a partial pause (e.g.
    test bypass routes new requests but the underlying flag never
    actually toggled, or the admin endpoint returned success but
    state didn't propagate) would silently let prod workloads keep
    contending for Ollama VRAM during the test run.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{PROD_MARSHAL_URL}/api/marshal/admin/pause",
                headers={"X-Marshal-Admin-Token": token},
                json={
                    "drain_timeout_s": PAUSE_DRAIN_TIMEOUT_S,
                    "auto_resume_after_seconds": PAUSE_AUTO_RESUME_S,
                },
            )
        except httpx.HTTPError as exc:
            _log.warning("prod_pause.network_error", error=str(exc))
            return False
    if resp.status_code == 409:
        _log.warning(
            "prod_pause.drain_timeout_but_paused",
            in_flight=resp.json().get("in_flight"),
            drain_timeout_s=PAUSE_DRAIN_TIMEOUT_S,
        )
    elif resp.status_code != 200:
        _log.warning("prod_pause.unexpected_status", status=resp.status_code)
        return False

    # Verify the flag actually took effect by reading the canonical
    # status payload. ``paused`` was added to /api/marshal/status in
    # v0.6.6 specifically so this check can run without holding the
    # admin token.
    return await _verify_paused()


async def _verify_paused() -> bool:
    """Poll /api/marshal/status until ``paused: True`` or short timeout.

    Returns True if pause was confirmed in effect; False otherwise.
    Loudly logs on failure so a silent no-op doesn't slip past.

    Per-attempt timeout (2s) is intentionally below the overall deadline
    (15s) so a single slow status response doesn't burn the whole budget.
    The status payload calls into psutil + queue + memory polling and can
    legitimately stall a few seconds on a contended box — hence the
    generous 15s overall window with fast retries. The structured log on
    failure distinguishes "field absent" (operator must upgrade prod
    marshal to v0.6.6+) from "field present but False" (real flag-stuck
    case worth investigating) so operators don't have to read prose.
    """
    deadline = asyncio.get_event_loop().time() + 15.0
    last_payload: dict[str, Any] | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(f"{PROD_MARSHAL_URL}/api/marshal/status")
                resp.raise_for_status()
                last_payload = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                _log.warning("prod_pause.verify_status_unreachable", error=str(exc))
                await asyncio.sleep(0.2)
                continue
            paused_field = last_payload.get("paused") if last_payload else None
            if paused_field is True:
                return True
            await asyncio.sleep(0.2)
    last = last_payload or {}
    field_present = "paused" in last
    _log.warning(
        "prod_pause.verify_failed",
        field_present=field_present,
        paused_field=last.get("paused"),
        reason=(
            "prod marshal lacks the ``paused`` field — upgrade to v0.6.6+"
            if not field_present
            else "dispatch flag did not toggle within 15s — investigate"
        ),
    )
    return False


async def _prod_resume(token: str) -> None:
    """Send resume request to prod marshal; log + best-effort, never raise."""
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{PROD_MARSHAL_URL}/api/marshal/admin/resume",
                headers={"X-Marshal-Admin-Token": token},
            )
        except httpx.HTTPError as exc:
            _log.warning(
                "prod_pause.resume_failed",
                error=str(exc),
                will_auto_resume_in_s=PAUSE_AUTO_RESUME_S,
            )
            return
        if resp.status_code != 200:
            _log.warning(
                "prod_pause.resume_non_200",
                status=resp.status_code,
                will_auto_resume_in_s=PAUSE_AUTO_RESUME_S,
            )


async def _read_prod_marshal_version() -> str | None:
    """GET prod ``/api/marshal/status`` and return its ``version`` field.

    Returns ``None`` when the status is unreachable, malformed, or lacks
    the ``version`` field (prod marshal predates v0.6.7). Token-free —
    ``version`` rides the canonical status payload.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{PROD_MARSHAL_URL}/api/marshal/status")
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        _log.warning("prod_pause.version_read_failed", error=str(exc))
        return None
    # Tolerate a 200 with valid-but-non-object JSON (null, [], "error"):
    # ``.get`` on a non-dict raises AttributeError, which would escape this
    # helper into the autouse fixture and break the whole integration suite.
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    return version if isinstance(version, str) else None


async def _warn_on_version_skew() -> None:
    """Log a warning when prod marshal's version differs from the test's.

    Option B floor (always-on, zero outage): a status read + pure compare.
    A skew means prod and test marshals are on different code (either side
    may be the stale one), so prod's behavior may not match what the suite
    expects.
    """
    reason = version_skew_reason(await _read_prod_marshal_version(), __version__)
    if reason is not None:
        _log.warning(
            "prod_pause.version_skew",
            reason=reason,
            hint=(
                "versions differ (either side may be stale); restart the "
                f"stale marshal — {PROD_MARSHAL_RESTART_ENV}=1 restarts prod"
            ),
        )


async def _wait_prod_healthy(timeout: float = 30.0) -> bool:
    """Poll prod ``/api/marshal/status`` until 200 or ``timeout``.

    Used after a ``launchctl`` restart to confirm prod came back up.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"{PROD_MARSHAL_URL}/api/marshal/status")
                if resp.status_code == 200:
                    return True
        except (httpx.HTTPError, OSError):
            pass
        await asyncio.sleep(0.3)
    return False


def _kickstart_launchd(launchctl: str, target: str) -> subprocess.CompletedProcess[str]:
    """Blocking ``launchctl kickstart -k <target>`` (call via ``asyncio.to_thread``).

    Isolated so the synchronous ``subprocess.run`` (and its S603 suppression)
    lives in one place and the caller can offload it off the event loop with
    a bounded timeout. Raises ``subprocess.TimeoutExpired`` (a
    ``SubprocessError``) on hang and ``OSError`` on spawn failure — both
    handled by the caller as a graceful no-op.
    """
    return subprocess.run(  # noqa: S603 — resolved argv; label from env/default
        [launchctl, "kickstart", "-k", target],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


async def _maybe_restart_prod_marshal(token: str | None) -> None:
    """Opt-in (``MARSHAL_TEST_RESTART_PROD=1``): launchctl-restart prod marshal.

    Restarts the launchd-managed prod marshal so it runs freshly-installed
    code before the suite, eliminating the stale-process version skew that
    motivated this fixture. Best-effort and graceful: no-op + warn when
    ``launchctl`` is unavailable (Linux dev box) or the launchd label is
    not registered (manual marshal launch).

    Sequence: best-effort drain via ``admin/pause`` (so in-flight Ollama
    inferences aren't killed mid-stream) -> ``launchctl kickstart -k`` ->
    wait for the restarted marshal to report healthy.
    """
    launchctl = shutil.which("launchctl")
    if launchctl is None:
        _log.warning(
            "prod_restart.launchctl_unavailable",
            reason="not a launchd platform (Linux?) — skipping opt-in restart",
        )
        return
    # Best-effort drain before the abrupt ``kickstart -k`` so in-flight
    # inferences stop accruing new work first. Non-fatal on error —
    # kickstart restarts regardless.
    if token:
        try:
            async with httpx.AsyncClient(
                timeout=float(PROD_RESTART_DRAIN_S) + 5.0
            ) as client:
                await client.post(
                    f"{PROD_MARSHAL_URL}/api/marshal/admin/pause",
                    headers={"X-Marshal-Admin-Token": token},
                    json={
                        "drain_timeout_s": PROD_RESTART_DRAIN_S,
                        # Arm the long (1h) failsafe so a kickstart failure
                        # below can't leave prod frozen for only the server
                        # default window; we also resume explicitly on failure.
                        "auto_resume_after_seconds": PAUSE_AUTO_RESUME_S,
                    },
                )
        except httpx.HTTPError as exc:
            _log.warning("prod_restart.predrain_failed", error=str(exc))
    target = f"gui/{os.getuid()}/{PROD_MARSHAL_LAUNCHD_LABEL}"
    # Run the blocking kickstart off the event loop with a hard timeout, and
    # treat spawn errors / timeouts as a graceful no-op. The session-scoped
    # autouse fixture must never hang or raise.
    try:
        proc = await asyncio.to_thread(_kickstart_launchd, launchctl, target)
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning("prod_restart.kickstart_error", target=target, error=str(exc))
        if token:
            # We may have paused prod above — don't leave it frozen.
            await _prod_resume(token)
        return
    if proc.returncode != 0:
        _log.warning(
            "prod_restart.kickstart_failed",
            target=target,
            returncode=proc.returncode,
            stderr=proc.stderr.strip(),
            hint="launchd label not registered? set MARSHAL_TEST_LAUNCHD_LABEL",
        )
        # Restart did not happen: prod is still alive and (if we drained)
        # paused. Resume so a failed opt-in restart never freezes prod.
        if token:
            await _prod_resume(token)
        return
    if await _wait_prod_healthy():
        _log.info("prod_restart.done", target=target)
    else:
        _log.warning("prod_restart.unhealthy_after_restart", target=target)


@pytest_asyncio.fixture(loop_scope="session", scope="session", autouse=True)
async def pause_local_prod_marshal() -> AsyncIterator[bool]:
    """Pause local prod marshal at :11435 for the integration session.

    Yields ``True`` if prod was paused, ``False`` if the fixture
    no-op'd. The autouse flag means EVERY test in the integration
    suite picks this up — preventing the cross-suite VRAM contention
    that has been blocking the suite since v0.5.0.

    No-op (yields False, does NOT skip tests) when:

    - ``MARSHAL_INTEGRATION_SKIP_PROD_PAUSE=1`` is set
    - Prod marshal isn't reachable at ``PROD_MARSHAL_URL``
    - No admin token is discoverable (env var or
      ``~/.ollama-marshal/admin-tokens.env``)
    - The pause endpoint returns non-200 (drain timeout, 401, 404
      because admin endpoints are disabled, etc.)
    - The pause request raises ``httpx.HTTPError`` (network blip)

    Pause/resume use separate short-lived httpx clients so the
    request lifetime is decoupled from the session-long event loop.
    Auto-resume failsafe at ``PAUSE_AUTO_RESUME_S`` (default 1h)
    protects against pytest crashes leaving prod paused indefinitely.
    Operators don't need to remember to source the tokens file —
    ``discover_admin_token`` reads it directly when env is unset.
    """
    if os.environ.get(SKIP_PAUSE_ENV) == "1":
        _log.info("prod_pause.skipped_via_env", env_var=SKIP_PAUSE_ENV)
        yield False
        return
    if not _prod_marshal_reachable():
        _log.info("prod_pause.not_reachable", url=PROD_MARSHAL_URL)
        yield False
        return

    token = discover_admin_token()

    # Option A (opt-in via MARSHAL_TEST_RESTART_PROD=1): restart prod marshal
    # so it runs freshly-installed code before the suite. Best-effort +
    # graceful (no-op + warn on non-launchd hosts / unregistered labels).
    if os.environ.get(PROD_MARSHAL_RESTART_ENV) == "1":
        await _maybe_restart_prod_marshal(token)

    # Option B (always-on floor): warn if prod marshal reports a different
    # version than the test marshal. Runs after any opt-in restart so it
    # reflects the post-restart version. Zero outage.
    await _warn_on_version_skew()

    if not token:
        _log.info("prod_pause.no_admin_token")
        yield False
        return

    if not await _prod_pause(token):
        _log.info("prod_pause.pause_call_failed")
        yield False
        return

    _log.info(
        "prod_pause.engaged",
        url=PROD_MARSHAL_URL,
        drain_timeout_s=PAUSE_DRAIN_TIMEOUT_S,
        auto_resume_after_seconds=PAUSE_AUTO_RESUME_S,
    )
    try:
        yield True
    finally:
        await _prod_resume(token)
        _log.info("prod_pause.released", url=PROD_MARSHAL_URL)


@pytest_asyncio.fixture(loop_scope="session", scope="session")
async def prod_marshal_pause(
    pause_local_prod_marshal: bool,
) -> None:
    """Adapter for Path A tests — skips when autouse pause didn't take.

    Path A tests fire requests against the live prod marshal via
    ``prod_marshal_client``. They require pause to be in effect AND
    a bypass token to be discoverable. The autouse fixture above
    handles the actual pause/resume; this adapter just decides
    whether the dependent tests should run.
    """
    if pause_local_prod_marshal:
        return
    if os.environ.get(SKIP_PAUSE_ENV) == "1":
        pytest.skip(f"{SKIP_PAUSE_ENV}=1 — skipping prod-bound tests")
    if not discover_admin_token():
        pytest.skip(
            "MARSHAL_TEST_ADMIN_TOKEN not set and no readable "
            "~/.ollama-marshal/admin-tokens.env — skipping prod-bound tests"
        )
    if not _prod_marshal_reachable():
        pytest.skip(
            f"prod marshal not reachable at {PROD_MARSHAL_URL} — "
            "skipping prod-bound tests"
        )
    pytest.skip(
        "prod marshal pause failed (admin endpoints disabled, drain "
        "timeout, or transient error) — skipping prod-bound tests"
    )


@pytest.fixture
async def prod_marshal_client(
    prod_marshal_pause: None,
) -> AsyncIterator[httpx.AsyncClient]:
    """``httpx.AsyncClient`` pointing at the live prod marshal.

    Auto-depends on ``prod_marshal_pause`` so tests using this client
    only run while prod is paused. Default headers include the
    test-bypass token so the request flows during pause AND the
    integration-test program ID for CRITICAL priority routing.
    """
    bypass_token = discover_bypass_token()
    if not bypass_token:
        pytest.skip(
            "MARSHAL_TEST_BYPASS_TOKEN not set and no readable "
            "~/.ollama-marshal/admin-tokens.env — skipping prod-bound tests"
        )

    async with httpx.AsyncClient(
        base_url=PROD_MARSHAL_URL,
        timeout=900,
        headers={
            "X-Marshal-Test-Bypass": bypass_token,
            "X-Program-ID": PROGRAM_CRITICAL,
        },
    ) as client:
        yield client
