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
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from ollama_marshal.server import create_app

if TYPE_CHECKING:
    from fastapi import FastAPI

# The smallest model used across the integration suite. ~1.6 GB.
# Tests that need a different model parametrize the model name
# explicitly; this is the default for "I just need any model".
REQUIRED_MODEL = "qwen3.5:0.8b-bf16"

# Default Ollama host the suite expects to be reachable.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

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
        "audit_path": tmp_path / "audit.jsonl",
        "metrics_path": tmp_path / "metrics.json",
    }


@pytest.fixture
def marshal_config(tmp_marshal_paths: dict[str, Path]) -> MarshalConfig:
    """Test-tuned MarshalConfig wired to the per-test temp paths.

    Test defaults that differ from production:

    - ``proxy.request_timeout_s = 90`` — generous enough that cold
      first-loads on a busy machine (Ollama under memory pressure
      while another marshal is also using it) don't surface as
      timeouts. Bounded so stuck requests still fail.
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
        proxy=ProxyConfig(host="127.0.0.1", port=11436, request_timeout_s=90),
        memory=MemoryConfig(poll_interval=1),
        scheduler=SchedulerConfig(
            metrics_path=str(tmp_marshal_paths["metrics_path"]),
            metrics_persist_interval_s=3600,  # don't write during tests
            benchmark_on_startup=False,
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
# at :11435 directly, after the session-scoped ``prod_marshal_pause``
# fixture pauses it via ``POST /api/marshal/admin/pause``. Requires
# environment vars ``MARSHAL_TEST_ADMIN_TOKEN`` and
# ``MARSHAL_TEST_BYPASS_TOKEN`` to match prod marshal's config.

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
        "  request_timeout_s: 90",
        "memory:",
        "  poll_interval: 1",
        "scheduler:",
        "  benchmark_on_startup: false",
        "  metrics_persist_interval_s: 3600",
        f"  idle_eviction_minutes: {idle_eviction_minutes}",
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        await _wait_for_marshal_ready(base_url, timeout=20.0)
        yield base_url, audit_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
async def marshal_subprocess_factory(tmp_path: Path):
    """Factory for tests needing custom-config marshal subprocesses.

    Yields a callable ``spawn(**overrides)`` that writes a per-test
    marshal.yaml from ``build_test_config_yaml(**overrides)``, spawns
    the subprocess, waits for ready, and returns
    ``(base_url, audit_path)``. Tracks all spawned subprocesses for
    teardown so a test can spawn multiple if needed.

    Use when ``marshal_subprocess`` doesn't fit (different audit
    enabled, custom idle_eviction_minutes, extra programs, etc.).
    """
    spawned: list[subprocess.Popen[bytes]] = []
    spawn_index = [0]

    async def spawn(**overrides: Any) -> tuple[str, Path]:
        spawn_index[0] += 1
        suffix = f"_{spawn_index[0]}" if spawn_index[0] > 1 else ""
        port = _reserve_ephemeral_port()
        base_url = f"http://127.0.0.1:{port}"
        audit_path = tmp_path / f"audit{suffix}.jsonl"
        config_path = tmp_path / f"marshal{suffix}.yaml"
        config_path.write_text(
            build_test_config_yaml(
                port=port,
                audit_path=audit_path,
                registry_path=tmp_path / f"registry{suffix}.json",
                metadata_path=tmp_path / f"metadata{suffix}.json",
                metrics_path=tmp_path / f"metrics{suffix}.json",
                **overrides,
            )
        )
        proc = subprocess.Popen(  # noqa: S603 — args hardcoded + tmp path
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        spawned.append(proc)
        await _wait_for_marshal_ready(base_url, timeout=20.0)
        return base_url, audit_path

    try:
        yield spawn
    finally:
        for proc in spawned:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


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
        timeout=60,
        headers={
            "X-Marshal-Test-Bypass": SUBPROCESS_BYPASS_TOKEN,
        },
    ) as client:
        yield client, audit_path


# ---------------------------------------------------------------------------
# Prod-marshal pause fixture (Path A tests)
# ---------------------------------------------------------------------------


# Default prod marshal URL — Path A tests target the user's running
# marshal directly. Override via env var if the operator runs marshal
# on a non-default port.
PROD_MARSHAL_URL = os.environ.get("MARSHAL_TEST_PROD_URL", "http://localhost:11435")


def _prod_admin_token() -> str | None:
    """Read prod marshal admin token from environment.

    Returns None when not set so Path A tests can be skipped cleanly
    rather than failing with a confusing 401.
    """
    return os.environ.get("MARSHAL_TEST_ADMIN_TOKEN") or None


def _prod_bypass_token() -> str | None:
    """Read prod marshal test-bypass token from environment."""
    return os.environ.get("MARSHAL_TEST_BYPASS_TOKEN") or None


def _prod_marshal_reachable() -> bool:
    """Sync probe — does prod marshal respond at PROD_MARSHAL_URL?"""
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{PROD_MARSHAL_URL}/api/marshal/status")
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


@pytest.fixture(scope="session")
async def prod_marshal_pause() -> AsyncIterator[None]:
    """Pause the live prod marshal for the duration of Path A tests.

    Soft-pause: prod marshal stops dispatching from its queue but
    continues accepting new requests (no client-visible 503s). Path A
    tests fire requests with the test-bypass token so they dispatch
    even during the pause. After all Path A tests complete, the
    fixture calls resume so prod drains its accumulated queue.

    Skips the entire Path A test phase if:
    - Prod marshal isn't reachable at PROD_MARSHAL_URL
    - MARSHAL_TEST_ADMIN_TOKEN env var is unset
    - The pause endpoint returns 409 (drain timeout exceeded — likely
      a long-running inference in flight)

    The auto-resume failsafe (default 5min) ensures prod resumes
    automatically if this fixture's teardown fails to fire (e.g.
    test session crashed).
    """
    admin_token = _prod_admin_token()
    if not admin_token:
        pytest.skip("MARSHAL_TEST_ADMIN_TOKEN not set — skipping prod-bound tests")
    if not _prod_marshal_reachable():
        pytest.skip(
            f"prod marshal not reachable at {PROD_MARSHAL_URL} — "
            "skipping prod-bound tests"
        )

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{PROD_MARSHAL_URL}/api/marshal/admin/pause",
            headers={"X-Marshal-Admin-Token": admin_token},
            json={"drain_timeout_s": 60, "auto_resume_after_seconds": 600},
        )
        if resp.status_code == 409:
            pytest.skip(
                "prod marshal drain timed out (long inference in flight) — "
                "skipping prod-bound tests"
            )
        if resp.status_code != 200:
            pytest.skip(
                f"prod marshal pause returned {resp.status_code}: "
                f"{resp.text} — skipping prod-bound tests"
            )

        try:
            yield
        finally:
            await client.post(
                f"{PROD_MARSHAL_URL}/api/marshal/admin/resume",
                headers={"X-Marshal-Admin-Token": admin_token},
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
    bypass_token = _prod_bypass_token()
    if not bypass_token:
        pytest.skip("MARSHAL_TEST_BYPASS_TOKEN not set — skipping prod-bound tests")

    async with httpx.AsyncClient(
        base_url=PROD_MARSHAL_URL,
        timeout=60,
        headers={
            "X-Marshal-Test-Bypass": bypass_token,
            "X-Program-ID": PROGRAM_CRITICAL,
        },
    ) as client:
        yield client
