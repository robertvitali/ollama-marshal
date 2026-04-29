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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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


def make_test_app(cfg: MarshalConfig, tmp_marshal_paths: dict[str, Path]) -> Any:
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
    """
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
