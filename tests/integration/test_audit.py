"""Audit log integration tests — JSONL records work end-to-end.

Validates:
- ``request.served`` events appear with correct fields after successful chats
- No prompt content leaks into JSONL records (privacy invariant)
- ``request.failed`` events fire on actual error paths

Each test uses an isolated marshal app with audit enabled and a
per-test temp path for the JSONL file.
"""

from __future__ import annotations

import asyncio
import json

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
    DEFAULT_OLLAMA_HOST,
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


def _required_model_pulled() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{DEFAULT_OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            return any(
                m.get("name") == REQUIRED_MODEL for m in resp.json().get("models", [])
            )
    except (httpx.HTTPError, OSError):
        return False


_REQUIRES_MODEL = pytest.mark.skipif(
    not _required_model_pulled(),
    reason=f"Required model {REQUIRED_MODEL!r} not pulled",
)


_HDR = {"X-Program-ID": PROGRAM_CRITICAL}


def _build_config(*, tmp_paths: dict, ollama_host: str) -> MarshalConfig:
    return MarshalConfig(
        ollama=OllamaConfig(host=ollama_host),
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
        audit=AuditConfig(
            enabled=True,
            path=str(tmp_paths["audit_path"]),
            retention_days=0,
            max_size_mb=0,
        ),
    )


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@_REQUIRES_MODEL
@pytest.mark.marshal_subprocess
async def test_request_served_events_recorded(marshal_subprocess_client):
    """5 successful requests → 5 ``request.served`` records with correct fields.

    Migrated to subprocess pattern (v0.6.1+). Audit file is the
    per-test temp path the fixture wires into the subprocess config.
    """
    client, audit_path = marshal_subprocess_client
    body = {
        "model": REQUIRED_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"num_predict": 4},
    }
    for _ in range(5):
        resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=60)
        assert resp.status_code == 200
    # AuditLogger flushes on a 100ms timer + final flush on shutdown.
    # Give it a beat AND let subprocess finalize before reading the
    # file. The marshal_subprocess fixture's teardown handles the
    # shutdown flush; here we just need the periodic flush to run.
    await asyncio.sleep(0.5)

    records = _read_jsonl(audit_path)
    served = [r for r in records if r.get("event") == "request.served"]
    assert len(served) == 5, f"expected 5 served records, got {len(served)}: {records}"
    for r in served:
        assert r["program_id"] == PROGRAM_CRITICAL
        assert r["model"] == REQUIRED_MODEL
        assert r["endpoint"] == "/api/chat"
        assert isinstance(r.get("wait_ms"), (int, float))
        assert r.get("stream") is False
        assert "ts" in r


@_REQUIRES_MODEL
@pytest.mark.marshal_subprocess
async def test_no_prompt_content_in_audit_records(marshal_subprocess_client):
    """Privacy invariant: prompt text NEVER appears in JSONL records.

    The audit module's ``record()`` signature already forbids a
    ``prompt`` kwarg, but this test verifies the actual file output —
    catches any accidental include via kwargs leak or future
    refactor.
    """
    client, audit_path = marshal_subprocess_client
    body = {
        "model": REQUIRED_MODEL,
        "messages": [{"role": "user", "content": "supersecretpassword12345"}],
        "stream": False,
        "options": {"num_predict": 4},
    }
    resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=60)
    assert resp.status_code == 200
    await asyncio.sleep(0.5)

    raw = audit_path.read_text()
    assert "supersecretpassword12345" not in raw, (
        f"prompt content leaked into audit log:\n{raw}"
    )


async def test_request_failed_event_on_error(tmp_marshal_paths):
    """Non-retryable error path produces a ``request.failed`` audit event."""
    async with fault_proxy() as proxy:
        # 500 is NOT in RETRYABLE_STATUS_CODES (502/503/504), so it
        # propagates to the client without retry. The serve path goes
        # through the success branch (forward_request returned a
        # response), so it actually emits request.served — NOT
        # request.failed. To force request.failed we need an exception
        # path. Use disconnect_next with retry disabled.
        proxy.disconnect_next("/api/chat", times=99)
        cfg = _build_config(tmp_paths=tmp_marshal_paths, ollama_host=proxy.url)
        # Disable retry so the disconnect surfaces as a request.failed
        # event without retry budget changing anything.
        cfg.retry = RetryConfig(enabled=False)
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
            resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=10)
            # Disconnect → marshal surfaces as 502 Bad Gateway. (Not 504,
            # which would imply a timeout — the connection actually
            # FAILED here, so 502 is the right code. /review tightened
            # this from `in (502, 504)` to exact 502 to catch any
            # future bug where marshal returns 504 for a hard
            # disconnect.)
            assert resp.status_code == 502, (
                f"expected 502 for disconnect, got {resp.status_code}"
            )
            await asyncio.sleep(0.5)

    records = _read_jsonl(tmp_marshal_paths["audit_path"])
    failed = [r for r in records if r.get("event") == "request.failed"]
    assert len(failed) >= 1, f"expected ≥1 failed record, got {records}"
    r = failed[0]
    assert r["program_id"] == PROGRAM_CRITICAL
    assert r["model"] == REQUIRED_MODEL
    assert r.get("error_type"), f"error_type missing/empty in {r}"
