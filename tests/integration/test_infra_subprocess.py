"""Meta-tests for the subprocess fixture infrastructure (v0.6.0+).

Proves that the ``marshal_subprocess`` fixture actually:
- Spawns a real ``ollama-marshal start`` subprocess
- Binds to the assigned ephemeral port
- Becomes ready within the timeout (responds to /api/marshal/status)
- Tears down cleanly on test exit (subprocess terminates within grace)

These run against the user's real Ollama at :11434 (no mocks) and
require Ollama to be reachable. They use the integration marker so
they're excluded from default ``make test``.
"""

from __future__ import annotations

import httpx
import pytest

from tests.integration.conftest import (
    _ollama_reachable,
    _reserve_ephemeral_port,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.marshal_subprocess,
    pytest.mark.skipif(
        not _ollama_reachable(), reason="Ollama not reachable on :11434"
    ),
]


class TestEphemeralPort:
    """``_reserve_ephemeral_port`` returns a usable, distinct port."""

    def test_returns_port_in_ephemeral_range(self):
        port = _reserve_ephemeral_port()
        # macOS/Linux ephemeral range starts well above well-known
        # service ports. 1024 is the conservative floor.
        assert 1024 <= port <= 65535

    def test_two_calls_return_different_ports(self):
        # Each call creates a fresh socket, binds, reads back. The
        # OS shouldn't recycle the same port within microseconds.
        port_a = _reserve_ephemeral_port()
        port_b = _reserve_ephemeral_port()
        assert port_a != port_b


class TestMarshalSubprocessFixture:
    """``marshal_subprocess`` spawns + tears down a real marshal process."""

    async def test_subprocess_is_reachable(self, marshal_subprocess):
        base_url, _audit_path = marshal_subprocess
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/api/marshal/status")
        assert resp.status_code == 200

    async def test_subprocess_status_payload_shape(self, marshal_subprocess):
        base_url, _audit_path = marshal_subprocess
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/api/marshal/status")
        body = resp.json()
        # Top-level keys the lifespan promises.
        for key in ("uptime_seconds", "loaded_models", "memory", "instances"):
            assert key in body, f"missing key: {key}"

    async def test_debug_endpoint_enabled_in_subprocess(self, marshal_subprocess):
        """build_test_config_yaml sets debug.endpoint_enabled=true."""
        base_url, _audit_path = marshal_subprocess
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/api/marshal/debug")
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body
        assert "scheduler" in body
        assert body["scheduler"]["is_paused"] is False

    async def test_admin_endpoints_enabled_in_subprocess(self, marshal_subprocess):
        """build_test_config_yaml sets admin.pause_endpoints_enabled=true."""
        from tests.integration.conftest import SUBPROCESS_ADMIN_TOKEN

        base_url, _audit_path = marshal_subprocess
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Pause then resume so we exercise both endpoints.
            pause_resp = await client.post(
                f"{base_url}/api/marshal/admin/pause",
                headers={"X-Marshal-Admin-Token": SUBPROCESS_ADMIN_TOKEN},
                json={"drain_timeout_s": 5, "auto_resume_after_seconds": 60},
            )
            assert pause_resp.status_code == 200, pause_resp.text
            resume_resp = await client.post(
                f"{base_url}/api/marshal/admin/resume",
                headers={"X-Marshal-Admin-Token": SUBPROCESS_ADMIN_TOKEN},
            )
            assert resume_resp.status_code == 200, resume_resp.text


class TestMarshalSubprocessClientFixture:
    """The httpx client fixture wires headers correctly."""

    async def test_client_targets_subprocess(self, marshal_subprocess_client):
        client, _audit_path = marshal_subprocess_client
        resp = await client.get("/api/marshal/status")
        assert resp.status_code == 200

    async def test_client_carries_bypass_token_by_default(
        self, marshal_subprocess_client
    ):
        from tests.integration.conftest import SUBPROCESS_BYPASS_TOKEN

        client, _audit_path = marshal_subprocess_client
        # The client's default headers should include the bypass token.
        assert client.headers.get("x-marshal-test-bypass") == SUBPROCESS_BYPASS_TOKEN
