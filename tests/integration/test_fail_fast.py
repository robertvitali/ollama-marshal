"""Surface B integration tests — fail-fast 404 on unknown models.

Validates the v0.4.0 surface: marshal returns 404 in milliseconds for
models that aren't in Ollama's `/api/tags`, instead of letting the
request sit in the queue for `proxy.request_timeout_s` while
`lifecycle.preload` retries forever.

Both tests run against the user's live Ollama. The first uses a model
name that's almost certainly NOT pulled. The second exercises the
opportunistic resync path: fire two requests close together for
different real models — the second should NOT be falsely 404'd by a
stale negative cache.
"""

from __future__ import annotations

import time

import pytest

from tests.integration.conftest import (
    PROGRAM_CRITICAL,
    REQUIRED_MODEL,
    _ollama_reachable,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not running on :11434",
    ),
]


_HDR = {"X-Program-ID": PROGRAM_CRITICAL}

# A model name that's almost certainly NOT pulled. Adversarial typo.
NEVER_PULLED_MODEL = "zzz-never-existed-bf16:latest"


async def test_unknown_model_returns_404_fast(marshal_app):
    """Marshal returns 404 in <500ms wall-clock for a missing model.

    Without this surface, the request would sit in the queue for the
    proxy.request_timeout_s (default 1h) while lifecycle.preload
    retries every ~2min trying to load a model Ollama doesn't have.
    """
    client, _app = marshal_app
    body = {
        "model": NEVER_PULLED_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    start = time.perf_counter()
    resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=5)
    elapsed = time.perf_counter() - start

    assert resp.status_code == 404, resp.text
    payload = resp.json()
    # Error body should mention the offending model name + the fix.
    assert NEVER_PULLED_MODEL in payload.get("error", "")
    assert "ollama pull" in payload.get("error", "").lower()
    # Wall-clock <500ms — much faster than the request_timeout_s.
    assert elapsed < 0.5, f"fail-fast took {elapsed:.3f}s, expected <500ms"


async def test_freshly_pulled_model_recognized(marshal_app):
    """Two consecutive real-model requests both succeed — no stale negatives.

    Fires REQUIRED_MODEL twice in a row. The first call's
    /api/tags read populates the registry's known-models cache. The
    second call must hit cache and dispatch without falsely 404'ing.
    Verifies the opportunistic resync didn't poison subsequent
    requests with a stale "not found" entry.
    """
    client, _app = marshal_app
    body = {
        "model": REQUIRED_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"num_predict": 4},
    }
    r1 = await client.post("/api/chat", json=body, headers=_HDR, timeout=60)
    assert r1.status_code == 200, r1.text
    r2 = await client.post("/api/chat", json=body, headers=_HDR, timeout=60)
    assert r2.status_code == 200, r2.text
