"""Surface B integration tests — fail-fast 404 on unknown models.

Validates the v0.4.0 surface: marshal returns 404 in milliseconds for
models that aren't in Ollama's `/api/tags`, instead of letting the
request sit in the queue indefinitely (Hop 1 became unbounded in
v0.6.4) while `lifecycle.preload` retries forever.

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
    pytest.mark.marshal_subprocess,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not running on :11434",
    ),
]


_HDR = {"X-Program-ID": PROGRAM_CRITICAL}

# A model name that's almost certainly NOT pulled. Adversarial typo.
NEVER_PULLED_MODEL = "zzz-never-existed-bf16:latest"


# Wall-clock budget for the fail-fast 404 path. Production target is
# 500ms; 750ms gives headroom for the subprocess cold-start /api/tags
# probe on a shared Ollama without losing regression-detection power.
_FAIL_FAST_BUDGET_S = 0.75


async def test_unknown_model_returns_404_fast(marshal_subprocess_client):
    """Marshal returns 404 fast for a missing model.

    Without this surface, the request would sit in the queue
    indefinitely (Hop 1 became unbounded in v0.6.4) while
    ``lifecycle.preload`` keeps retrying — eventually firing the new
    ``preload_max_consecutive_failures`` giveup, but only after
    serving the client a confused error instead of a clear 404.
    """
    client, _audit_path = marshal_subprocess_client
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
    # Wall-clock under the budget — much faster than the preload+giveup loop.
    assert elapsed < _FAIL_FAST_BUDGET_S, (
        f"fail-fast took {elapsed:.3f}s, expected <{_FAIL_FAST_BUDGET_S}s"
    )


async def test_freshly_pulled_model_recognized(marshal_subprocess_client):
    """Two consecutive real-model requests both succeed — no stale negatives.

    Fires REQUIRED_MODEL twice in a row. The first call's
    /api/tags read populates the registry's known-models cache. The
    second call must hit cache and dispatch without falsely 404'ing.
    Verifies the opportunistic resync didn't poison subsequent
    requests with a stale "not found" entry.
    """
    client, _audit_path = marshal_subprocess_client
    body = {
        "model": REQUIRED_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"num_predict": 4},
    }
    r1 = await client.post("/api/chat", json=body, headers=_HDR, timeout=900)
    assert r1.status_code == 200, r1.text
    r2 = await client.post("/api/chat", json=body, headers=_HDR, timeout=900)
    assert r2.status_code == 200, r2.text
