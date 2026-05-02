"""Surface C1 Dim 1 integration tests — prompt-driven num_ctx sizing.

Validates that marshal's num_ctx injection actually flows through to
Ollama's slot allocation. Reads marshal's allocated num_ctx via the
v0.6.0 ``/api/marshal/debug`` endpoint
(``memory.allocated_num_ctx_per_model``) — that's the value marshal
told ``lifecycle.preload`` to use, which is the value Ollama
allocated.

Migrated to subprocess pattern (v0.6.1+).
"""

from __future__ import annotations

import httpx
import pytest

from tests.integration.conftest import (
    DEFAULT_OLLAMA_HOST,
    PROGRAM_CRITICAL,
    REQUIRED_MODEL,
    _ollama_reachable,
    wait_for,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.marshal_subprocess,
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


async def _fire_chat(
    client: httpx.AsyncClient,
    *,
    prompt: str,
    num_ctx: int | None = None,
) -> None:
    body: dict = {
        "model": REQUIRED_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    options = {"num_predict": 4}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    body["options"] = options
    resp = await client.post("/api/chat", json=body, headers=_HDR, timeout=60)
    assert resp.status_code == 200, resp.text


async def _allocated_num_ctx(client: httpx.AsyncClient, model: str) -> int | None:
    """Fetch marshal's allocated num_ctx for ``model`` via /api/marshal/debug."""
    resp = await client.get("/api/marshal/debug")
    resp.raise_for_status()
    body = resp.json()
    return body["memory"]["allocated_num_ctx_per_model"].get(model)


async def _model_max_ctx(client: httpx.AsyncClient, model: str) -> int | None:
    """Fetch model's max_context_length from registry metadata via debug."""
    resp = await client.get("/api/marshal/debug")
    resp.raise_for_status()
    body = resp.json()
    meta = body["registry"]["metadata_per_model"].get(model)
    return meta["max_context_length"] if meta else None


async def _reload_count(client: httpx.AsyncClient) -> int:
    resp = await client.get("/api/marshal/debug")
    resp.raise_for_status()
    return int(resp.json()["metrics"]["reload_count"])


@_REQUIRES_MODEL
async def test_short_prompt_gets_smallest_boundary(marshal_subprocess_client):
    """A trivial prompt rounds up to the smallest power-of-2 (8192).

    Token math: prompt "hi" ≈ 0 tokens, + default completion budget
    4096, + safety 256 = 4352. Rounds up to 8192.
    """
    client, _audit_path = marshal_subprocess_client
    await _fire_chat(client, prompt="hi")
    await wait_for(
        lambda: _has_allocation(client, REQUIRED_MODEL),
        timeout=15,
        description="allocation recorded",
    )
    allocated = await _allocated_num_ctx(client, REQUIRED_MODEL)
    assert allocated == 8192, (
        f"expected 8192 for short prompt, got {allocated}. "
        f"Verify _resolve_num_ctx_decision rounds prompt+budget+safety "
        f"to the next power-of-2 boundary."
    )


async def _has_allocation(client: httpx.AsyncClient, model: str) -> bool:
    """wait_for-friendly predicate: True once allocation is recorded."""
    return (await _allocated_num_ctx(client, model)) is not None


@_REQUIRES_MODEL
async def test_long_prompt_gets_larger_boundary(marshal_subprocess_client):
    """A 50K-char prompt (~15K tokens) lands at a larger boundary.

    50_000 chars * 0.3 tokens/char = 15_000 + 4096 + 256 = 19_352.
    Next power-of-2 is 32768.
    """
    client, _audit_path = marshal_subprocess_client
    await _fire_chat(client, prompt="x" * 50_000)
    await wait_for(
        lambda: _has_allocation(client, REQUIRED_MODEL),
        timeout=15,
        description="allocation recorded",
    )
    allocated = await _allocated_num_ctx(client, REQUIRED_MODEL)
    assert allocated == 32768, f"expected 32768 for 50K-char prompt, got {allocated}"


@_REQUIRES_MODEL
async def test_client_num_ctx_clamped_to_model_max(marshal_subprocess_client):
    """Adversarial client num_ctx is clamped to model.max_context_length.

    Without this clamp, num_ctx=999_999_999 would trigger reload-on-need,
    fail preload, infinite-loop the scheduler, and grow reload_count
    unboundedly. One bad request would brick the proxy.
    """
    client, _audit_path = marshal_subprocess_client
    # qwen3.5:0.8b-bf16's max_context_length is 262144. The clamp
    # should bring 999_999_999 down to 262144.
    await _fire_chat(client, prompt="hi", num_ctx=999_999_999)
    await wait_for(
        lambda: _has_allocation(client, REQUIRED_MODEL),
        timeout=30,
        description="allocation recorded after clamp",
    )
    allocated = await _allocated_num_ctx(client, REQUIRED_MODEL)
    # Must be <= model max, NOT the absurd input value.
    max_ctx = await _model_max_ctx(client, REQUIRED_MODEL)
    assert max_ctx is not None, "registry never probed model metadata"
    assert allocated == max_ctx, (
        f"expected clamp to model max ({max_ctx}), got {allocated}. "
        f"The trust-boundary clamp didn't fire."
    )
    # And no reload loop — reload_count should be small + bounded,
    # NOT growing each iteration. Subprocess startup with empty
    # registry can produce 1-2 reloads naturally as the model loads
    # at default num_ctx and then re-loads at the clamped value once
    # the request's intended num_ctx is resolved. The original
    # assertion was <=1 against a warm in-process registry; <=3
    # accommodates the subprocess cold-start path while still
    # catching the failure mode (reload count blows up).
    reload_count = await _reload_count(client)
    assert reload_count <= 3, (
        f"reload_count grew to {reload_count} "
        f"— suggests the clamp failed and we entered a reload loop."
    )
