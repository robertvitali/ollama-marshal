"""Surface C1 Dim 1 integration tests — prompt-driven num_ctx sizing.

Validates that marshal's num_ctx injection actually flows through to
Ollama's slot allocation. We can't easily peek at what marshal sent
on the wire, but we can read the resulting allocated slot via
``app.state.memory.get_allocated_num_ctx(model)`` — that's the value
marshal told ``lifecycle.preload`` to use, which is the value Ollama
allocated.
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


@_REQUIRES_MODEL
async def test_short_prompt_gets_smallest_boundary(marshal_app):
    """A trivial prompt rounds up to the smallest power-of-2 (8192).

    Token math: prompt "hi" ≈ 0 tokens, + default completion budget
    4096, + safety 256 = 4352. Rounds up to 8192.
    """
    client, app = marshal_app
    await _fire_chat(client, prompt="hi")
    await wait_for(
        lambda: app.state.memory.get_allocated_num_ctx(REQUIRED_MODEL) is not None,
        timeout=15,
        description="allocation recorded",
    )
    allocated = app.state.memory.get_allocated_num_ctx(REQUIRED_MODEL)
    assert allocated == 8192, (
        f"expected 8192 for short prompt, got {allocated}. "
        f"Verify _resolve_num_ctx_decision rounds prompt+budget+safety "
        f"to the next power-of-2 boundary."
    )


@_REQUIRES_MODEL
async def test_long_prompt_gets_larger_boundary(marshal_app):
    """A 50K-char prompt (~15K tokens) lands at a larger boundary.

    50_000 chars * 0.3 tokens/char = 15_000 + 4096 + 256 = 19_352.
    Next power-of-2 is 32768.
    """
    client, app = marshal_app
    await _fire_chat(client, prompt="x" * 50_000)
    await wait_for(
        lambda: app.state.memory.get_allocated_num_ctx(REQUIRED_MODEL) is not None,
        timeout=15,
        description="allocation recorded",
    )
    allocated = app.state.memory.get_allocated_num_ctx(REQUIRED_MODEL)
    assert allocated == 32768, f"expected 32768 for 50K-char prompt, got {allocated}"


@_REQUIRES_MODEL
async def test_client_num_ctx_clamped_to_model_max(marshal_app):
    """Adversarial client num_ctx is clamped to model.max_context_length.

    Without this clamp, num_ctx=999_999_999 would trigger reload-on-need,
    fail preload, infinite-loop the scheduler, and grow reload_count
    unboundedly. One bad request would brick the proxy.
    """
    client, app = marshal_app
    # qwen3.5:0.8b-bf16's max_context_length is 262144. The clamp
    # should bring 999_999_999 down to 262144.
    await _fire_chat(client, prompt="hi", num_ctx=999_999_999)
    await wait_for(
        lambda: app.state.memory.get_allocated_num_ctx(REQUIRED_MODEL) is not None,
        timeout=30,
        description="allocation recorded after clamp",
    )
    allocated = app.state.memory.get_allocated_num_ctx(REQUIRED_MODEL)
    # Must be <= model max, NOT the absurd input value.
    meta = app.state.registry.get_metadata(REQUIRED_MODEL)
    assert meta is not None, "registry never probed model metadata"
    assert allocated == meta.max_context_length, (
        f"expected clamp to model max ({meta.max_context_length}), "
        f"got {allocated}. The trust-boundary clamp didn't fire."
    )
    # And no reload loop — reload_count should be 0 or 1, NOT growing.
    assert app.state.scheduler.metrics.reload_count <= 1, (
        f"reload_count grew to {app.state.scheduler.metrics.reload_count} "
        f"— suggests the clamp failed and we entered a reload loop."
    )
