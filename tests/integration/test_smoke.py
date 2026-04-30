"""Smoke tests — baseline verification of marshal end-to-end.

If any of these fail, every other integration test will fail. They're
the canary: do we boot, do we round-trip, does streaming work?

Each test takes ~3-5 seconds against the user's real Ollama.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.integration.conftest import (
    DEFAULT_OLLAMA_HOST,
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


def _required_model_pulled() -> bool:
    """Sync check that REQUIRED_MODEL is in /api/tags."""
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
    reason=(
        f"Required model {REQUIRED_MODEL!r} not pulled "
        f"(run `ollama pull {REQUIRED_MODEL}`)"
    ),
)


async def test_app_starts_and_status_endpoint_works(marshal_app):
    """Marshal lifespan boots cleanly and /api/marshal/status returns valid JSON.

    Uses no real Ollama traffic — just verifies the app comes up and
    responds. If THIS fails, every subsequent test will also fail and
    the whole suite is invalid.
    """
    client, _app = marshal_app
    resp = await client.get("/api/marshal/status")
    assert resp.status_code == 200
    body = resp.json()
    # Verify the canonical top-level shape — anything missing here
    # indicates the lifespan didn't fully initialize.
    for key in (
        "uptime_seconds",
        "loaded_models",
        "instances",
        "memory",
        "queue",
        "metrics",
    ):
        assert key in body, f"missing top-level key: {key}"
    assert isinstance(body["loaded_models"], list)
    assert isinstance(body["metrics"], dict)
    # v0.4.0 metrics MUST be present (regression for the bug local
    # /review caught on PR #6).
    for key in (
        "requests_served",
        "model_swaps",
        "evictions",
        "retries_attempted",
        "retries_succeeded",
        "unexpected_unloads",
        "reload_count",
    ):
        assert key in body["metrics"], f"missing metric: {key}"
    # v0.5.0+: per-instance breakdown. Even on legacy single-instance
    # setups this is a list with exactly one entry (the validator
    # backfills `instances=[primary]`).
    assert isinstance(body["instances"], list)
    assert len(body["instances"]) >= 1
    for inst in body["instances"]:
        for key in (
            "url",
            "kv_cache_type",
            "tier_label",
            "reachable",
            "loaded_models",
            "used_vram",
        ):
            assert key in inst, f"missing instance field: {key}"
        assert isinstance(inst["loaded_models"], list)
        assert isinstance(inst["reachable"], bool)


@_REQUIRES_MODEL
async def test_real_chat_round_trip(marshal_app):
    """Real /api/chat with a tiny prompt completes and loads the model.

    Verifies the full path: marshal accepts the request, scheduler
    dispatches it, lifecycle preloads the model, stream module forwards
    to Ollama, response comes back. Uses CRITICAL priority program ID
    (the default for tests).
    """
    client, _app = marshal_app
    body = {
        "model": REQUIRED_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    resp = await client.post(
        "/api/chat",
        json=body,
        headers={"X-Program-ID": PROGRAM_CRITICAL},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("done") is True, f"chat did not complete: {data}"
    msg = data.get("message", {})
    assert msg.get("role") == "assistant"
    assert isinstance(msg.get("content"), str)
    # Model should appear in /api/marshal/status as loaded.
    status = (await client.get("/api/marshal/status")).json()
    loaded_names = [m["name"] for m in status["loaded_models"]]
    assert REQUIRED_MODEL in loaded_names, (
        f"expected {REQUIRED_MODEL} loaded; got {loaded_names}"
    )
    # v0.5.0+: each loaded model entry is tagged with its instance and
    # tier so an operator can correlate model → instance without
    # cross-referencing the instances array.
    loaded_entry = next(
        m for m in status["loaded_models"] if m["name"] == REQUIRED_MODEL
    )
    assert loaded_entry.get("instance_url"), "instance_url not populated"
    assert loaded_entry.get("tier_label"), "tier_label not populated"


@_REQUIRES_MODEL
async def test_real_generate_streaming_round_trip(marshal_app):
    """Real /api/generate with stream=true yields ordered NDJSON chunks.

    Verifies the streaming path through the proxy: marshal forwards
    Ollama's chunked NDJSON without buffering, chunks arrive in order,
    final chunk has done=true. The smallest qwen3.5 model is a
    reasoning model — its tokens may land in the ``thinking`` field
    rather than ``response``. Either is fine for verifying the
    streaming pipe; this test is about the transport, not the content.
    """
    client, _app = marshal_app
    body = {
        "model": REQUIRED_MODEL,
        "prompt": "Write the word hello.",
        "stream": True,
        # Generous enough to exit the thinking phase even on a
        # reasoning model so we get at least one populated chunk
        # plus the done=true terminator.
        "options": {"num_predict": 64},
    }
    chunks: list[dict] = []
    async with client.stream(
        "POST",
        "/api/generate",
        json=body,
        headers={"X-Program-ID": PROGRAM_CRITICAL},
        timeout=60,
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        async for line in resp.aiter_lines():
            if not line:
                continue
            chunks.append(json.loads(line))
    assert len(chunks) >= 2, f"expected ≥2 chunks, got {len(chunks)}"
    assert chunks[-1].get("done") is True, f"last chunk not done: {chunks[-1]}"
    # Verify SOME content was streamed — either via `response` (final
    # answer) or `thinking` (reasoning model's intermediate tokens).
    # We don't care which; we care that the pipe carried tokens.
    total_text = "".join(c.get("response", "") + c.get("thinking", "") for c in chunks)
    assert len(total_text) > 0, f"no output text in any chunk: {chunks}"
