"""Surface C3 integration test — ``ollama-marshal doctor`` CLI.

Spawns the actual CLI subprocess and verifies the output includes the
expected recommendations + matches reality (model count from /api/tags).
"""

from __future__ import annotations

import re
import subprocess

import httpx
import pytest

from tests.integration.conftest import DEFAULT_OLLAMA_HOST, _ollama_reachable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not running on :11434",
    ),
]


def test_doctor_cli_produces_recommendations():
    """Subprocess invocation of ``ollama-marshal doctor`` returns sensible output.

    Verifies:
    - Exit code 0
    - Output mentions OLLAMA_KV_CACHE_TYPE=q8_0 (always-on recommendation)
    - Output mentions OLLAMA_FLASH_ATTENTION=1 (always-on recommendation)
    - Output mentions OLLAMA_NUM_PARALLEL=<n> (computed)
    - "Models installed: N" matches the actual /api/tags count
    """
    # Get the actual model count for cross-checking.
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{DEFAULT_OLLAMA_HOST}/api/tags")
        resp.raise_for_status()
        actual_count = len(resp.json().get("models", []))

    # Find the absolute path to ollama-marshal in this venv (avoids
    # shell PATH lookup, which ruff S607 flags as risky for portability).
    import shutil

    marshal_bin = shutil.which("ollama-marshal")
    assert marshal_bin is not None, (
        "ollama-marshal CLI not on PATH; install with `make install-dev`"
    )
    result = subprocess.run(  # noqa: S603 — fully-resolved path, hardcoded args
        [marshal_bin, "doctor"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"doctor exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    out = result.stdout
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in out, f"missing q8_0 recommendation in:\n{out}"
    assert "OLLAMA_FLASH_ATTENTION=1" in out, (
        f"missing FLASH_ATTENTION recommendation in:\n{out}"
    )
    # NUM_PARALLEL recommendation should be a number 1-4 (capped at 4).
    match = re.search(r"OLLAMA_NUM_PARALLEL=(\d+)", out)
    assert match is not None, f"missing NUM_PARALLEL recommendation in:\n{out}"
    parallel = int(match.group(1))
    assert 1 <= parallel <= 4, f"NUM_PARALLEL={parallel} outside [1, 4]"
    # Model count from doctor should match /api/tags.
    match = re.search(r"Models installed:\s+(\d+)", out)
    assert match is not None, f"missing 'Models installed:' line in:\n{out}"
    reported_count = int(match.group(1))
    assert reported_count == actual_count, (
        f"doctor reported {reported_count} models, /api/tags has {actual_count}"
    )
