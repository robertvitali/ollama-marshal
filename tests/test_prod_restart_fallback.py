"""Unit test for the opt-in prod-marshal restart graceful fallback.

Imports the helper from ``tests/integration/conftest.py`` directly — the
package is importable even though ``tests/integration`` is collection-
ignored in the default ``make test`` run. Locks the graceful-degradation
contract: on a non-launchd host (``launchctl`` absent, e.g. Linux/CI) the
opt-in restart is a no-op and never spawns a subprocess, so it can never
hang or break the session-scoped autouse fixture.
"""

from __future__ import annotations

import shutil
import subprocess

from tests.integration.conftest import _maybe_restart_prod_marshal


async def test_restart_noop_when_launchctl_absent(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError(
            "subprocess.run must not be called when launchctl is absent"
        )

    monkeypatch.setattr(subprocess, "run", _must_not_run)
    # token=None also exercises the no-token drain-skip branch; must not raise.
    await _maybe_restart_prod_marshal(None)
