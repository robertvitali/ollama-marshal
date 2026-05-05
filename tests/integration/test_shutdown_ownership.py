"""Integration coverage for Bug 8: shutdown ownership filtering.

# What this catches

The lifespan-shutdown unload path used to read
``_memory.get_loaded_models().keys()`` — i.e. every model in /api/ps —
and call ``lifecycle.unload_all`` against the union. /api/ps is a
GLOBAL view of the Ollama daemon: a marshal sharing an Ollama with
another marshal (or with a human running ``ollama run``) would
attempt to tear down models it never loaded itself. On the user's
M3 Ultra rig that meant the integration suite's test marshal trying
to unload prod marshal's heavy models (``gpt-oss:120b``,
``qwen3:235b``); the slow unloads blew past asgi-lifespan's 10s
shutdown deadline and produced ``shutdown.timed_out`` errors that
broke ``test_fault_proxy_one_instance_failure_does_not_break_others``
(among others).

# What this asserts

End-to-end against a real marshal lifespan: the shutdown teardown
calls ``lifecycle.unload_all`` only for owned models. Foreign models
in /api/ps survive.

# Synthetic-state design (deliberate)

Earlier iterations of this test relied on a real preload via
``/api/chat`` to drive ``mark_owned``. That turned out to be flaky
on shared dev rigs: if ANY other process (prod marshal, another
test, the user's normal usage) had the test model resident in
Ollama between our cold-start unload and marshal's first /api/ps
poll, the chat request would dispatch via
``_forward_loaded_model_requests`` (already-loaded fast path) and
never hit ``_attempt_preload`` — so ``mark_owned`` would never fire
and the precondition assertion would falsely fail.

The unit suite already covers ``mark_owned`` firing correctly out
of ``_attempt_preload`` (``TestPreloadOwnershipClaim``). What this
integration test uniquely catches is the LIFESPAN SHUTDOWN wiring:
``server.py`` reading ``get_owned_loaded_models``, iterating
per-instance, and calling ``lifecycle.unload_all`` with the right
arguments end-to-end against a wired-up real app. Driving that
state directly via ``mark_owned`` and a ``_loaded_models``
injection is honest about what's under test and avoids the
shared-rig flakiness.
"""

from __future__ import annotations

import httpx
import pytest
from asgi_lifespan import LifespanManager

from ollama_marshal.memory import LoadedModel
from tests.integration.conftest import (
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


async def test_shutdown_unloads_only_owned_models(
    marshal_config, tmp_marshal_paths
) -> None:
    """End-to-end Bug 8: foreign /api/ps entries survive lifespan shutdown.

    Steps:
    1. Build a test marshal app with shutdown.unload_models=True
       (default in marshal_config).
    2. Inside the lifespan: stop the poll loop so our injections
       aren't blown away by the next /api/ps refresh.
    3. Stamp ownership for ``owned_name`` via ``memory.mark_owned``
       and inject a matching ``_loaded_models`` entry — simulating
       a successful preload by THIS marshal.
    4. Inject a foreign ``_loaded_models`` entry for ``foreign_name``
       — simulating "another marshal loaded gpt-oss:120b on the
       same Ollama". No mark_owned for it; this is the
       contamination case Bug 8 fixes.
    5. Wrap ``lifecycle.unload_all`` with a delegating spy so the
       real cleanup still runs but we can record arguments.
    6. Exit the lifespan; the shutdown path runs.
    7. Assert: the spy was invoked, ``owned_name`` was unloaded, and
       the foreign entry was NEVER passed to ``unload_all``.
    """
    app = make_test_app(marshal_config, tmp_marshal_paths)
    transport = httpx.ASGITransport(app=app)
    spy_calls: list[tuple[list[str], str | None]] = []

    owned_name = "ownership-integration-owned:1"
    foreign_name = "ownership-integration-foreign:1"

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver"),
    ):
        internals = app.state._marshal_internals
        memory = internals.memory
        lifecycle = internals.lifecycle
        primary_url = memory.instances[0].url

        # Stop the /api/ps poll loop BEFORE injecting state so the
        # next refresh doesn't overwrite our synthetic _loaded_models.
        await memory.stop_polling()

        # Stamp ownership directly. Production code path is
        # scheduler._attempt_preload → memory.mark_owned; the unit
        # suite covers that wire-up. This test's unique value is
        # the lifespan-shutdown side, so we drive state directly.
        memory.mark_owned(owned_name, primary_url)
        memory._loaded_models[primary_url][owned_name] = LoadedModel(
            name=owned_name,
            size_vram=1_000_000,
            instance_url=primary_url,
        )

        # Foreign model: present in /api/ps view, never claimed.
        # Production analog: another marshal preloaded gpt-oss:120b
        # on the same Ollama daemon.
        memory._loaded_models[primary_url][foreign_name] = LoadedModel(
            name=foreign_name,
            size_vram=120_000_000_000,
            instance_url=primary_url,
        )

        # Sanity preconditions before we exit the lifespan.
        owned = memory.get_owned_loaded_models()
        assert owned == {primary_url: {owned_name}}, (
            "ownership filter wrong before shutdown; "
            f"get_owned_loaded_models()={owned!r}"
        )
        assert foreign_name in memory.get_loaded_models()

        # Spy on unload_all so we can assert exactly what shutdown
        # passed in. We still delegate to the real implementation
        # so any side effects (audit logs, structlog events) stay
        # representative.
        original_unload_all = lifecycle.unload_all

        async def spy_unload_all(
            models: list[str], instance_url: str | None = None
        ) -> None:
            spy_calls.append((sorted(models), instance_url))
            await original_unload_all(models, instance_url=instance_url)

        lifecycle.unload_all = spy_unload_all

    # Lifespan has exited — assert the shutdown unload path filtered
    # out the foreign entry.
    assert spy_calls, "expected lifespan shutdown to call unload_all"
    flat = [name for models, _ in spy_calls for name in models]
    assert owned_name in flat, (
        f"expected owned model {owned_name!r} to be unloaded at shutdown; "
        f"spy_calls={spy_calls!r}"
    )
    assert foreign_name not in flat, (
        f"foreign model {foreign_name!r} was passed to unload_all — "
        f"shutdown ownership filter is broken; spy_calls={spy_calls!r}"
    )
    # All recorded unload_all calls targeted the owning instance.
    for _models, instance_url in spy_calls:
        assert instance_url == app.state._marshal_internals.memory.instances[0].url, (
            f"unload_all routed to wrong instance: {instance_url!r}"
        )
