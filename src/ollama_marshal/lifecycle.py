"""Model lifecycle management: preload and unload via Ollama API.

# Multi-instance (v0.5.0+)

Every operation takes an optional ``instance_url`` parameter. When
omitted, calls fall back to the constructor's ``ollama_host`` — that
keeps the existing single-instance call sites (and their unit tests)
working without modification. The scheduler always passes an explicit
``instance_url`` derived from the routing decision.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

# Default load timeout for legacy unit-test paths that don't pass one.
# Production call sites thread the configured Hop 2 timeout
# (``scheduler.ollama_forward_timeout_s``) through ``preload()`` —
# v0.6.4 raised the default from 300 (5 min) to 3600 (1 hour) so big
# first-time quantizations + slow disks don't trip a spurious failure.
_LOAD_TIMEOUT = 3600
_UNLOAD_TIMEOUT = 60
_PS_POLL_INTERVAL = 1  # seconds
_PS_POLL_MAX_WAIT = 120  # seconds


class LoadResult(enum.Enum):
    """Outcome of a ``ModelLifecycle.preload`` attempt (Bug 13).

    Classifies a load by a ``/api/ps`` snapshot taken just before our
    ``/api/generate``: ``NEW_LOAD`` when the model was absent in that
    snapshot (so this call most likely caused the load), ``ALREADY_LOADED``
    when it was already resident (loaded by another marshal or a human
    ``ollama run`` on a shared Ollama, or already ours). On a shared
    Ollama, claiming ownership (``memory.mark_owned``) of a model we did
    not load would make our shutdown teardown unload it; the scheduler
    therefore claims ownership only on ``NEW_LOAD``.

    Best-effort, not a proof of authorship: the pre-snapshot only
    *narrows* the race — a foreign loader can still slip in between the
    snapshot and our ``/api/generate``, yielding a ``NEW_LOAD`` for a
    model we did not actually load. The residual window is small (one
    ``/api/ps`` round-trip) and was the accepted trade-off of the Bug 13
    Option A fix; eliminating it entirely needs a cross-loader lease or
    an exclusive-daemon assumption (a separate, larger change).
    """

    NEW_LOAD = "new_load"
    ALREADY_LOADED = "already_loaded"
    FAILED = "failed"

    @property
    def loaded(self) -> bool:
        """True when the model is resident after the call (i.e. not FAILED).

        Both ``NEW_LOAD`` and ``ALREADY_LOADED`` mean the model is
        available to serve; only ``FAILED`` means it isn't. Callers that
        only care about success read ``result.loaded``; callers that
        gate ownership check ``result is LoadResult.NEW_LOAD``.
        """
        return self is not LoadResult.FAILED

    def __bool__(self) -> bool:
        """Truthiness mirrors ``loaded`` so ``if await preload(...):`` is safe.

        Without this, every enum member would be truthy by default and a
        caller that fell back to bare truthiness (``if result:``) would
        treat ``FAILED`` as success. Aligning ``__bool__`` with ``loaded``
        means ``FAILED`` is falsy and both success states are truthy — a
        guard against a future or overlooked bool-style caller after the
        ``preload`` return type changed from ``bool`` to ``LoadResult``.
        """
        return self.loaded


class ModelLifecycle:
    """Handles preloading and unloading models via the Ollama API.

    Preloads by sending an empty-prompt generate request, which causes
    Ollama to load the model into VRAM without producing output.
    Unloads by sending a request with keep_alive: "0".

    Multi-instance: every method takes an optional ``instance_url``
    that overrides the default ``ollama_host``. Pass it from the
    scheduler's routing decision.
    """

    def __init__(self, ollama_host: str = "http://localhost:11434") -> None:
        self._ollama_host = ollama_host

    def _resolve_host(self, instance_url: str | None) -> str:
        """Use the explicit instance URL if given, else the default."""
        return instance_url if instance_url is not None else self._ollama_host

    async def preload(
        self,
        model: str,
        num_ctx: int | None = None,
        instance_url: str | None = None,
        load_timeout_s: int | None = None,
        is_known_model_check: Callable[[str], Awaitable[bool]] | None = None,
    ) -> LoadResult:
        """Preload a model into Ollama's VRAM.

        Sends an empty-prompt request which triggers model loading, then
        waits for the model to appear in /api/ps.

        When `num_ctx` is provided, it's passed to Ollama's
        `/api/generate` so the KV cache slot is allocated at that size.
        This is the lever marshal uses to control per-model KV-cache
        cost (Surface C1 Dim 4) — without it, Ollama would allocate at
        its server-side default (typically the model's max), wasting
        VRAM on small-prompt programs.

        Args:
            model: The model name to preload.
            num_ctx: If set, allocate the KV slot at this context size.
            instance_url: Which Ollama instance to preload on. None
                falls back to the constructor default — single-instance
                behavior unchanged.
            load_timeout_s: Wall-clock budget for the preload HTTP
                call in seconds. None falls back to the
                ``_LOAD_TIMEOUT`` default. Production call sites pass
                ``scheduler.ollama_forward_timeout_s`` through.
            is_known_model_check: Optional async predicate that returns
                True if `model` is currently installed in Ollama.
                When provided and returning False, the preload skips
                the /api/generate call and returns False immediately
                — defense in depth against `ollama rm <model>` between
                request entry and preload time, which would otherwise
                drive lifecycle into a /api/generate loop that retries
                until the model gives up. Production call sites in
                the scheduler pass `registry.is_known_model`.

        Returns:
            A ``LoadResult``: ``NEW_LOAD`` if this call observably
            loaded the model (absent from /api/ps before our request,
            present after), ``ALREADY_LOADED`` if the model was already
            resident when we arrived (foreign-loaded on a shared Ollama,
            or already ours — the scheduler skips ``mark_owned`` so
            shutdown never tears it down), or ``FAILED`` if the load did
            not complete. Use ``result.loaded`` for plain success.
        """
        host = self._resolve_host(instance_url)
        timeout = _LOAD_TIMEOUT if load_timeout_s is None else load_timeout_s

        if is_known_model_check is not None:
            # Defense in depth must itself be defensive. A malformed
            # /api/tags response (e.g. a proxy mid-flight returning a
            # text/plain error page) makes the underlying httpx
            # `.json()` raise `json.JSONDecodeError` (a `ValueError`)
            # — outside the existing `httpx.HTTPError` catch in
            # registry.is_known_model. Without this wrap, that
            # exception would propagate into the scheduler tick and
            # poison the dispatcher. Fail-open on any predicate error:
            # better to attempt the preload (and let Ollama answer the
            # truth) than to hard-fail the request on a transient
            # registry hiccup.
            try:
                known = await is_known_model_check(model)
            except Exception:
                logger.warning(
                    "lifecycle.is_known_model_check_failed",
                    model=model,
                    instance=host,
                    exc_info=True,
                )
                known = True
            if not known:
                logger.warning(
                    "lifecycle.preload_skipped_unknown_model",
                    model=model,
                    instance=host,
                )
                return LoadResult.FAILED

        logger.info(
            "lifecycle.preloading",
            model=model,
            num_ctx=num_ctx,
            instance=host,
        )
        try:
            async with httpx.AsyncClient() as client:
                # Bug 13: snapshot residency BEFORE our load request. A
                # model already in /api/ps was loaded by another marshal
                # or a human ``ollama run`` in the window since our last
                # poll; our /api/generate would succeed against it and,
                # if the scheduler then claimed ownership, shutdown
                # teardown would unload a model we never loaded. ``None``
                # (a /api/ps error) is treated downstream as "possibly
                # resident" so the ambiguous case never wrongly claims
                # ownership.
                already_present = await self._is_loaded_now(client, model, host)
                payload: dict[str, Any] = {
                    "model": model,
                    "prompt": "",
                    "keep_alive": "24h",
                }
                if num_ctx is not None:
                    # Note: Ollama allocates KV slots at LOAD time using
                    # the num_ctx of the first request. After that, slot
                    # size is fixed until a reload — that's why
                    # marshal's reload-on-need logic exists.
                    payload["options"] = {"num_ctx": num_ctx}
                resp = await client.post(
                    f"{host}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                # If Ollama doesn't have the model (e.g. removed
                # between request entry and preload, or the registry
                # check is fooled by a multi-instance routing
                # mismatch), /api/generate returns 404. Without
                # raise_for_status, the response is silently consumed
                # and `_wait_for_model` polls /api/ps for up to 120s
                # waiting on a model that will never appear — stalling
                # the scheduler. Surface the HTTP error here so the
                # surrounding `except httpx.HTTPError` returns False
                # immediately and the caller's failure-counter logic
                # (scheduler._record_preload_failure) takes over.
                resp.raise_for_status()

                # Wait for model to appear in /api/ps on this instance.
                loaded = await self._wait_for_model(client, model, host)
                if not loaded:
                    logger.warning(
                        "lifecycle.preload_timeout",
                        model=model,
                        instance=host,
                    )
                    return LoadResult.FAILED
                # Bug 13: only NEW_LOAD (definitively absent before our
                # request) claims ownership. A True or unknown (None)
                # pre-snapshot both resolve to ALREADY_LOADED so an
                # ambiguous /api/ps read never produces a wrongful claim.
                if already_present is not False:
                    logger.info(
                        "lifecycle.preload_already_loaded",
                        model=model,
                        instance=host,
                        ps_snapshot=(
                            "unknown" if already_present is None else "present"
                        ),
                    )
                    return LoadResult.ALREADY_LOADED
                logger.info(
                    "lifecycle.preloaded",
                    model=model,
                    num_ctx=num_ctx,
                    instance=host,
                )
                return LoadResult.NEW_LOAD

        except httpx.HTTPError:
            logger.error(
                "lifecycle.preload_failed",
                model=model,
                instance=host,
                exc_info=True,
            )
            return LoadResult.FAILED

    async def unload(
        self,
        model: str,
        instance_url: str | None = None,
    ) -> bool:
        """Unload a model from Ollama's VRAM.

        Sends a request with keep_alive: "0" which causes Ollama to
        immediately evict the model.

        Args:
            model: The model name to unload.
            instance_url: Which Ollama instance to unload from. None
                falls back to the constructor default.

        Returns:
            True if the unload request was sent successfully.
        """
        host = self._resolve_host(instance_url)
        logger.info("lifecycle.unloading", model=model, instance=host)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{host}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "keep_alive": "0",
                    },
                    timeout=_UNLOAD_TIMEOUT,
                )
                logger.info("lifecycle.unloaded", model=model, instance=host)
                return True

        except httpx.HTTPError:
            logger.error(
                "lifecycle.unload_failed",
                model=model,
                instance=host,
                exc_info=True,
            )
            return False

    async def unload_all(
        self,
        models: list[str],
        instance_url: str | None = None,
    ) -> None:
        """Unload multiple models from VRAM.

        Args:
            models: List of model names to unload.
            instance_url: Which instance to unload from. None falls
                back to the constructor default.
        """
        for model in models:
            await self.unload(model, instance_url=instance_url)

    @staticmethod
    def _ps_contains(data: Any, model: str) -> bool:
        """True if ``model`` appears in an ``/api/ps`` payload.

        Ollama keys loaded models under either ``name`` or ``model``
        depending on version; match on both. Defends against a 200 whose
        body isn't the expected object (``data`` not a dict, or an entry
        not a dict) so a malformed/hostile response can't raise
        ``AttributeError`` into the caller — returns False instead.
        """
        if not isinstance(data, dict):
            return False
        for m in data.get("models", []):
            if isinstance(m, dict) and (
                m.get("name") == model or m.get("model") == model
            ):
                return True
        return False

    async def _is_loaded_now(
        self,
        client: httpx.AsyncClient,
        model: str,
        host: str,
    ) -> bool | None:
        """Single-shot ``/api/ps`` residency check (Bug 13 pre-snapshot).

        Args:
            client: The httpx client to use.
            model: The model name to check for.
            host: Base URL of the instance to poll.

        Returns:
            ``True``/``False`` when ``/api/ps`` answers cleanly, or
            ``None`` when residency cannot be determined (HTTP error or
            a malformed/non-JSON payload). Callers treat ``None``
            conservatively as "possibly resident" so an ambiguous
            snapshot never leads to a wrongful ownership claim.
        """
        try:
            resp = await client.get(f"{host}/api/ps", timeout=10)
            resp.raise_for_status()
            data: Any = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        # A 200 with a non-dict body (``[]``, a string) can't be
        # interpreted — treat as "can't determine" (None), which the
        # caller resolves conservatively to ALREADY_LOADED so a malformed
        # snapshot never drives a wrongful ownership claim.
        if not isinstance(data, dict):
            return None
        return self._ps_contains(data, model)

    async def _wait_for_model(
        self,
        client: httpx.AsyncClient,
        model: str,
        host: str,
    ) -> bool:
        """Wait for a model to appear in /api/ps on `host` after preloading.

        Args:
            client: The httpx client to use.
            model: The model name to wait for.
            host: Base URL of the instance to poll.

        Returns:
            True if the model appeared before timeout.
        """
        elapsed = 0.0
        while elapsed < _PS_POLL_MAX_WAIT:
            try:
                resp = await client.get(
                    f"{host}/api/ps",
                    timeout=10,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                if self._ps_contains(data, model):
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(_PS_POLL_INTERVAL)
            elapsed += _PS_POLL_INTERVAL
        return False

    async def ensure_loaded(
        self,
        model: str,
        loaded_models: set[str],
        num_ctx: int | None = None,
        instance_url: str | None = None,
        load_timeout_s: int | None = None,
    ) -> bool:
        """Ensure a model is loaded, preloading if necessary.

        WARNING (Bug 8): this method bypasses ownership tracking. It
        calls ``self.preload`` directly, NOT through the scheduler's
        ``_attempt_preload`` wrapper that calls ``memory.mark_owned``
        on success. Production scheduler code should always use
        ``_attempt_preload``; ``ensure_loaded`` is kept for legacy
        single-instance callers and tests. New code that goes through
        ``ensure_loaded`` will produce loaded-but-unowned models that
        survive shutdown teardown — usually unintended.

        Args:
            model: The model name.
            loaded_models: Set of currently loaded model names.
            num_ctx: If preloading, allocate the KV slot at this size.
            instance_url: Which Ollama instance to load on.
            load_timeout_s: Forwarded to ``preload`` when the model
                isn't already loaded.

        Returns:
            True if the model is (now) loaded.
        """
        if model in loaded_models:
            return True
        result = await self.preload(
            model,
            num_ctx=num_ctx,
            instance_url=instance_url,
            load_timeout_s=load_timeout_s,
        )
        return result.loaded

    @staticmethod
    def override_keep_alive(request_body: dict[str, Any]) -> dict[str, Any]:
        """Override keep_alive in a request to prevent Ollama auto-eviction.

        The proxy manages model lifecycle, so every request gets a long
        keep_alive to prevent Ollama's LRU from interfering.

        Args:
            request_body: The original request body.

        Returns:
            Modified request body with keep_alive set to 24h.
        """
        body = dict(request_body)
        body["keep_alive"] = "24h"
        return body
