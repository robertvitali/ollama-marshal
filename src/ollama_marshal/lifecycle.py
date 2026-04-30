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
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

# Timeout for loading a model (can be large for first-time quantization)
_LOAD_TIMEOUT = 300  # 5 minutes
_UNLOAD_TIMEOUT = 60
_PS_POLL_INTERVAL = 1  # seconds
_PS_POLL_MAX_WAIT = 120  # seconds


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
    ) -> bool:
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

        Returns:
            True if the model was successfully loaded.
        """
        host = self._resolve_host(instance_url)
        logger.info(
            "lifecycle.preloading",
            model=model,
            num_ctx=num_ctx,
            instance=host,
        )
        try:
            async with httpx.AsyncClient() as client:
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
                await client.post(
                    f"{host}/api/generate",
                    json=payload,
                    timeout=_LOAD_TIMEOUT,
                )

                # Wait for model to appear in /api/ps on this instance.
                loaded = await self._wait_for_model(client, model, host)
                if loaded:
                    logger.info(
                        "lifecycle.preloaded",
                        model=model,
                        num_ctx=num_ctx,
                        instance=host,
                    )
                else:
                    logger.warning(
                        "lifecycle.preload_timeout",
                        model=model,
                        instance=host,
                    )
                return loaded

        except httpx.HTTPError:
            logger.error(
                "lifecycle.preload_failed",
                model=model,
                instance=host,
                exc_info=True,
            )
            return False

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
                for m in data.get("models", []):
                    if m.get("name") == model or m.get("model") == model:
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
    ) -> bool:
        """Ensure a model is loaded, preloading if necessary.

        Args:
            model: The model name.
            loaded_models: Set of currently loaded model names.
            num_ctx: If preloading, allocate the KV slot at this size.
            instance_url: Which Ollama instance to load on.

        Returns:
            True if the model is (now) loaded.
        """
        if model in loaded_models:
            return True
        return await self.preload(model, num_ctx=num_ctx, instance_url=instance_url)

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
