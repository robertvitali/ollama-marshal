"""Model lifecycle management: preload and unload via Ollama API."""

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
    """

    def __init__(self, ollama_host: str = "http://localhost:11434") -> None:
        self._ollama_host = ollama_host

    async def preload(self, model: str) -> bool:
        """Preload a model into Ollama's VRAM.

        Sends an empty-prompt request which triggers model loading, then
        waits for the model to appear in /api/ps.

        Args:
            model: The model name to preload.

        Returns:
            True if the model was successfully loaded.
        """
        logger.info("lifecycle.preloading", model=model)
        try:
            async with httpx.AsyncClient() as client:
                # Send empty-prompt request to trigger loading
                await client.post(
                    f"{self._ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "keep_alive": "24h",
                    },
                    timeout=_LOAD_TIMEOUT,
                )

                # Wait for model to appear in /api/ps
                loaded = await self._wait_for_model(client, model)
                if loaded:
                    logger.info("lifecycle.preloaded", model=model)
                else:
                    logger.warning("lifecycle.preload_timeout", model=model)
                return loaded

        except httpx.HTTPError:
            logger.error("lifecycle.preload_failed", model=model, exc_info=True)
            return False

    async def unload(self, model: str) -> bool:
        """Unload a model from Ollama's VRAM.

        Sends a request with keep_alive: "0" which causes Ollama to
        immediately evict the model.

        Args:
            model: The model name to unload.

        Returns:
            True if the unload request was sent successfully.
        """
        logger.info("lifecycle.unloading", model=model)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self._ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "keep_alive": "0",
                    },
                    timeout=_UNLOAD_TIMEOUT,
                )
                logger.info("lifecycle.unloaded", model=model)
                return True

        except httpx.HTTPError:
            logger.error("lifecycle.unload_failed", model=model, exc_info=True)
            return False

    async def unload_all(self, models: list[str]) -> None:
        """Unload multiple models from VRAM.

        Args:
            models: List of model names to unload.
        """
        for model in models:
            await self.unload(model)

    async def _wait_for_model(
        self,
        client: httpx.AsyncClient,
        model: str,
    ) -> bool:
        """Wait for a model to appear in /api/ps after preloading.

        Args:
            client: The httpx client to use.
            model: The model name to wait for.

        Returns:
            True if the model appeared before timeout.
        """
        elapsed = 0.0
        while elapsed < _PS_POLL_MAX_WAIT:
            try:
                resp = await client.get(
                    f"{self._ollama_host}/api/ps",
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

    async def ensure_loaded(self, model: str, loaded_models: set[str]) -> bool:
        """Ensure a model is loaded, preloading if necessary.

        Args:
            model: The model name.
            loaded_models: Set of currently loaded model names.

        Returns:
            True if the model is (now) loaded.
        """
        if model in loaded_models:
            return True
        return await self.preload(model)

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
