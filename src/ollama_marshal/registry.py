"""Model size registry with background benchmarking and caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

_DEFAULT_REGISTRY_PATH = Path.home() / ".ollama-marshal" / "model_sizes.json"


class ModelRegistry:
    """Tracks the VRAM size of each Ollama model.

    On startup, loads a cached registry from disk and diffs it against
    the current set of downloaded models. New models are benchmarked
    in the background. Deleted models are removed from the cache.

    Attributes:
        registry_path: Path to the JSON cache file.
        ollama_host: Base URL of the Ollama API.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        registry_path: Path | None = None,
    ) -> None:
        self.ollama_host = ollama_host
        self.registry_path = registry_path or _DEFAULT_REGISTRY_PATH
        self._sizes: dict[str, int] = {}
        self._benchmarking: set[str] = set()

    async def initialize(self) -> None:
        """Load cached registry and sync with current Ollama models."""
        self._load_cache()
        await self._sync_with_ollama()

    def _load_cache(self) -> None:
        """Load model sizes from the JSON cache file."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                if isinstance(data, dict):
                    self._sizes = {k: int(v) for k, v in data.items()}
                    logger.info(
                        "model_registry.cache_loaded", model_count=len(self._sizes)
                    )
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "model_registry.cache_corrupt", path=str(self.registry_path)
                )
                self._sizes = {}

    def _save_cache(self) -> None:
        """Write current model sizes to the JSON cache file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self._sizes, indent=2) + "\n")
        logger.debug("model_registry.cache_saved", model_count=len(self._sizes))

    async def _sync_with_ollama(self) -> None:
        """Sync the registry against models currently downloaded in Ollama.

        Removes entries for models no longer present. New models are
        identified but not benchmarked here (call benchmark_unknown for that).
        """
        try:
            current_models = await self._fetch_model_list()
        except httpx.HTTPError:
            logger.warning("model_registry.sync_failed", reason="cannot reach Ollama")
            return

        # Remove models no longer downloaded
        stale = set(self._sizes.keys()) - set(current_models)
        for model in stale:
            del self._sizes[model]
            logger.info("model_registry.removed_stale", model=model)

        if stale:
            self._save_cache()

        unknown = set(current_models) - set(self._sizes.keys())
        if unknown:
            logger.info("model_registry.new_models_found", models=sorted(unknown))

    async def _fetch_model_list(self) -> list[str]:
        """Fetch the list of downloaded model names from Ollama.

        Returns:
            List of model name strings.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.ollama_host}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

    async def benchmark_model(self, model: str) -> int | None:
        """Benchmark a single model by loading it and measuring VRAM.

        Loads the model with an empty prompt, reads size_vram from /api/ps,
        then unloads it. Result is cached.

        Args:
            model: The model name to benchmark.

        Returns:
            VRAM size in bytes, or None if benchmarking failed.
        """
        if model in self._benchmarking:
            return None
        self._benchmarking.add(model)
        logger.info("model_registry.benchmarking", model=model)

        try:
            async with httpx.AsyncClient() as client:
                # Load the model
                await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": "5m"},
                    timeout=300,
                )

                # Read its VRAM size
                resp = await client.get(f"{self.ollama_host}/api/ps", timeout=10)
                resp.raise_for_status()
                ps_data = resp.json()

                size_vram = self._extract_model_vram(ps_data, model)
                if size_vram is not None:
                    self._sizes[model] = size_vram
                    self._save_cache()
                    logger.info(
                        "model_registry.benchmarked",
                        model=model,
                        size_vram=size_vram,
                        size_gb=round(size_vram / (1024**3), 2),
                    )

                # Unload the model
                await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": "0"},
                    timeout=60,
                )

                return size_vram

        except httpx.HTTPError:
            logger.warning("model_registry.benchmark_failed", model=model)
            return None
        finally:
            self._benchmarking.discard(model)

    async def benchmark_unknown(self) -> None:
        """Benchmark all models not yet in the registry.

        Loads each unknown model one at a time to avoid VRAM conflicts.
        """
        try:
            current_models = await self._fetch_model_list()
        except httpx.HTTPError:
            return

        unknown = [m for m in current_models if m not in self._sizes]
        if not unknown:
            logger.info("model_registry.all_benchmarked")
            return

        logger.info("model_registry.benchmark_starting", count=len(unknown))
        for model in unknown:
            await self.benchmark_model(model)

    def get_model_size(self, model: str) -> int | None:
        """Get the cached VRAM size for a model.

        Args:
            model: The model name.

        Returns:
            VRAM size in bytes, or None if not yet benchmarked.
        """
        return self._sizes.get(model)

    def is_benchmarked(self, model: str) -> bool:
        """Check if a model has been benchmarked.

        Args:
            model: The model name.

        Returns:
            True if the model's VRAM size is known.
        """
        return model in self._sizes

    def list_models(self) -> dict[str, int]:
        """Get all cached model sizes.

        Returns:
            Dict mapping model names to VRAM sizes in bytes.
        """
        return dict(self._sizes)

    def remove_model(self, model: str) -> None:
        """Remove a model from the registry.

        Args:
            model: The model name to remove.
        """
        if model in self._sizes:
            del self._sizes[model]
            self._save_cache()

    async def get_or_estimate_size(self, model: str) -> int:
        """Get model size from cache, or estimate from /api/show.

        Falls back to the model's file size from /api/show as a rough
        estimate when the model hasn't been benchmarked yet.

        Args:
            model: The model name.

        Returns:
            VRAM size in bytes (exact if benchmarked, estimated otherwise).
        """
        cached = self._sizes.get(model)
        if cached is not None:
            return cached

        # Estimate from /api/show
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_host}/api/show",
                    json={"name": model},
                    timeout=10,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                # Use model_info parameter count as rough estimate
                # Approximate: 4 bytes per parameter for Q4 quantization
                model_info = data.get("model_info", {})
                param_count = model_info.get("general.parameter_count", 0)
                if param_count:
                    estimated = int(param_count * 4)  # Q4 ~ 4 bytes/param
                    logger.debug(
                        "model_registry.estimated_size",
                        model=model,
                        estimated_bytes=estimated,
                    )
                    return estimated
        except httpx.HTTPError:
            pass

        # Last resort: return a conservative 4GB estimate
        logger.warning("model_registry.size_unknown", model=model, default="4GB")
        return 4 * 1024**3

    @staticmethod
    def _extract_model_vram(ps_data: dict[str, Any], model: str) -> int | None:
        """Extract size_vram for a model from /api/ps response data.

        Args:
            ps_data: The parsed JSON response from /api/ps.
            model: The model name to look for.

        Returns:
            VRAM size in bytes, or None if model not found.
        """
        for m in ps_data.get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return int(m.get("size_vram", 0))
        return None
