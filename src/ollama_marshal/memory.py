"""Memory management: RAM detection, budget calculation, and /api/ps polling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import psutil
import structlog

from ollama_marshal.config import MarshalConfig, parse_size

logger = structlog.get_logger()


@dataclass
class LoadedModel:
    """A model currently loaded in Ollama's memory.

    Attributes:
        name: Model name (e.g., 'llama3:latest').
        size_vram: Actual VRAM usage in bytes.
        expires_at: When Ollama would auto-evict (overridden by proxy).
    """

    name: str
    size_vram: int
    expires_at: str = ""


@dataclass
class MemoryBudget:
    """Calculated memory budget for model loading.

    Attributes:
        total_ram: Total system RAM in bytes.
        os_overhead: RAM reserved for the OS in bytes.
        safety_margin: Safety buffer in bytes.
        available: Usable RAM for models in bytes.
    """

    total_ram: int
    os_overhead: int
    safety_margin: int
    available: int = field(init=False)

    def __post_init__(self) -> None:
        self.available = self.total_ram - self.os_overhead - self.safety_margin


class MemoryManager:
    """Tracks loaded models and manages VRAM budget.

    Polls Ollama's /api/ps endpoint to maintain an accurate view of
    which models are loaded and how much VRAM they use. Provides
    methods to check if new models can fit and to select eviction
    candidates.
    """

    def __init__(self, config: MarshalConfig) -> None:
        self._config = config
        self._ollama_host = config.ollama.host
        self._poll_interval = config.memory.poll_interval
        self._loaded_models: dict[str, LoadedModel] = {}
        self._budget = self._calculate_budget()
        self._poll_task: asyncio.Task[None] | None = None

    def _calculate_budget(self) -> MemoryBudget:
        """Calculate the memory budget from config and system info."""
        if self._config.memory.total_ram:
            total = parse_size(self._config.memory.total_ram)
        else:
            total = psutil.virtual_memory().total

        overhead = parse_size(self._config.memory.os_overhead)
        margin = parse_size(self._config.memory.safety_margin)

        budget = MemoryBudget(
            total_ram=total,
            os_overhead=overhead,
            safety_margin=margin,
        )
        logger.info(
            "memory.budget_calculated",
            total_gb=round(total / (1024**3), 1),
            overhead_gb=round(overhead / (1024**3), 1),
            margin_gb=round(margin / (1024**3), 1),
            available_gb=round(budget.available / (1024**3), 1),
        )
        return budget

    async def start_polling(self) -> None:
        """Start the background /api/ps polling task."""
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("memory.polling_started", interval_s=self._poll_interval)

    async def stop_polling(self) -> None:
        """Stop the background polling task."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
            logger.info("memory.polling_stopped")

    async def _poll_loop(self) -> None:
        """Continuously poll /api/ps at the configured interval."""
        while True:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("memory.poll_error", exc_info=True)
            await asyncio.sleep(self._poll_interval)

    async def refresh(self) -> None:
        """Fetch current loaded models from /api/ps and update state."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._ollama_host}/api/ps",
                    timeout=10,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                self._update_from_ps(data)
        except httpx.HTTPError:
            logger.warning("memory.refresh_failed", reason="cannot reach Ollama")

    def _update_from_ps(self, ps_data: dict[str, Any]) -> None:
        """Update loaded models from /api/ps response.

        Args:
            ps_data: Parsed JSON from /api/ps.
        """
        new_loaded: dict[str, LoadedModel] = {}
        for m in ps_data.get("models", []):
            name = m.get("name", "")
            new_loaded[name] = LoadedModel(
                name=name,
                size_vram=int(m.get("size_vram", 0)),
                expires_at=m.get("expires_at", ""),
            )

        # Log changes
        added = set(new_loaded) - set(self._loaded_models)
        removed = set(self._loaded_models) - set(new_loaded)
        if added:
            logger.info("memory.models_loaded", models=sorted(added))
        if removed:
            logger.info("memory.models_unloaded", models=sorted(removed))

        self._loaded_models = new_loaded

    def get_loaded_models(self) -> dict[str, LoadedModel]:
        """Get currently loaded models.

        Returns:
            Dict mapping model names to LoadedModel instances.
        """
        return dict(self._loaded_models)

    def is_loaded(self, model: str) -> bool:
        """Check if a model is currently loaded in VRAM.

        Args:
            model: The model name.

        Returns:
            True if the model is loaded.
        """
        return model in self._loaded_models

    def used_vram(self) -> int:
        """Get total VRAM currently used by loaded models.

        Returns:
            Total VRAM usage in bytes.
        """
        return sum(m.size_vram for m in self._loaded_models.values())

    def available_vram(self) -> int:
        """Get remaining VRAM available for loading models.

        Returns:
            Available VRAM in bytes.
        """
        return self._budget.available - self.used_vram()

    def can_fit_model(self, model_size: int) -> bool:
        """Check if a model of the given size can fit in available VRAM.

        Args:
            model_size: Size of the model in bytes.

        Returns:
            True if the model fits without eviction.
        """
        return model_size <= self.available_vram()

    def get_eviction_candidates(
        self,
        pending_counts: dict[str, int],
        program_priorities: dict[str, str],
    ) -> list[str]:
        """Get loaded models ranked by eviction suitability.

        Models are scored by: fewest pending requests first, then lowest
        priority, then largest VRAM (free more space). Models with pending
        requests are less suitable for eviction.

        Args:
            pending_counts: Dict of model_name -> pending request count.
            program_priorities: Dict of model_name -> priority level string.

        Returns:
            List of model names ordered from best to worst eviction candidate.
        """
        priority_order = {"critical": 1, "normal": 0}

        candidates: list[tuple[int, int, int, str]] = []
        for name, model in self._loaded_models.items():
            pending = pending_counts.get(name, 0)
            priority = priority_order.get(program_priorities.get(name, "normal"), 0)
            # Sort key: (pending ascending, priority ascending, -size descending)
            candidates.append((pending, priority, -model.size_vram, name))

        candidates.sort()
        return [name for _, _, _, name in candidates]

    @property
    def budget(self) -> MemoryBudget:
        """Get the calculated memory budget."""
        return self._budget
