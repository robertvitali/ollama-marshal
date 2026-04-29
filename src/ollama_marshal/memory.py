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
        # Tracks the num_ctx that marshal asked Ollama to allocate when
        # each currently-loaded model was last preloaded. Distinct from
        # "the largest num_ctx any served request needed" — we care
        # about what slot Ollama actually has, so reload-on-need can
        # decide whether the next request fits without reloading.
        self._allocated_num_ctx: dict[str, int] = {}
        # Names that marshal explicitly unloaded itself. Observing a
        # model leave /api/ps WITHOUT being in this set is an
        # unexpected (Ollama-side memory-pressure) eviction. The set
        # is consumed and cleared on the next poll cycle.
        self._intended_unloads: set[str] = set()
        # Counter incremented whenever the poll loop detects an
        # unexpected unload. The scheduler reads + zeros this to roll
        # the value into SchedulerMetrics.unexpected_unloads.
        self.unexpected_unloads_observed: int = 0

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
        # /api/ps comes from a process we don't control (Ollama, or a
        # proxy in front of it). Validate shape defensively: a single
        # malformed entry must not crash the polling loop, since a
        # crash-then-broad-except would leave _loaded_models stale and
        # silently turn unexpected_unloads detection into a false
        # negative on the very signal Surface C2 was meant to catch.
        models_raw = ps_data.get("models")
        if not isinstance(models_raw, list):
            models_raw = []
        for m in models_raw:
            if not isinstance(m, dict):
                continue
            name = m.get("name", "")
            if not isinstance(name, str) or not name:
                continue
            try:
                size_vram = int(m.get("size_vram", 0))
            except (TypeError, ValueError):
                size_vram = 0
            expires_at = m.get("expires_at", "")
            if not isinstance(expires_at, str):
                expires_at = ""
            new_loaded[name] = LoadedModel(
                name=name, size_vram=size_vram, expires_at=expires_at
            )

        # Log changes
        added = set(new_loaded) - set(self._loaded_models)
        removed = set(self._loaded_models) - set(new_loaded)
        if added:
            logger.info("memory.models_loaded", models=sorted(added))
        if removed:
            logger.info("memory.models_unloaded", models=sorted(removed))

        # Detect Ollama-side memory-pressure evictions: anything that
        # disappeared without marshal having marked it for unload is
        # an unexpected unload, signaling Ollama-side tuning is needed
        # (e.g. lower OLLAMA_NUM_PARALLEL, q8_0 KV cache).
        for name in removed:
            if name in self._intended_unloads:
                self._intended_unloads.discard(name)
            else:
                self.unexpected_unloads_observed += 1
                logger.warning(
                    "memory.unexpected_unload",
                    model=name,
                    reason=(
                        "Ollama dropped this model without marshal asking. "
                        "Run `ollama-marshal doctor` for tuning suggestions."
                    ),
                )
            # Forget the allocated num_ctx for any model that's no
            # longer loaded — its KV slots are gone.
            self._allocated_num_ctx.pop(name, None)

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

    # ------------------------------------------------------------------
    # Allocated num_ctx tracking (Surface C1 Dim 4)
    # ------------------------------------------------------------------

    def record_allocated_num_ctx(self, model: str, num_ctx: int) -> None:
        """Record what num_ctx marshal asked Ollama to allocate at preload.

        Called from the scheduler immediately after a successful
        `lifecycle.preload(model, num_ctx=N)`. Used by `needs_reload`
        to decide if a subsequent request needs more context than the
        current slot allocation.
        """
        self._allocated_num_ctx[model] = num_ctx

    def get_allocated_num_ctx(self, model: str) -> int | None:
        """Return the num_ctx marshal preloaded the model with (or None)."""
        return self._allocated_num_ctx.get(model)

    def needs_reload(self, model: str, requested_num_ctx: int) -> bool:
        """Check if `requested_num_ctx` exceeds what the model has allocated.

        A True result means the scheduler must unload + preload the
        model at a larger num_ctx before serving the next request.
        False means the request fits and can dispatch immediately.

        Returns False if the model isn't loaded yet (the upcoming
        preload will use the right size) or if no allocation has been
        recorded — except: a recorded allocation of 0 is a SENTINEL
        meaning "previous reload's preload failed, we don't actually
        know what slot Ollama has." In that case always return True so
        the scheduler tries again rather than silently dispatching
        against an unknown slot size.
        """
        if model not in self._loaded_models:
            return False
        current = self._allocated_num_ctx.get(model)
        if current is None:
            return False
        if current == 0:
            # Sentinel: prior reload failed mid-flight. Always reload.
            return True
        return requested_num_ctx > current

    def mark_intended_unload(self, model: str) -> None:
        """Tell the poll loop that marshal is intentionally unloading `model`.

        Without this, the next /api/ps poll would see the model gone
        and wrongly count it as an unexpected (Ollama-side) eviction.
        """
        self._intended_unloads.add(model)

    def take_unexpected_unload_count(self) -> int:
        """Return the count of unexpected unloads since last call, then reset."""
        n = self.unexpected_unloads_observed
        self.unexpected_unloads_observed = 0
        return n
