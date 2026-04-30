"""Memory management: RAM detection, budget calculation, and /api/ps polling.

# Multi-instance invariants (v0.5.0+)

Marshal can front multiple Ollama instances at once (each with a
different ``OLLAMA_KV_CACHE_TYPE``) for memory-pressure failover.
``MemoryManager`` tracks loaded models per-instance — but the **VRAM
budget is GLOBAL**, not partitioned per instance. Reason: on a single
Mac, all Ollama processes share the same unified-memory pool, so
double-counting the budget would refuse requests that actually fit.

Per-instance state exists only so routing can answer "is this model
loaded on instance X" and "what's allocated at what num_ctx on
instance X". Budget math (``available_vram``, ``can_fit_model``)
sums across all instances and compares against the single shared
budget.

Single-instance setups (the default) have one entry in
``config.instances`` (auto-backfilled from legacy ``ollama.host``);
the per-instance bookkeeping degenerates to a flat map and behavior is
identical to v0.4.x.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import psutil
import structlog

from ollama_marshal.config import MarshalConfig, OllamaInstance, parse_size
from ollama_marshal.routing import FitProbe

logger = structlog.get_logger()


@dataclass
class LoadedModel:
    """A model currently loaded in Ollama's memory.

    Attributes:
        name: Model name (e.g., 'llama3:latest').
        size_vram: Actual VRAM usage in bytes.
        expires_at: When Ollama would auto-evict (overridden by proxy).
        instance_url: URL of the Ollama instance holding this copy.
            Always set in v0.5.0+; present so callers that consume a
            flat ``LoadedModel`` snapshot can still attribute the
            entry back to its instance.
    """

    name: str
    size_vram: int
    expires_at: str = ""
    instance_url: str = ""


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
    """Tracks loaded models (per-instance) and the GLOBAL VRAM budget.

    Polls each configured Ollama instance's /api/ps endpoint to maintain
    an accurate view of which models are loaded where and how much VRAM
    they use. Provides per-instance APIs for routing
    (``probe_fit``, ``loaded_on``, ``is_loaded_on``) and a global view
    for bin-packing (``available_vram``, ``can_fit_model``).

    Per-instance state:
        ``_loaded_models[instance_url][model_name] = LoadedModel``
        ``_allocated_num_ctx[instance_url][model_name] = int``
        ``_intended_unloads[instance_url] = set[model_name]``

    Single-instance setups (one entry in ``config.instances``) work
    identically to v0.4.x — the per-instance maps just have one key.
    """

    def __init__(self, config: MarshalConfig) -> None:
        self._config = config
        # Snapshot the instance list at construction so the rest of the
        # class can rely on a stable order. Config validator guarantees
        # at least one entry (legacy ``ollama.host`` is auto-promoted).
        self._instances: list[OllamaInstance] = list(config.instances)
        self._poll_interval = config.memory.poll_interval
        # Per-instance loaded models. Outer key is instance URL, inner
        # key is model name.
        self._loaded_models: dict[str, dict[str, LoadedModel]] = {
            inst.url: {} for inst in self._instances
        }
        self._budget = self._calculate_budget()
        self._poll_task: asyncio.Task[None] | None = None
        # Per-instance num_ctx tracking. Same structure as
        # ``_loaded_models`` but stores the num_ctx marshal preloaded
        # each model with on each instance. Used by ``needs_reload``.
        self._allocated_num_ctx: dict[str, dict[str, int]] = {
            inst.url: {} for inst in self._instances
        }
        # Per-instance intended-unload tracking. When marshal is the
        # one removing a model from an instance, we record it here so
        # the next poll doesn't wrongly count it as an unexpected
        # Ollama-side eviction.
        self._intended_unloads: dict[str, set[str]] = {
            inst.url: set() for inst in self._instances
        }
        # Counter incremented whenever the poll loop detects an
        # unexpected unload on ANY instance. The scheduler reads + zeros
        # this to roll the value into SchedulerMetrics.unexpected_unloads.
        self.unexpected_unloads_observed: int = 0

    @property
    def instances(self) -> list[OllamaInstance]:
        """Configured instances, ordered highest precision first."""
        return list(self._instances)

    def _calculate_budget(self) -> MemoryBudget:
        """Calculate the memory budget from config and system info.

        Budget is GLOBAL across all instances. Mac unified memory is a
        single pool so partitioning per-instance would double-count.
        """
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
            instance_count=len(self._instances),
        )
        return budget

    async def start_polling(self) -> None:
        """Start the background /api/ps polling task."""
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "memory.polling_started",
            interval_s=self._poll_interval,
            instance_count=len(self._instances),
        )

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
        """Fetch /api/ps from every instance and update per-instance state."""
        # Fan out the polls so a slow instance doesn't gate the others.
        # Errors are logged per-instance and don't abort the others.
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *(self._refresh_one(client, inst) for inst in self._instances),
                return_exceptions=True,
            )
        for inst, result in zip(self._instances, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "memory.refresh_failed",
                    instance=inst.url,
                    reason="cannot reach Ollama",
                )

    async def _refresh_one(
        self,
        client: httpx.AsyncClient,
        instance: OllamaInstance,
    ) -> None:
        """Poll one instance's /api/ps and update its slot of state."""
        try:
            resp = await client.get(
                f"{instance.url}/api/ps",
                timeout=10,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPError:
            # Re-raise so the caller logs at the right granularity.
            raise
        self._update_from_ps(instance.url, data)

    def _update_from_ps(self, instance_url: str, ps_data: dict[str, Any]) -> None:
        """Update one instance's loaded models from its /api/ps response.

        Args:
            instance_url: URL of the instance this data came from.
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
                name=name,
                size_vram=size_vram,
                expires_at=expires_at,
                instance_url=instance_url,
            )

        prev_loaded = self._loaded_models.get(instance_url, {})
        added = set(new_loaded) - set(prev_loaded)
        removed = set(prev_loaded) - set(new_loaded)
        if added:
            logger.info(
                "memory.models_loaded",
                instance=instance_url,
                models=sorted(added),
            )
        if removed:
            logger.info(
                "memory.models_unloaded",
                instance=instance_url,
                models=sorted(removed),
            )

        intended = self._intended_unloads.get(instance_url, set())
        # Detect Ollama-side memory-pressure evictions: anything that
        # disappeared without marshal having marked it for unload is
        # an unexpected unload, signaling Ollama-side tuning is needed
        # (e.g. lower OLLAMA_NUM_PARALLEL, q8_0 KV cache).
        for name in removed:
            if name in intended:
                intended.discard(name)
            else:
                self.unexpected_unloads_observed += 1
                logger.warning(
                    "memory.unexpected_unload",
                    instance=instance_url,
                    model=name,
                    reason=(
                        "Ollama dropped this model without marshal asking. "
                        "Run `ollama-marshal doctor` for tuning suggestions."
                    ),
                )
            # Forget the allocated num_ctx for any model that's no
            # longer loaded — its KV slots are gone.
            self._allocated_num_ctx.get(instance_url, {}).pop(name, None)

        self._loaded_models[instance_url] = new_loaded

    # ------------------------------------------------------------------
    # Loaded-models accessors (flat-view kept for callers that don't care
    # about which instance, plus per-instance accessors for routing)
    # ------------------------------------------------------------------

    def get_loaded_models(self) -> dict[str, LoadedModel]:
        """Get all loaded models across all instances (flat view).

        Higher-precision instances win on conflict — if the same model
        is somehow loaded on both f16 and q8 (e.g. mid-promotion), the
        f16 entry shadows the q8 entry in this flat view. Routing
        callers that need to see both copies use ``loaded_on`` /
        ``is_loaded_on`` directly.
        """
        flat: dict[str, LoadedModel] = {}
        # Walk in instance order (highest precision first); first-write-
        # wins gives the higher-precision copy precedence.
        for inst in self._instances:
            for name, model in self._loaded_models.get(inst.url, {}).items():
                flat.setdefault(name, model)
        return flat

    def get_loaded_models_on(self, instance_url: str) -> dict[str, LoadedModel]:
        """Get loaded models on a specific instance."""
        return dict(self._loaded_models.get(instance_url, {}))

    def loaded_on(self) -> dict[str, set[str]]:
        """Map of instance_url -> set of model names loaded there.

        Used by ``routing.pick_instance`` via ``RoutingState``.
        """
        return {url: set(loaded.keys()) for url, loaded in self._loaded_models.items()}

    def is_loaded(self, model: str) -> bool:
        """Check if a model is loaded on ANY instance.

        Args:
            model: The model name.

        Returns:
            True if at least one instance holds this model.
        """
        return any(model in loaded for loaded in self._loaded_models.values())

    def is_loaded_on(self, model: str, instance_url: str) -> bool:
        """Check if a model is loaded on a specific instance."""
        return model in self._loaded_models.get(instance_url, {})

    def find_instance_for(self, model: str) -> str | None:
        """Return the instance URL holding `model`, or None if unloaded.

        Walks instances in declared order (highest precision first), so
        on conflict the higher-precision copy wins.
        """
        for inst in self._instances:
            if model in self._loaded_models.get(inst.url, {}):
                return inst.url
        return None

    def used_vram(self) -> int:
        """Get total VRAM currently used by loaded models (across all instances).

        Mac unified memory is a single pool, so summing across instances
        gives the real used budget.
        """
        return sum(
            m.size_vram
            for loaded in self._loaded_models.values()
            for m in loaded.values()
        )

    def available_vram(self) -> int:
        """Get remaining VRAM available for loading models (global)."""
        return self._budget.available - self.used_vram()

    def can_fit_model(self, model_size: int) -> bool:
        """Check if a model of the given size can fit in available VRAM.

        Compares against the global budget — same semantics as v0.4.x.
        Routing's per-instance probe uses ``probe_fit`` which adds
        per-instance attribution on top of this.

        Args:
            model_size: Size of the model in bytes.

        Returns:
            True if the model fits without eviction.
        """
        return model_size <= self.available_vram()

    # ------------------------------------------------------------------
    # Routing fit-probe (memory-pressure failover)
    # ------------------------------------------------------------------

    def probe_fit(
        self,
        instance_url: str,
        model_size: int,
        non_idle_loaded_on_instance: set[str],
    ) -> FitProbe:
        """Answer "would `model_size` fit on `instance_url`?".

        Three outcomes (see ``routing.FitProbe``):

        1. ``fits=True`` — global budget has room; no eviction needed.
        2. ``fits=False, would_evict_non_idle=True`` — fits only by
           evicting a model on this instance that has pending requests
           or recent activity. Triggers B-rule fallback.
        3. ``fits=False, would_evict_non_idle=False`` — this instance
           has no non-idle models to evict; only idle models on the
           instance could be freed. Equivalent to "could fit cleanly
           after idle-evictions on this instance".

        Args:
            model_size: Bytes the model would take on this instance.
            instance_url: Which instance to probe.
            non_idle_loaded_on_instance: Model names CURRENTLY loaded
                on `instance_url` that have pending requests or recent
                activity. Caller (the scheduler) computes this from
                the queue + active-program state.

        Returns:
            ``FitProbe`` describing the outcome.
        """
        if model_size <= self.available_vram():
            return FitProbe(fits=True, would_evict_non_idle=False)

        # No room without eviction. Walk this instance's loaded models —
        # if any are non-idle, evicting them would drop work, which
        # the B-rule wants to avoid.
        loaded_here = self._loaded_models.get(instance_url, {})
        for name in loaded_here:
            if name in non_idle_loaded_on_instance:
                return FitProbe(fits=False, would_evict_non_idle=True)
        # All loaded models on this instance are idle (or instance is
        # empty). Evicting them would free space. We don't compute
        # exact freed-bytes here — the caller's bin-packer will run
        # one eviction at a time anyway.
        return FitProbe(fits=False, would_evict_non_idle=False)

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
        # Walk all instances; flat view (first-write-wins, higher
        # precision shadows lower) matches the scheduler's existing
        # eviction model.
        seen: set[str] = set()
        for inst in self._instances:
            for name, model in self._loaded_models.get(inst.url, {}).items():
                if name in seen:
                    continue
                seen.add(name)
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
    # Allocated num_ctx tracking (Surface C1 Dim 4) — per-instance
    # ------------------------------------------------------------------

    def record_allocated_num_ctx(
        self,
        model: str,
        num_ctx: int,
        instance_url: str | None = None,
    ) -> None:
        """Record the num_ctx marshal preloaded a model with on an instance.

        Called from the scheduler immediately after a successful
        ``lifecycle.preload(model, num_ctx=N, instance_url=URL)``. Used
        by ``needs_reload`` to decide if a subsequent request needs more
        context than the current slot allocation on that instance.

        Args:
            model: Model name.
            num_ctx: Slot size requested.
            instance_url: Which instance the slot was allocated on.
                Defaults to the primary (first) instance for callers
                that haven't been instance-aware yet — matches v0.4.x
                behavior on single-instance setups.
        """
        url = instance_url if instance_url is not None else self._instances[0].url
        self._allocated_num_ctx.setdefault(url, {})[model] = num_ctx

    def get_allocated_num_ctx(
        self,
        model: str,
        instance_url: str | None = None,
    ) -> int | None:
        """Return the num_ctx marshal preloaded `model` with (or None).

        Args:
            model: Model name.
            instance_url: Which instance to query. None means "any
                instance" — returns the primary's allocation if set,
                otherwise the first instance that has one. The
                ``None`` path is used by tests + the legacy single-
                instance call sites; routing always passes an explicit
                URL.
        """
        if instance_url is not None:
            return self._allocated_num_ctx.get(instance_url, {}).get(model)
        # No URL specified — walk instances in declared order.
        for inst in self._instances:
            v = self._allocated_num_ctx.get(inst.url, {}).get(model)
            if v is not None:
                return v
        return None

    def needs_reload(
        self,
        model: str,
        requested_num_ctx: int,
        instance_url: str | None = None,
    ) -> bool:
        """Check if `requested_num_ctx` exceeds the model's current slot.

        A True result means the scheduler must unload + preload the
        model at a larger num_ctx before serving the next request.
        False means the request fits and can dispatch immediately.

        Returns False if the model isn't loaded on `instance_url`
        (the upcoming preload will use the right size) or if no
        allocation has been recorded — except: a recorded allocation
        of 0 is a SENTINEL meaning "previous reload's preload failed,
        we don't actually know what slot Ollama has." In that case
        always return True so the scheduler tries again rather than
        silently dispatching against an unknown slot size.

        Args:
            model: Model name.
            requested_num_ctx: Context size the next request needs.
            instance_url: Which instance to check. None resolves to the
                instance currently holding the model (or primary if
                none does — defensive single-instance fallback).
        """
        if instance_url is None:
            instance_url = self.find_instance_for(model)
            if instance_url is None:
                return False
        if not self.is_loaded_on(model, instance_url):
            return False
        current = self._allocated_num_ctx.get(instance_url, {}).get(model)
        if current is None:
            return False
        if current == 0:
            # Sentinel: prior reload failed mid-flight. Always reload.
            return True
        return requested_num_ctx > current

    def mark_intended_unload(
        self,
        model: str,
        instance_url: str | None = None,
    ) -> None:
        """Tell the poll loop that marshal is intentionally unloading `model`.

        Without this, the next /api/ps poll would see the model gone
        and wrongly count it as an unexpected (Ollama-side) eviction.

        Args:
            model: Model name being unloaded.
            instance_url: Which instance to mark. None marks every
                instance that currently holds the model — useful when
                unloading across all tiers (e.g. shutdown).
        """
        if instance_url is None:
            for inst in self._instances:
                if self.is_loaded_on(model, inst.url):
                    self._intended_unloads.setdefault(inst.url, set()).add(model)
            return
        self._intended_unloads.setdefault(instance_url, set()).add(model)

    def take_unexpected_unload_count(self) -> int:
        """Return the count of unexpected unloads since last call, then reset."""
        n = self.unexpected_unloads_observed
        self.unexpected_unloads_observed = 0
        return n
