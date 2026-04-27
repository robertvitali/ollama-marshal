"""FIFO + bin-packing model-affinity scheduler with fairness constraints."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from ollama_marshal.config import MarshalConfig, Priority
from ollama_marshal.lifecycle import ModelLifecycle
from ollama_marshal.memory import MemoryManager
from ollama_marshal.queue import ModelQueues, RequestEnvelope
from ollama_marshal.registry import ModelRegistry
from ollama_marshal.stream import forward_request

logger = structlog.get_logger()

_SCHEDULER_TICK = 0.1  # seconds between scheduler loop iterations


@dataclass
class SchedulerMetrics:
    """Tracks scheduler performance metrics.

    Attributes:
        requests_served: Total requests forwarded to Ollama.
        model_swaps: Number of model load/unload cycles.
        evictions: Number of forced model evictions.
        total_wait_ms: Cumulative wait time across all requests.
        started_at: Monotonic timestamp when the scheduler started.
    """

    requests_served: int = 0
    model_swaps: int = 0
    evictions: int = 0
    total_wait_ms: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def average_wait_ms(self) -> float:
        """Average wait time per request in milliseconds."""
        if self.requests_served == 0:
            return 0.0
        return self.total_wait_ms / self.requests_served


class Scheduler:
    """Model-affinity scheduler with FIFO + bin-packing + fairness.

    The scheduler runs as a single async task that continuously:
    1. Forwards requests for already-loaded models immediately
    2. Force-loads models for unskippable requests (fairness)
    3. Bin-packs smaller models into remaining VRAM (efficiency)
    4. Evicts lowest-value models when VRAM is needed

    Attributes:
        queues: The global request queue manager.
        memory: The memory/VRAM manager.
        registry: The model size registry.
        lifecycle: The model load/unload manager.
        config: The global configuration.
        metrics: Performance counters.
    """

    def __init__(
        self,
        queues: ModelQueues,
        memory: MemoryManager,
        registry: ModelRegistry,
        lifecycle: ModelLifecycle,
        config: MarshalConfig,
    ) -> None:
        self.queues = queues
        self.memory = memory
        self.registry = registry
        self.lifecycle = lifecycle
        self.config = config
        self.metrics = SchedulerMetrics()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        # Maps model name -> monotonic timestamp of last successful dispatch.
        # Used by _idle_evict_unused_models to time-evict models that haven't
        # been used in `config.scheduler.idle_eviction_minutes` minutes.
        self._last_activity: dict[str, float] = {}

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("scheduler.started")

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(
            "scheduler.stopped",
            requests_served=self.metrics.requests_served,
            model_swaps=self.metrics.model_swaps,
            evictions=self.metrics.evictions,
        )

    async def _run(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("scheduler.tick_error", exc_info=True)
            await asyncio.sleep(_SCHEDULER_TICK)

    async def _tick(self) -> None:
        """One iteration of the scheduler loop.

        1. Forward requests for loaded models (immediate)
        2. Handle unskippable requests (fairness enforcement)
        3. Bin-pack fitting models into remaining VRAM
        4. Increment skip counters for remaining requests
        """
        # Step 1: Forward requests for models already loaded
        await self._forward_loaded_model_requests()

        # Step 2: Handle critical priority preemption
        await self._handle_critical_preemption()

        # Step 3: Handle unskippable requests (skip limit reached)
        await self._handle_unskippable_requests()

        # Step 4: Bin-pack — load fitting models + increment skip counters
        # for models that were passed over this round
        await self._bin_pack_models()

        # Step 5: Idle eviction — unload models that haven't been touched
        # in a while, regardless of memory pressure. Configurable; 0 disables.
        await self._idle_evict_unused_models()

    async def _idle_evict_unused_models(self) -> None:
        """Evict loaded models that have been idle longer than the threshold.

        Skips models with pending requests (they're about to be served).
        Only fires if `scheduler.idle_eviction_minutes` > 0. The check runs
        every tick (cheap — just walks loaded models and compares timestamps).
        """
        threshold_min = self.config.scheduler.idle_eviction_minutes
        if threshold_min <= 0:
            return

        threshold_s = threshold_min * 60
        now = time.monotonic()
        loaded = self.memory.get_loaded_models()

        for model_name in list(loaded):
            # Stamp first-seen models as active now — gives them a full
            # idle window before they're eligible for time-eviction.
            if model_name not in self._last_activity:
                self._last_activity[model_name] = now
                continue

            idle_s = now - self._last_activity[model_name]
            if idle_s < threshold_s:
                continue

            # Don't evict a model that has pending requests.
            pending = await self.queues.pending_count(model_name)
            if pending > 0:
                continue

            logger.info(
                "scheduler.idle_evict",
                model=model_name,
                idle_s=round(idle_s, 1),
                threshold_s=threshold_s,
            )
            success = await self.lifecycle.unload(model_name)
            if success:
                self.metrics.evictions += 1
                self._last_activity.pop(model_name, None)
                await self.memory.refresh()
            # Only evict one per tick — keeps the loop lightweight.
            break

    async def _forward_loaded_model_requests(self) -> None:
        """Forward all pending requests whose model is already loaded."""
        loaded = self.memory.get_loaded_models()
        for model_name in loaded:
            pending = await self.queues.pending_count(model_name)
            if pending == 0:
                continue

            batch = await self.queues.dequeue_batch(model_name)
            if batch:
                await self._process_batch(batch)

    async def _handle_critical_preemption(self) -> None:
        """Check for critical-priority requests that need preemption."""
        all_pending = await self.queues.get_all_sorted_by_arrival()
        for envelope in all_pending:
            program_config = self.config.get_program_config(envelope.program_id)
            if program_config.priority != Priority.CRITICAL:
                continue
            if self.memory.is_loaded(envelope.model):
                continue
            # Critical request for unloaded model — preempt
            logger.info(
                "scheduler.critical_preemption",
                model=envelope.model,
                program=envelope.program_id,
            )
            await self._ensure_model_loaded(envelope.model)
            break  # Handle one preemption per tick

    async def _handle_unskippable_requests(self) -> None:
        """Force-load models for requests that have exceeded their skip limit."""
        max_skips = self.config.scheduler.max_skips
        unskippable = await self.queues.get_unskippable(max_skips)

        for envelope in unskippable:
            if self.memory.is_loaded(envelope.model):
                continue  # Already loaded — will be served in forward step

            logger.info(
                "scheduler.forced_load",
                model=envelope.model,
                skip_count=envelope.skip_count,
                max_skips=max_skips,
            )
            await self._ensure_model_loaded(envelope.model)
            break  # Load one forced model per tick to avoid thrashing

    async def _bin_pack_models(self) -> None:
        """Load models that fit in remaining VRAM, FIFO order."""
        all_pending = await self.queues.get_all_sorted_by_arrival()

        # Collect unique models not yet loaded, in arrival order
        seen: set[str] = set()
        models_to_try: list[str] = []
        for envelope in all_pending:
            if envelope.model not in seen and not self.memory.is_loaded(envelope.model):
                seen.add(envelope.model)
                models_to_try.append(envelope.model)

        skipped_models: list[str] = []
        for model in models_to_try:
            model_size = await self.registry.get_or_estimate_size(model)
            if self.memory.can_fit_model(model_size):
                logger.info(
                    "scheduler.bin_pack_load",
                    model=model,
                    size_gb=round(model_size / (1024**3), 2),
                    available_gb=round(self.memory.available_vram() / (1024**3), 2),
                )
                success = await self.lifecycle.preload(model)
                if success:
                    self.metrics.model_swaps += 1
                    await self.memory.refresh()
            else:
                skipped_models.append(model)

        # Only increment skip counters for models that were actually
        # passed over this round (didn't fit in VRAM), not every tick.
        for model in skipped_models:
            await self.queues.increment_skips_for_model(model)

    async def _ensure_model_loaded(self, model: str) -> bool:
        """Ensure a model is loaded, evicting others if needed.

        Args:
            model: The model to load.

        Returns:
            True if the model is now loaded.
        """
        if self.memory.is_loaded(model):
            return True

        model_size = await self.registry.get_or_estimate_size(model)

        # Evict if needed
        while not self.memory.can_fit_model(model_size):
            evicted = await self._evict_one(model)
            if not evicted:
                logger.error(
                    "scheduler.cannot_fit_model",
                    model=model,
                    size_gb=round(model_size / (1024**3), 2),
                    available_gb=round(self.memory.available_vram() / (1024**3), 2),
                )
                return False

        success = await self.lifecycle.preload(model)
        if success:
            self.metrics.model_swaps += 1
            await self.memory.refresh()
        return success

    async def _evict_one(self, needed_for: str) -> bool:
        """Evict the least valuable loaded model.

        Args:
            needed_for: The model we need to make room for (not evicted).

        Returns:
            True if a model was evicted.
        """
        pending_counts = await self.queues.pending_by_model()

        # Build priority map: use the highest priority of any pending
        # request for each loaded model (critical > normal)
        program_priorities: dict[str, str] = {}
        all_pending = await self.queues.get_all_sorted_by_arrival()
        for envelope in all_pending:
            prog_cfg = self.config.get_program_config(envelope.program_id)
            current = program_priorities.get(envelope.model, "normal")
            if prog_cfg.priority == Priority.CRITICAL:
                program_priorities[envelope.model] = "critical"
            elif envelope.model not in program_priorities:
                program_priorities[envelope.model] = current
        # Ensure all loaded models have an entry (default normal)
        for model_name in self.memory.get_loaded_models():
            if model_name not in program_priorities:
                program_priorities[model_name] = "normal"

        candidates = self.memory.get_eviction_candidates(
            pending_counts, program_priorities
        )

        # Don't evict the model we're trying to load
        candidates = [c for c in candidates if c != needed_for]

        if not candidates:
            return False

        target = candidates[0]
        logger.info(
            "scheduler.evicting",
            model=target,
            pending=pending_counts.get(target, 0),
            reason=f"making room for {needed_for}",
        )

        # Drain pending requests for the eviction target first (drain-before-evict)
        pending = await self.queues.pending_count(target)
        if pending > 0:
            batch = await self.queues.dequeue_batch(target)
            if batch:
                logger.info(
                    "scheduler.drain_before_evict",
                    model=target,
                    request_count=len(batch),
                )
                await self._process_batch(batch)

        success = await self.lifecycle.unload(target)
        if success:
            self.metrics.evictions += 1
            await self.memory.refresh()
        return success

    async def _process_batch(self, batch: list[RequestEnvelope]) -> None:
        """Process a batch of requests by forwarding them to Ollama.

        Streaming and non-streaming requests are handled differently.
        Embedding requests are sent concurrently for throughput.

        Args:
            batch: List of request envelopes to process.
        """
        # Separate embeddings (can be batched concurrently) from others
        embeddings: list[RequestEnvelope] = []
        others: list[RequestEnvelope] = []
        for envelope in batch:
            if envelope.endpoint in ("/api/embeddings", "/v1/embeddings"):
                embeddings.append(envelope)
            else:
                others.append(envelope)

        # Process non-embedding requests sequentially
        for envelope in others:
            await self._forward_single(envelope)

        # Process embeddings concurrently
        if embeddings:
            tasks = [self._forward_single(e) for e in embeddings]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _forward_single(self, envelope: RequestEnvelope) -> None:
        """Forward a single request to Ollama and complete the envelope.

        Args:
            envelope: The request to forward.
        """
        try:
            result = await forward_request(
                ollama_host=self.config.ollama.host,
                endpoint=envelope.endpoint,
                request_body=envelope.request_body,
                stream=envelope.stream,
            )
            envelope.complete(result)
            # Stamp this model as recently active so idle-eviction doesn't
            # touch it. Done on dispatch (not on completion) so a long
            # streaming response doesn't get unloaded mid-flight.
            self._last_activity[envelope.model] = time.monotonic()
            wait_ms = envelope.wait_time * 1000
            self.metrics.requests_served += 1
            self.metrics.total_wait_ms += wait_ms
            logger.debug(
                "scheduler.request_served",
                model=envelope.model,
                program=envelope.program_id,
                wait_ms=round(wait_ms, 1),
                endpoint=envelope.endpoint,
            )
        except Exception as exc:
            # Some httpx exception subclasses have empty str(); always include
            # the type name so the log entry is useful.
            logger.error(
                "scheduler.request_failed",
                model=envelope.model,
                endpoint=envelope.endpoint,
                error=str(exc) or repr(exc),
                error_type=type(exc).__name__,
            )
            envelope.fail(exc)
