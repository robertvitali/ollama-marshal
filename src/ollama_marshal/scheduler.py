"""FIFO + bin-packing model-affinity scheduler with fairness constraints."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from ollama_marshal.config import MarshalConfig, Priority
from ollama_marshal.lifecycle import ModelLifecycle
from ollama_marshal.memory import MemoryManager
from ollama_marshal.queue import ModelQueues, RequestEnvelope
from ollama_marshal.registry import ModelRegistry
from ollama_marshal.stream import forward_request

logger = structlog.get_logger()

_SCHEDULER_TICK = 0.1  # seconds between scheduler loop iterations

# How long a single X-Burst-Size hint stays "active" without renewal.
# Each request from the same program-model pair refreshes the timer, so
# steady streams stay protected. After this many seconds of silence, the
# hint expires and the boost falls back to actual queue depth.
_BURST_HINT_TTL_S = 30.0

# Hard cap on per-request burst boost so an adversarial client can't
# claim to be in a 1,000,000-call burst and starve every other program.
# Anchored at max_skips * 4 so the cap scales with whatever fairness
# threshold the user has configured. (Default max_skips=3 → cap=12.)
_BURST_HINT_CAP_MULTIPLIER = 4

# Hard cap on the total number of LIVE burst hints stored at any time.
# Prevents an adversarial client from flooding many distinct
# (program_id, model) pairs to grow the dict unboundedly within the
# 30s TTL window before prune_expired runs. Excess hints beyond this
# cap are silently dropped (record() returns 0).
_MAX_LIVE_HINTS = 256

# Per-model aggregate boost cap. Even if multiple programs each
# legitimately register a hint at the per-pair cap, the model-level
# boost can't exceed this multiplier times max_skips. Without this, a
# malicious client could flood distinct fake program_ids each at
# the per-pair cap to produce arbitrary aggregate boost on a target
# model and starve every other program's eviction.
_BURST_HINT_AGGREGATE_MULTIPLIER = 8


@dataclass
class BurstHint:
    """A live X-Burst-Size hint for a (program, model) pair.

    The boost added to that model's effective pending count when scoring
    eviction candidates. Refreshed each time a request for the same pair
    sets a new (or same) X-Burst-Size header. Expires automatically
    `_BURST_HINT_TTL_S` seconds after the last refresh.
    """

    boost: int
    expires_at: float  # monotonic deadline


class BurstHints:
    """Per-program-model burst-size hint store with TTL-based expiry.

    Used by the eviction scorer (memory.py:get_eviction_candidates) to
    treat a program's "expected pending demand" as larger than the actual
    queue depth — so a model can survive eviction across N sequential
    calls from a long-running program even when only one call is in
    flight at a time.
    """

    def __init__(
        self,
        ttl_s: float = _BURST_HINT_TTL_S,
        cap_multiplier: int = _BURST_HINT_CAP_MULTIPLIER,
        max_live: int = _MAX_LIVE_HINTS,
        aggregate_multiplier: int = _BURST_HINT_AGGREGATE_MULTIPLIER,
    ) -> None:
        self._hints: dict[tuple[str, str], BurstHint] = {}
        self._ttl_s = ttl_s
        self._cap_multiplier = cap_multiplier
        self._max_live = max_live
        self._aggregate_multiplier = aggregate_multiplier

    def record(
        self,
        program_id: str,
        model: str,
        n: int,
        max_skips: int,
        now: float | None = None,
    ) -> int:
        """Record (or refresh) a burst hint for a (program, model) pair.

        Three caps protect against adversarial clients:
        1. Per-pair cap: `n` is clamped to `max_skips * cap_multiplier`.
        2. Total-dict cap: if the live hint dict is already at
           `max_live` entries and this would be a NEW pair, drop the
           record. Refreshing an existing pair is always allowed.
        3. Aggregate-per-model cap is enforced by all_boosts_by_model
           (not here) so it can never exceed `max_skips *
           aggregate_multiplier` regardless of how many programs hint.

        Returns the effective per-pair boost stored after capping (0
        if the record was dropped by the dict cap).
        """
        if not program_id or not model or n <= 0:
            return 0
        key = (program_id, model)
        # Total-dict cap — drop NEW pairs when over capacity. Refresh
        # of an existing pair is always allowed.
        if key not in self._hints and len(self._hints) >= self._max_live:
            logger.warning(
                "scheduler.burst_hint_capacity_exceeded",
                program=program_id,
                model=model,
                live_hints=len(self._hints),
            )
            return 0
        cap = max(1, max_skips * self._cap_multiplier)
        boost = min(int(n), cap)
        ts = time.monotonic() if now is None else now
        self._hints[key] = BurstHint(boost=boost, expires_at=ts + self._ttl_s)
        return boost

    def _aggregate_cap(self, max_skips: int) -> int:
        """Per-model cap on summed boosts across programs."""
        return max(1, max_skips * self._aggregate_multiplier)

    def boost_for_model(
        self,
        model: str,
        max_skips: int = 0,
        now: float | None = None,
    ) -> int:
        """Sum of all live hint boosts targeting this model, capped.

        Multiple programs hinting the same model add together (the
        eviction scorer should treat the model as "expecting all of
        them combined"), but the sum is clamped at `max_skips *
        aggregate_multiplier` so a flood of distinct attacker-controlled
        program_ids each at the per-pair cap can't produce arbitrary
        aggregate boost. Pass max_skips=0 to disable the aggregate cap
        (only safe when caller has already validated all program_ids).
        """
        ts = time.monotonic() if now is None else now
        total = 0
        for (_prog, m), hint in self._hints.items():
            if m != model or hint.expires_at <= ts:
                continue
            total += hint.boost
        if max_skips > 0:
            total = min(total, self._aggregate_cap(max_skips))
        return total

    def all_boosts_by_model(
        self,
        max_skips: int = 0,
        now: float | None = None,
    ) -> dict[str, int]:
        """Return current per-model boost dict (live hints only).

        Each per-model sum is clamped at `max_skips *
        aggregate_multiplier` to prevent the program_id-flooding
        attack described in `boost_for_model`. Pass max_skips=0 to
        disable the cap.
        """
        ts = time.monotonic() if now is None else now
        result: dict[str, int] = {}
        for (_prog, model), hint in self._hints.items():
            if hint.expires_at <= ts:
                continue
            result[model] = result.get(model, 0) + hint.boost
        if max_skips > 0:
            cap = self._aggregate_cap(max_skips)
            result = {m: min(v, cap) for m, v in result.items()}
        return result

    def prune_expired(self, now: float | None = None) -> int:
        """Drop expired entries. Returns the count removed.

        Cheap to call every scheduler tick — typical hint dict is small
        (one entry per active program-model pair).
        """
        ts = time.monotonic() if now is None else now
        before = len(self._hints)
        self._hints = {k: v for k, v in self._hints.items() if v.expires_at > ts}
        return before - len(self._hints)


class InflightTracker:
    """Per-model concurrent-dispatch gate.

    Issues an `asyncio.Semaphore` per model name, sized by
    `scheduler.parallel_per_model`. Marshal's `_process_batch` acquires
    a slot before forwarding a non-embedding request, so for any given
    loaded model at most N requests are in flight at once.

    This complements (rather than replaces) Ollama's own NUM_PARALLEL:

    - Ollama allocates KV cache slots when a model loads, sized by its
      OLLAMA_NUM_PARALLEL env var. That fixes Ollama's hard ceiling.
    - Marshal's tracker gates dispatch beneath that ceiling. With
      OLLAMA_NUM_PARALLEL=8 and parallel_per_model=4, at most 4 requests
      are in flight; the other 4 of Ollama's slots stay idle.

    Concurrency naturally adapts to demand: if only 1 envelope is queued,
    only 1 is dispatched (semaphore stays at 0/N). If 5 same-model
    envelopes arrive concurrently with N=4, 4 dispatch in parallel and
    the 5th waits on the semaphore (in-process — Ollama doesn't see it
    yet).
    """

    def __init__(self, parallel_per_model: int) -> None:
        self._parallel = max(1, int(parallel_per_model))
        self._sems: dict[str, asyncio.Semaphore] = {}

    @property
    def parallel_per_model(self) -> int:
        """Configured per-model concurrency limit."""
        return self._parallel

    def semaphore_for(self, model: str) -> asyncio.Semaphore:
        """Return (creating if needed) the semaphore for `model`.

        Created lazily so memory isn't burned on cold models. Lifetime
        matches the scheduler — semaphores are not pruned on eviction
        because reusing the same semaphore on reload is harmless and
        cheaper than recreating.
        """
        sem = self._sems.get(model)
        if sem is None:
            sem = asyncio.Semaphore(self._parallel)
            self._sems[model] = sem
        return sem


# Bumped on schema-breaking changes to the persisted metrics file. On a
# mismatch we log a warning and start fresh — better than crashing.
_METRICS_SCHEMA_VERSION = 1


@dataclass
class SchedulerMetrics:
    """Tracks scheduler performance metrics.

    Counters are LIFETIME (not per-process) when persistence is enabled.
    On startup, marshal reads the on-disk snapshot and seeds these
    counters; on shutdown and every 60s during runtime, the snapshot is
    rewritten. The dashboard's `Δ since dashboard started` view computes
    a baseline at first poll, so its delta math is unaffected by the
    reseeding.

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

    def to_json_dict(self) -> dict[str, Any]:
        """Serializable snapshot for `~/.ollama-marshal/metrics.json`.

        `started_at` is intentionally NOT persisted — it's a monotonic
        reference point that's only meaningful within the current
        process lifetime.
        """
        return {
            "schema_version": _METRICS_SCHEMA_VERSION,
            "saved_at": datetime.now(UTC).isoformat(),
            "requests_served": self.requests_served,
            "model_swaps": self.model_swaps,
            "evictions": self.evictions,
            "total_wait_ms": float(self.total_wait_ms),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> SchedulerMetrics:
        """Reconstruct from on-disk snapshot, validating schema version.

        Raises ValueError on schema mismatch so callers can log and fall
        back to a fresh metrics object instead of silently using stale
        or differently-shaped state.
        """
        version = data.get("schema_version")
        if version != _METRICS_SCHEMA_VERSION:
            raise ValueError(
                f"metrics schema version {version!r} != expected "
                f"{_METRICS_SCHEMA_VERSION}"
            )
        return cls(
            requests_served=int(data.get("requests_served", 0)),
            model_swaps=int(data.get("model_swaps", 0)),
            evictions=int(data.get("evictions", 0)),
            total_wait_ms=float(data.get("total_wait_ms", 0.0)),
            # started_at is intentionally fresh — current process clock.
        )

    def save_to(self, path: Path) -> None:
        """Write the JSON snapshot to disk. Best-effort.

        Creates parent dir if missing. Logs (not raises) on I/O error so
        a transient disk hiccup never crashes the proxy.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.to_json_dict(), indent=2) + "\n"
            path.write_text(payload)
        except OSError as exc:
            logger.warning(
                "scheduler.metrics_save_failed",
                path=str(path),
                error=str(exc) or repr(exc),
                error_type=type(exc).__name__,
            )

    @classmethod
    def load_from(cls, path: Path) -> SchedulerMetrics:
        """Load the JSON snapshot, falling back to fresh on any failure.

        Failures handled (each logs a warning and returns a fresh
        SchedulerMetrics):
        - File doesn't exist (first run, expected)
        - File unreadable / corrupt JSON
        - Schema version mismatch
        - Wrong types in fields
        """
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "scheduler.metrics_load_corrupt",
                path=str(path),
                error=str(exc) or repr(exc),
            )
            return cls()
        if not isinstance(data, dict):
            logger.warning("scheduler.metrics_load_wrong_shape", path=str(path))
            return cls()
        try:
            return cls.from_json_dict(data)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "scheduler.metrics_schema_mismatch",
                path=str(path),
                error=str(exc),
            )
            return cls()


class _NoopAudit:
    """Sentinel audit logger — used until lifespan installs a real one.

    Defining a tiny no-op class avoids a forward import dependency from
    scheduler → audit (which would create a circular import in the test
    fixtures). The real AuditLogger is duck-type-compatible with this.
    """

    enabled: bool = False

    async def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None


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
        # Audit logger — defaults to a NULL implementation so callers
        # don't need `if audit:` branches everywhere. Replaced by the
        # server lifespan with a real AuditLogger when audit.enabled.
        self.audit: Any = _NoopAudit()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        # Maps model name -> monotonic timestamp of last successful dispatch.
        # Used by _idle_evict_unused_models to time-evict models that haven't
        # been used in `config.scheduler.idle_eviction_minutes` minutes.
        self._last_activity: dict[str, float] = {}
        # Maps model name -> {program_id: monotonic timestamp of last dispatch}.
        # Surfaces "who is currently using this loaded model" in the status
        # payload and dashboard. Populated in _forward_single, cleared on
        # idle/forced eviction so dropped models don't show stale callers.
        self._active_programs: dict[str, dict[str, float]] = {}
        # X-Burst-Size hints (sequential-submission programs declaring
        # "I have more requests coming for this model"). Used by the
        # eviction scorer to keep loaded models alive across silent gaps
        # between same-program calls. Tunables come from SchedulerConfig
        # so operators can override TTL/caps without recompiling.
        self.burst_hints = BurstHints(
            ttl_s=config.scheduler.burst_hint_ttl_s,
            cap_multiplier=config.scheduler.burst_hint_cap_multiplier,
            max_live=config.scheduler.burst_hint_max_live,
            aggregate_multiplier=config.scheduler.burst_hint_aggregate_multiplier,
        )
        # Per-model concurrent-dispatch gate. parallel_per_model defaults
        # to 1 (current sequential behavior). When raised, _process_batch
        # fans out same-model envelopes through this semaphore.
        self.inflight = InflightTracker(config.scheduler.parallel_per_model)

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

        1. Forward requests for already-loaded models (immediate)
        2. Handle critical-priority preemption (load critical model now)
        3. Handle unskippable requests (fairness enforcement)
        4. Bin-pack fitting models into remaining VRAM
        5. Idle eviction — unload models that have been quiet longer than
           `scheduler.idle_eviction_minutes` (0 disables)
        6. Drop expired X-Burst-Size hints (cleanup; doesn't affect
           dispatch decisions on this tick)
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

        # Step 6: Drop expired X-Burst-Size hints. Cheap dict comprehension.
        # Doing this on the tick (rather than lazy-on-read) means the hint
        # store can't grow indefinitely if a client sets hints on a
        # never-recurring program-model pair.
        self.burst_hints.prune_expired()

    def active_programs_by_model(self) -> dict[str, list[str]]:
        """Return programs that have recently dispatched against each loaded model.

        Used by the status payload to show "who is using this loaded model"
        even when the model is currently idle (no pending requests). Dropped
        when the model is unloaded.
        """
        return {
            model: sorted(progs.keys())
            for model, progs in self._active_programs.items()
            if progs
        }

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
                self._active_programs.pop(model_name, None)
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
        # Add live X-Burst-Size hint boosts so a sequential-submission
        # program with only 1 envelope visible but a declared 50-burst
        # is treated as a 50-deep queue for eviction-scoring purposes.
        # Pass max_skips so per-model aggregate cap is enforced (prevents
        # a flood of distinct fake program_ids from bypassing the
        # per-pair cap to produce arbitrary boost on a target model).
        burst_boosts = self.burst_hints.all_boosts_by_model(
            max_skips=self.config.scheduler.max_skips
        )
        if burst_boosts:
            pending_counts = dict(pending_counts)
            for model_name, boost in burst_boosts.items():
                pending_counts[model_name] = pending_counts.get(model_name, 0) + boost

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
            needed_for=needed_for,
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
            self._last_activity.pop(target, None)
            self._active_programs.pop(target, None)
            await self.memory.refresh()
        return success

    async def _process_batch(self, batch: list[RequestEnvelope]) -> None:
        """Process a batch of requests by forwarding them to Ollama.

        Embeddings always run concurrently (they're cheap and short).
        Non-embeddings are gated through `InflightTracker` so per-model
        concurrent dispatches never exceed `scheduler.parallel_per_model`.
        With the default value of 1, this preserves v0.2.x sequential
        behavior. With a higher value (and matching OLLAMA_NUM_PARALLEL),
        same-model envelopes fan out and Ollama serves them in parallel.

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

        # Non-embeddings: gated by per-model parallelism budget. All
        # envelopes in `others` share the same model (callers always
        # build per-model batches via dequeue_batch), so a single
        # semaphore handles the whole group.
        if others:
            sem = self.inflight.semaphore_for(others[0].model)

            async def _gated(env: RequestEnvelope) -> None:
                async with sem:
                    await self._forward_single(env)

            await asyncio.gather(*(_gated(e) for e in others), return_exceptions=True)

        # Embeddings always concurrent (no semaphore — they're fast).
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
            now = time.monotonic()
            self._last_activity[envelope.model] = now
            self._active_programs.setdefault(envelope.model, {})[
                envelope.program_id
            ] = now
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
            # Best-effort audit emission (no-op when audit.enabled=false).
            await self.audit.record(
                "request.served",
                program_id=envelope.program_id,
                model=envelope.model,
                endpoint=envelope.endpoint,
                wait_ms=wait_ms,
                stream=envelope.stream,
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
            await self.audit.record(
                "request.failed",
                program_id=envelope.program_id,
                model=envelope.model,
                endpoint=envelope.endpoint,
                error_type=type(exc).__name__,
            )
