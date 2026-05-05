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
from ollama_marshal.queue import ModelQueues, PreloadFailedError, RequestEnvelope
from ollama_marshal.registry import ModelRegistry
from ollama_marshal.retry import backoff_delay, call_with_retry
from ollama_marshal.routing import (
    FitProbe,
    RoutingDecision,
    RoutingReason,
    RoutingState,
    pick_instance,
)
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
class _PreloadFailureState:
    """Per-model preload-failure tracking for backoff + giveup.

    Attributes:
        consecutive_failures: Count of consecutive failed preload
            attempts for this model. Cleared on a successful preload
            and after the giveup path runs.
        cooldown_until: Monotonic deadline before the next preload
            attempt for this model is allowed. The scheduler tick
            short-circuits any preload work for this model until the
            deadline passes.
    """

    consecutive_failures: int
    cooldown_until: float


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
        retries_attempted: Times marshal retried a forward_request after
            a transient failure. Lifetime counter.
        retries_succeeded: Times a retried request ultimately succeeded
            (subset of `retries_attempted`).
        unexpected_unloads: Times marshal observed Ollama drop a loaded
            model on its own (memory-pressure eviction). Persistent
            non-zero values signal Ollama-side memory tuning is needed.
        reload_count: Times marshal reloaded a model at a larger num_ctx
            because an incoming request needed more context than the
            current slot allocation (Surface C1 Dim 4).
        started_at: Monotonic timestamp when the scheduler started.
    """

    requests_served: int = 0
    model_swaps: int = 0
    evictions: int = 0
    total_wait_ms: float = 0.0
    retries_attempted: int = 0
    retries_succeeded: int = 0
    unexpected_unloads: int = 0
    reload_count: int = 0
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
            "retries_attempted": self.retries_attempted,
            "retries_succeeded": self.retries_succeeded,
            "unexpected_unloads": self.unexpected_unloads,
            "reload_count": self.reload_count,
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
            # New v0.4.0 fields default to 0 if loading a v0.3.x snapshot.
            retries_attempted=int(data.get("retries_attempted", 0)),
            retries_succeeded=int(data.get("retries_succeeded", 0)),
            unexpected_unloads=int(data.get("unexpected_unloads", 0)),
            reload_count=int(data.get("reload_count", 0)),
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
        # Admin pause state (v0.6.0+). When ``_dispatch_paused`` is set,
        # the scheduler tick suspends queue draining (envelopes pile up
        # but aren't dispatched). Bypass-flagged envelopes
        # (``RequestEnvelope.bypass_pause=True``) still dispatch — the
        # tick logic in v0.6.0 Stage 2 handles that filtering.
        # ``_in_flight_count`` tracks how many envelopes are currently
        # mid-dispatch to Ollama so the admin pause endpoint can wait
        # for the scheduler to actually be idle (drain) before
        # returning 200. Counter rather than set because RequestEnvelope
        # isn't hashable (mutable asyncio.Event field) and the drain
        # wait only needs the count, not envelope identity. Incremented
        # and decremented at the ``_forward_single`` boundary; see usage
        # in v0.6.0 Stage 2.
        self._dispatch_paused = False
        self._in_flight_count = 0
        # Auto-resume failsafe task. If a test session crashes (Ctrl-C,
        # OOM) before calling resume, the scheduler would otherwise
        # stay paused indefinitely and prod would be stuck at 503-free
        # but dispatch-frozen. The auto-resume timer flips the flag
        # back after ``auto_resume_after_seconds`` regardless. Cancelled
        # by an explicit resume.
        self._auto_resume_task: asyncio.Task[None] | None = None
        # Per-model preload-failure state (v0.6.4+). Without this, when
        # Ollama crashes/restarts, the 0.1s scheduler tick keeps calling
        # ``lifecycle.preload`` ~10 times/sec on every queued model
        # while Ollama is unreachable — observed 313 ``lifecycle.preload_failed``
        # entries in 30s during one Ollama crash recovery. Now: each
        # failed preload bumps a per-model counter and parks future
        # attempts behind an exponential-backoff cooldown
        # (``preload_backoff_base_s`` → ``preload_backoff_max_s`` with
        # full jitter). After ``preload_max_consecutive_failures``
        # attempts, queued envelopes for the model fail with
        # ``PreloadFailedError`` so clients get clear feedback rather
        # than waiting forever.
        self._preload_failures: dict[str, _PreloadFailureState] = {}
        # Per-(model, instance_url) reload requests fed by the memory
        # poller's ``_recent_unexpected_unloads`` set. Drained at the
        # top of every ``_tick`` and consumed by bin-packing — entries
        # bypass the preload cooldown (the model was a known-good
        # load that Ollama evicted, not a marshal preload failure)
        # so requests for an evicted model don't sit waiting behind
        # an unrelated cooldown window.
        self._needs_reload: set[tuple[str, str]] = set()

    def is_paused(self) -> bool:
        """Return True if dispatch is currently paused.

        Reads the in-memory flag — caller does not need to acquire a
        lock. Race: a concurrent ``pause()`` / ``resume()`` may flip
        the flag immediately after the read, but Python's GIL makes
        the bool read itself atomic. Callers using this for routing
        (e.g. middleware deciding whether to apply the pause guard)
        must accept that the value reflects the moment of the read.
        """
        return self._dispatch_paused

    async def pause(
        self,
        drain_timeout_s: float = 60.0,
        auto_resume_after_seconds: float = 300.0,
    ) -> bool:
        """Stop dispatching from the queue and wait for in-flight to drain.

        SOFT PAUSE: incoming requests still enqueue normally (no 503s).
        The scheduler stops popping envelopes off the queue; bypass-
        flagged envelopes (``RequestEnvelope.bypass_pause=True``)
        continue to dispatch via the bypass path. Returns once
        in-flight count reaches zero (scheduler is idle), so callers
        that pause to do test work can be confident no production
        inference is mid-stream.

        Idempotent — calling ``pause`` while already paused returns
        immediately with the current drain status. Each call resets
        the auto-resume timer (so a long test session can extend the
        pause by re-calling pause periodically).

        Args:
            drain_timeout_s: Max seconds to wait for in-flight
                dispatches to complete naturally. Long-running
                inferences (a 5min generation) may exceed this; the
                caller decides what to do (skip the test, retry
                later, etc).
            auto_resume_after_seconds: Failsafe — if no explicit
                ``resume`` call arrives within this many seconds, the
                scheduler resumes itself and emits an audit event
                (``admin.auto_resumed``). Defends against test-session
                crashes leaving prod paused forever. Set to a value
                comfortably larger than the expected pause duration.

        Returns:
            True if drain completed within ``drain_timeout_s``, False
            if timeout was hit (one or more envelopes still in flight).
            The pause flag is set in either case — a False return is
            informational, not a failure to pause.
        """
        self._dispatch_paused = True
        # Reset auto-resume timer on every pause call (idempotent extend).
        self._cancel_auto_resume_task()
        self._auto_resume_task = asyncio.create_task(
            self._auto_resume_after(auto_resume_after_seconds)
        )
        return await self._wait_for_drain(timeout_s=drain_timeout_s)

    def resume(self) -> None:
        """Resume dispatching from the queue.

        Drops the pause flag and cancels the auto-resume timer.
        The next scheduler tick picks up the accumulated queue at
        full speed. Idempotent — calling ``resume`` on an already-
        running scheduler is a no-op.
        """
        self._dispatch_paused = False
        self._cancel_auto_resume_task()

    def _cancel_auto_resume_task(self) -> None:
        """Cancel any pending auto-resume timer.

        Safe to call when no timer is active. Used by ``resume`` and
        by ``pause`` (to extend the timer on re-pause).
        """
        if self._auto_resume_task is not None and not self._auto_resume_task.done():
            self._auto_resume_task.cancel()
        self._auto_resume_task = None

    async def _auto_resume_after(self, delay_s: float) -> None:
        """Sleep ``delay_s`` then resume dispatch if still paused.

        Catches its own CancelledError so the cancel from
        ``_cancel_auto_resume_task`` doesn't propagate as
        "Task exception was never retrieved" warnings. If the timer
        actually fires (no explicit resume arrived), logs a warning
        and emits ``admin.auto_resumed`` via the audit logger so
        operators see the failsafe activated.
        """
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        if not self._dispatch_paused:
            return
        self._dispatch_paused = False
        logger.warning(
            "scheduler.auto_resumed",
            after_seconds=delay_s,
            reason="explicit resume never arrived; failsafe activated",
        )
        try:
            await self.audit.record(event="admin.auto_resumed")
        except Exception as exc:
            logger.warning("scheduler.auto_resume_audit_failed", error=str(exc))

    async def _wait_for_drain(self, timeout_s: float) -> bool:
        """Poll until ``_in_flight_count`` is zero or timeout fires.

        Polled rather than event-driven because in-flight tracking is
        an increment/decrement pair around ``_forward_single`` —
        adding an event-set would couple the dispatch path to the
        pause machinery. Polling at 100ms granularity is cheap and
        keeps the dispatch path uncoupled from pause state.

        Args:
            timeout_s: Max seconds to wait.

        Returns:
            True if the counter drained to zero, False if timeout
            fired first.
        """
        deadline = time.monotonic() + timeout_s
        while self._in_flight_count > 0:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.1)
        return True

    def in_flight_count(self) -> int:
        """Number of envelopes currently mid-dispatch to Ollama.

        Used by ``/api/marshal/admin/pause`` response payload to
        surface drain status (so operators see whether the scheduler
        is actually idle when pause returns).
        """
        return self._in_flight_count

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
        # Cancel any pending auto-resume timer so it doesn't leak past
        # shutdown. Safe to call even if no timer was active.
        self._cancel_auto_resume_task()
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

        Pause: when ``_dispatch_paused`` is set, the entire normal tick
        body is skipped and only ``_tick_bypass_only`` runs. Non-bypass
        envelopes pile up in the queue (no client-visible 503s); they
        drain at full speed once ``resume`` flips the flag.
        """
        if self._dispatch_paused:
            await self._tick_bypass_only()
            return

        # Step 0: Pull any Ollama-side evictions surfaced by the memory
        # poller into ``_needs_reload``. Bin-packing later in this tick
        # (Step 4) skips the per-model cooldown for these so the just-
        # evicted model gets a fresh preload immediately, instead of
        # potentially waiting behind a preload-backoff window from an
        # unrelated earlier failure. Pre-v0.6.5 the scheduler reacted
        # to these evictions only on the next dispatch cycle that
        # naturally tried to load the model — adding up to one full
        # tick of latency between detection and reload.
        evicted = self.memory.take_recent_unexpected_unloads()
        for model, instance_url in evicted:
            self._needs_reload.add((model, instance_url))
            logger.info(
                "scheduler.eviction_recorded_for_reload",
                model=model,
                instance=instance_url,
            )

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

        # Step 7: Roll memory's observed-unexpected-unloads counter up
        # into the persisted SchedulerMetrics. The memory poll loop ran
        # independently in the background; we drain the count here so
        # the dashboard sees a single authoritative number.
        unexpected = self.memory.take_unexpected_unload_count()
        if unexpected > 0:
            self.metrics.unexpected_unloads += unexpected

    async def _tick_bypass_only(self) -> None:
        """During admin pause, dispatch only bypass-flagged envelopes.

        Bypass envelopes are tagged when the request handler sees a
        valid ``X-Marshal-Test-Bypass`` header (matching
        ``admin.test_bypass_token``). They represent integration test
        traffic that needs to flow even while production dispatch is
        frozen.

        The flow per loaded-or-loadable model:
          1. Pre-check the queue for any pending bypass envelopes
             (cheap — peeks without removing).
          2. If the model isn't loaded, force-load it via the same
             eviction-aware path CRITICAL preemption uses. Test traffic
             gets the strongest dispatch guarantee — no point pausing
             dispatch for tests if the tests can't actually run.
          3. Pop only the bypass envelopes for that model and dispatch
             them via the standard ``_process_batch`` path. Non-bypass
             envelopes for the same model stay in the queue.

        Skip-counter incrementing and idle-eviction don't run during
        pause — both would create misleading state for the post-resume
        drain.
        """
        models = await self.queues.peek_models()
        for model in models:
            pending = await self.queues.pending_for_model(model)
            if not any(e.bypass_pause for e in pending):
                continue
            if not self.memory.is_loaded(model):
                num_ctx = await self._max_num_ctx_for_pending(model)
                success = await self._ensure_model_loaded(model, num_ctx=num_ctx)
                if not success:
                    # Couldn't load — skip this model's bypass envelopes
                    # this tick. Will retry next tick. Better than popping
                    # them and failing the dispatches.
                    continue
            bypass_batch = await self.queues.dequeue_bypass_for_model(model)
            if bypass_batch:
                await self._process_batch(bypass_batch)

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

    def _is_in_preload_cooldown(self, model: str, now: float | None = None) -> bool:
        """Check whether ``model``'s preload is currently parked on cooldown.

        Returns True when the model has prior failures and the cooldown
        deadline hasn't yet passed. Pass ``now`` to override the clock
        for tests.
        """
        state = self._preload_failures.get(model)
        if state is None:
            return False
        ts = time.monotonic() if now is None else now
        return ts < state.cooldown_until

    def _record_preload_failure(self, model: str) -> int:
        """Bump per-model failure count and recompute the backoff cooldown.

        Reuses ``backoff_delay`` from ``retry.py`` for the same
        full-jitter exponential as Hop 2 forward retries. Returns the
        new consecutive-failure count so callers can decide whether to
        give up.
        """
        cfg = self.config.scheduler
        prior = self._preload_failures.get(model)
        failures = (prior.consecutive_failures + 1) if prior else 1
        delay = backoff_delay(
            failures,
            cfg.preload_backoff_base_s,
            cfg.preload_backoff_max_s,
        )
        self._preload_failures[model] = _PreloadFailureState(
            consecutive_failures=failures,
            cooldown_until=time.monotonic() + delay,
        )
        logger.warning(
            "scheduler.preload_failure_recorded",
            model=model,
            consecutive_failures=failures,
            cooldown_s=round(delay, 2),
        )
        return failures

    def _clear_preload_failure(self, model: str) -> None:
        """Drop ``model``'s failure state after a successful preload or giveup."""
        if self._preload_failures.pop(model, None) is not None:
            logger.info("scheduler.preload_failure_cleared", model=model)

    async def _give_up_on_preload(self, model: str) -> None:
        """Drain ``model``'s queue and fail every envelope with PreloadFailedError.

        Called after the per-model failure counter reaches
        ``preload_max_consecutive_failures``. Clears the failure state
        so the next request for the same model starts a fresh backoff
        sequence — the cooldown is per-batch, not permanent.
        """
        cfg = self.config.scheduler
        drained = await self.queues.dequeue_batch(model)
        error = PreloadFailedError(
            f"preload failed {cfg.preload_max_consecutive_failures} consecutive "
            f"times for model {model!r}; giving up"
        )
        for env in drained:
            env.fail(error)
        logger.error(
            "scheduler.preload_giving_up",
            model=model,
            consecutive_failures=cfg.preload_max_consecutive_failures,
            failed_envelopes=len(drained),
        )
        try:
            # Use ``error_type`` to align with the ``request.failed``
            # convention. The numeric ``failed_envelopes`` is in the
            # structured log above; ``AuditLogger.record`` doesn't accept
            # arbitrary kwargs, so passing ``failed_envelopes=...`` here
            # would TypeError and the audit entry would be silently
            # dropped by the surrounding except.
            await self.audit.record(
                "scheduler.preload_giving_up",
                model=model,
                error_type="PreloadFailedError",
            )
        except Exception as exc:
            logger.warning(
                "scheduler.preload_giving_up_audit_failed",
                model=model,
                error=str(exc),
            )
        self._clear_preload_failure(model)

    async def _attempt_preload(
        self,
        model: str,
        num_ctx: int | None,
        instance_url: str,
        *,
        bypass_cooldown: bool = False,
    ) -> bool:
        """Wrap ``lifecycle.preload`` with cooldown + backoff + giveup.

        Production preload sites in the scheduler call this instead of
        ``lifecycle.preload`` directly so the per-model failure state
        machine applies uniformly. Returns True only on a successful
        load; False on cooldown skip, transient failure, or giveup.

        Threads the configured Hop 2 timeout
        (``scheduler.ollama_forward_timeout_s``) through to
        ``lifecycle.preload`` so the load call shares the same
        wall-clock budget as forward calls.

        ``bypass_cooldown`` (v0.6.5): when True, skip the per-model
        cooldown check. Used by the bin-packer for models flagged in
        ``_needs_reload`` (Ollama-side eviction, not a marshal preload
        failure) so the eviction-triggered reload doesn't sit behind
        a cooldown from an unrelated earlier failure.
        """
        if not bypass_cooldown and self._is_in_preload_cooldown(model):
            return False
        # Defense in depth: server.py's _is_known_model fail-fast at
        # request entry catches the common case, but a model removed
        # from Ollama between enqueue and preload time would otherwise
        # ride the full retry budget. Passing the registry predicate
        # through makes lifecycle short-circuit before /api/generate.
        success = await self.lifecycle.preload(
            model,
            num_ctx=num_ctx,
            instance_url=instance_url,
            load_timeout_s=self.config.scheduler.ollama_forward_timeout_s,
            is_known_model_check=self.registry.is_known_model,
        )
        if success:
            self._clear_preload_failure(model)
            # Claim ownership so shutdown.unload_models only tears down
            # models THIS marshal loaded — not models loaded by another
            # marshal or a human against the same Ollama. Auto-released
            # by MemoryManager when the next /api/ps poll observes the
            # model gone (whether marshal-initiated or external).
            self.memory.mark_owned(model, instance_url)
            return True
        failures = self._record_preload_failure(model)
        if failures >= self.config.scheduler.preload_max_consecutive_failures:
            await self._give_up_on_preload(model)
        return False

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

            instance_url = self.memory.find_instance_for(model_name)
            logger.info(
                "scheduler.idle_evict",
                model=model_name,
                idle_s=round(idle_s, 1),
                threshold_s=threshold_s,
                instance=instance_url,
            )
            self.memory.mark_intended_unload(model_name, instance_url=instance_url)
            success = await self.lifecycle.unload(model_name, instance_url=instance_url)
            if success:
                self.metrics.evictions += 1
                self._last_activity.pop(model_name, None)
                self._active_programs.pop(model_name, None)
                await self.memory.refresh()
            # Only evict one per tick — keeps the loop lightweight.
            break

    async def _forward_loaded_model_requests(self) -> None:
        """Forward all pending requests whose model is already loaded.

        Also handles reload-on-need (Surface C1 Dim 4): if any pending
        envelope for a loaded model needs more num_ctx than the model
        currently has allocated, trigger a reload first.
        """
        loaded = self.memory.get_loaded_models()
        for model_name in loaded:
            pending = await self.queues.pending_count(model_name)
            if pending == 0:
                continue

            instance_url = self.memory.find_instance_for(model_name)
            if instance_url is None:
                # Race: poll loop unloaded the model between the
                # ``get_loaded_models`` snapshot above and now. Skip
                # this dispatch tick — the next tick will see the
                # model truly absent and route through ``ensure_model_loaded``
                # instead. Without this guard, the batch would dispatch
                # with ``envelope.instance_url=None`` and forward_request
                # would fall back to ``config.ollama.host`` (the
                # primary), routing to an instance that no longer
                # holds the model in multi-instance setups.
                continue

            # Check whether any pending envelope needs more context
            # than the model's current slot allocation. If so, reload
            # at the larger size before dispatching anything.
            requested = await self._max_num_ctx_for_pending(model_name)
            if requested is not None and self.memory.needs_reload(
                model_name, requested, instance_url=instance_url
            ):
                # _ensure_model_loaded handles drain-before-reload +
                # unload + reload + record_allocated_num_ctx + metric.
                await self._ensure_model_loaded(model_name, num_ctx=requested)
                # Skip dispatch this tick; the next tick will pick up
                # whatever's now ready (the drain-before-reload may
                # have already served some envelopes).
                continue

            batch = await self.queues.dequeue_batch(model_name)
            if batch:
                # Tag each envelope with the instance currently holding
                # the model so forward_request hits the right upstream.
                self._tag_batch_with_instance(batch, instance_url)
                await self._process_batch(batch)

    def _tag_batch_with_instance(
        self,
        batch: list[RequestEnvelope],
        instance_url: str | None,
    ) -> None:
        """Stamp each envelope with the instance URL chosen at dispatch.

        The scheduler tags envelopes here rather than at enqueue time
        because routing decisions can change between enqueue and
        dispatch (e.g. a model gets unloaded then reloaded on a
        different tier in between). ``forward_request`` reads
        ``envelope.instance_url`` to pick the upstream.

        Populates ``tier_label`` and ``routing_reason`` too so
        steady-state already-loaded dispatches show up in the audit log
        with the same routing context as cold-start dispatches. For
        these paths the routing reason is ``ALREADY_LOADED`` — the
        model was already on the chosen instance from a previous
        decision.
        """
        if instance_url is None:
            return
        # Look up the OllamaInstance to populate tier_label. Slow path
        # is O(N) but N is the instance count (typically 1-3), and this
        # runs once per batch dispatch, not per envelope.
        tier_label: str | None = None
        for inst in self.memory.instances:
            if inst.url == instance_url:
                tier_label = inst.tier_label
                break
        for env in batch:
            if env.instance_url is None:
                env.instance_url = instance_url
                env.tier_label = tier_label
                env.routing_reason = RoutingReason.ALREADY_LOADED.value

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
            num_ctx = await self._max_num_ctx_for_pending(envelope.model)
            await self._ensure_model_loaded(envelope.model, num_ctx=num_ctx)
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
            num_ctx = await self._max_num_ctx_for_pending(envelope.model)
            await self._ensure_model_loaded(envelope.model, num_ctx=num_ctx)
            break  # Load one forced model per tick to avoid thrashing

    async def _bin_pack_models(self) -> None:
        """Load models that fit in remaining VRAM, FIFO order.

        Uses ``routing.pick_instance`` to choose the target tier so
        memory-pressure failover applies to bin-packed loads too (not
        just forced loads from ``_ensure_model_loaded``).
        """
        all_pending = await self.queues.get_all_sorted_by_arrival()

        # Collect unique models not yet loaded, in arrival order
        seen: set[str] = set()
        models_to_try: list[str] = []
        for envelope in all_pending:
            if envelope.model not in seen and not self.memory.is_loaded(envelope.model):
                seen.add(envelope.model)
                models_to_try.append(envelope.model)

        skipped_models: list[str] = []
        # Models with any pending ``_needs_reload`` entry skip the
        # cooldown gate — Ollama just evicted them, marshal didn't fail
        # to load them, so the cooldown (which protects against marshal-
        # preload-storm) doesn't apply.
        reload_priority_models = {m for m, _ in self._needs_reload}
        for model in models_to_try:
            # Per-model preload backoff (v0.6.4+): if a recent preload
            # failed and the cooldown window hasn't passed, skip without
            # touching Ollama. Avoids the storm of ~10 preload calls/sec
            # when Ollama is briefly unreachable. Bypass for any model
            # the memory poller just flagged as evicted by Ollama (Bug 4,
            # v0.6.5) — those weren't marshal-preload failures.
            if model not in reload_priority_models and self._is_in_preload_cooldown(
                model
            ):
                continue
            model_size = await self.registry.get_or_estimate_size(model)
            if self.memory.can_fit_model(model_size):
                num_ctx = await self._max_num_ctx_for_pending(model)
                decision = await self._resolve_routing(model, num_ctx)
                # Honor any unload_from from the decision (rare for
                # cold-start bin-pack but possible if a stale q4 copy
                # sits around).
                for stale in decision.unload_from:
                    self.memory.mark_intended_unload(model, instance_url=stale.url)
                    await self.lifecycle.unload(model, instance_url=stale.url)
                logger.info(
                    "scheduler.bin_pack_load",
                    model=model,
                    size_gb=round(model_size / (1024**3), 2),
                    available_gb=round(self.memory.available_vram() / (1024**3), 2),
                    num_ctx=num_ctx,
                    instance=decision.instance.url,
                    routing_reason=decision.reason.value,
                    needs_reload=model in reload_priority_models,
                )
                success = await self._attempt_preload(
                    model,
                    num_ctx=num_ctx,
                    instance_url=decision.instance.url,
                    bypass_cooldown=model in reload_priority_models,
                )
                # Drop the matching (model, instance) entry whether
                # success or failure. Two reasons: (a) on failure, leaving
                # the entry would re-bypass the cooldown next tick and
                # convert v0.6.4's jittered backoff into per-tick
                # hammering against a flapping Ollama; (b) on success,
                # only the chosen instance is freshly loaded — entries
                # for other instances must survive so each gets its own
                # reload chance on a future tick.
                self._needs_reload.discard((model, decision.instance.url))
                if success:
                    self.metrics.model_swaps += 1
                    if num_ctx is not None:
                        self.memory.record_allocated_num_ctx(
                            model, num_ctx, instance_url=decision.instance.url
                        )
                    await self.memory.refresh()
                    await self._tag_pending_with_decision(model, decision)
            else:
                skipped_models.append(model)

        # Only increment skip counters for models that were actually
        # passed over this round (didn't fit in VRAM), not every tick.
        # CRITICAL-priority programs are exempt from the fairness floor
        # (their dedicated preemption path in
        # _handle_critical_preemption already guarantees forced load on
        # the next tick) so we exclude them here. Without this, a
        # CRITICAL request stuck behind a single-preemption-per-tick
        # queue would have its skip_count climb anyway and surface
        # spurious "forced_load" log noise.
        critical_program_ids = {
            program_id
            for program_id, profile in self.config.programs.items()
            if profile.priority == Priority.CRITICAL
        }
        for model in skipped_models:
            await self.queues.increment_skips_for_model(
                model, exclude_program_ids=critical_program_ids
            )

    @staticmethod
    def _envelope_num_ctx(envelope: RequestEnvelope) -> int | None:
        """Read the injected `options.num_ctx` from an envelope's request body.

        Returns None when the body has no options or the value isn't an
        int. Used to decide what slot size to preload at, and whether
        a reload is needed before dispatch.
        """
        options = envelope.request_body.get("options")
        if not isinstance(options, dict):
            return None
        value = options.get("num_ctx")
        if isinstance(value, int) and value > 0:
            return value
        return None

    async def _max_num_ctx_for_pending(self, model: str) -> int | None:
        """Maximum injected `num_ctx` across all pending envelopes for `model`.

        Used to size the initial preload so that small first-arrival
        prompts don't get a tiny slot that gets immediately reloaded
        when a later (larger) request dispatches.
        """
        pending = await self.queues.pending_for_model(model)
        max_ctx: int | None = None
        for env in pending:
            n = self._envelope_num_ctx(env)
            if n is None:
                continue
            if max_ctx is None or n > max_ctx:
                max_ctx = n
        return max_ctx

    async def _non_idle_models_per_instance(self) -> dict[str, set[str]]:
        """For each instance, set of currently loaded models that are non-idle.

        "Non-idle" = has pending requests in queue OR has dispatched
        recently (active program tracking). The B-rule (avoid evicting
        non-idle work) uses this to decide whether falling back to a
        lower-precision instance is preferable to evicting work on the
        primary.
        """
        pending = await self.queues.pending_by_model()
        active = self._active_programs  # model -> {prog: ts}
        result: dict[str, set[str]] = {}
        for inst in self.memory.instances:
            here = self.memory.get_loaded_models_on(inst.url)
            non_idle = {
                name
                for name in here
                if pending.get(name, 0) > 0 or bool(active.get(name))
            }
            result[inst.url] = non_idle
        return result

    async def _build_fit_probe(
        self,
        model: str,
        num_ctx: int | None,
        non_idle: dict[str, set[str]],
    ) -> dict[str, FitProbe]:
        """Per-instance FitProbe map keyed by ``instance.url``.

        Asks ``MemoryManager.probe_fit`` for each configured instance,
        scaling the model footprint estimate by the instance's KV
        precision multiplier so a q4_0 instance correctly looks
        "smaller" for fit purposes than f16.
        """
        probe: dict[str, FitProbe] = {}
        for inst in self.memory.instances:
            footprint = await self.registry.get_total_footprint(
                model,
                num_ctx if num_ctx is not None else 0,
                inst.kv_cache_type,
            )
            probe[inst.url] = self.memory.probe_fit(
                instance_url=inst.url,
                model_size=footprint,
                non_idle_loaded_on_instance=non_idle.get(inst.url, set()),
            )
        return probe

    async def _resolve_routing(
        self,
        model: str,
        num_ctx: int | None,
    ) -> RoutingDecision:
        """Build a routing decision for `model` based on current memory state.

        Pure-function wrapper: gathers state, calls
        ``routing.pick_instance``, returns the structured decision.
        Caller uses the decision to drive preload (and any
        ``unload_from`` cleanup of stale-tier copies).
        """
        non_idle = await self._non_idle_models_per_instance()
        probe = await self._build_fit_probe(model, num_ctx, non_idle)
        state = RoutingState(
            model_name=model,
            requested_num_ctx=num_ctx if num_ctx is not None else 0,
            instances=self.memory.instances,
            loaded_on=self.memory.loaded_on(),
        )
        return pick_instance(state, probe)

    async def _ensure_model_loaded(
        self, model: str, num_ctx: int | None = None
    ) -> bool:
        """Ensure a model is loaded with at least `num_ctx` slot allocation.

        Routes to the right Ollama instance (memory-pressure failover
        across f16/q8_0/q4_0 tiers when configured). When `num_ctx` is
        set and the model is already loaded on the chosen instance but
        at a smaller allocated slot size, this triggers a reload-on-
        need: unload it on that instance, then preload at the larger
        size. Reload count is tracked in `metrics.reload_count`.

        Args:
            model: The model to load.
            num_ctx: Required minimum slot allocation; None means "any
                allocation will do".

        Returns:
            True if the model is now loaded at the requested size.
        """
        # Per-model preload backoff (v0.6.4+): bail before doing any
        # eviction work when a recent preload failed and we're still
        # within the cooldown window. Without the early return,
        # ``_evict_one`` would tear down loaded models to make room for
        # a preload that immediately gets skipped, leaving VRAM empty.
        # Bug 4 (v0.6.5): bypass the cooldown when the memory poller
        # flagged this model as Ollama-side evicted — that wasn't a
        # marshal preload failure, so the cooldown shouldn't park
        # CRITICAL/unskippable preemption either.
        bypass_cooldown = any(m == model for m, _ in self._needs_reload)
        if not bypass_cooldown and self._is_in_preload_cooldown(model):
            return False

        decision = await self._resolve_routing(model, num_ctx)
        chosen_url = decision.instance.url

        # Unload stale-tier copies of the same model first (e.g. a
        # promotion off q4_0 returns ``unload_from=[q4]``). Done
        # BEFORE the new preload so VRAM frees up first.
        for stale in decision.unload_from:
            self.memory.mark_intended_unload(model, instance_url=stale.url)
            await self.lifecycle.unload(model, instance_url=stale.url)
        if decision.unload_from:
            await self.memory.refresh()

        is_reload = False
        if self.memory.is_loaded_on(model, chosen_url):
            if num_ctx is None or not self.memory.needs_reload(
                model, num_ctx, instance_url=chosen_url
            ):
                return True
            # Reload-on-need: existing slot on the chosen instance is
            # too small for an incoming request. We do NOT drain
            # pending requests first — the request that triggered this
            # reload would dispatch against the OLD smaller slot and
            # Ollama would silently truncate it, defeating the entire
            # point of the surface ("Marshal NEVER silently truncates a
            # real prompt"). Just unload now and let the next tick
            # dispatch against the new larger slot.
            current = self.memory.get_allocated_num_ctx(model, instance_url=chosen_url)
            logger.info(
                "scheduler.reload_for_num_ctx",
                model=model,
                current_num_ctx=current,
                requested_num_ctx=num_ctx,
                instance=chosen_url,
            )
            self.memory.mark_intended_unload(model, instance_url=chosen_url)
            await self.lifecycle.unload(model, instance_url=chosen_url)
            await self.memory.refresh()
            is_reload = True

        model_size = await self.registry.get_or_estimate_size(model)

        # Evict if needed (global budget — see MemoryManager docstring).
        while not self.memory.can_fit_model(model_size):
            evicted = await self._evict_one(model)
            if not evicted:
                logger.error(
                    "scheduler.cannot_fit_model",
                    model=model,
                    size_gb=round(model_size / (1024**3), 2),
                    available_gb=round(self.memory.available_vram() / (1024**3), 2),
                )
                # Failed preload after a successful unload is the worst
                # case: model is now gone but we couldn't replace it.
                # Mark allocation as 0 (sentinel) so future calls to
                # `needs_reload` return True until a successful preload
                # writes the real value, instead of "defensively"
                # returning False (which would silently truncate).
                if is_reload:
                    self.memory.record_allocated_num_ctx(
                        model, 0, instance_url=chosen_url
                    )
                return False

        success = await self._attempt_preload(
            model,
            num_ctx=num_ctx,
            instance_url=chosen_url,
            bypass_cooldown=bypass_cooldown,
        )
        # Drop the matching (model, instance) eviction record either
        # way — same rationale as bin-pack: leaving it on failure
        # converts the v0.6.4 jittered backoff into per-tick hammering.
        self._needs_reload.discard((model, chosen_url))
        if success:
            self.metrics.model_swaps += 1
            if is_reload:
                self.metrics.reload_count += 1
            if num_ctx is not None:
                self.memory.record_allocated_num_ctx(
                    model, num_ctx, instance_url=chosen_url
                )
            await self.memory.refresh()
            # Stamp pending envelopes with the routing decision so
            # forward_request hits the right upstream and audit log
            # entries record the tier_label + reason.
            await self._tag_pending_with_decision(model, decision)
        elif is_reload:
            # Same sentinel as above — model was unloaded then preload
            # failed. Don't pretend we have an allocation we don't.
            self.memory.record_allocated_num_ctx(model, 0, instance_url=chosen_url)
        return success

    async def _tag_pending_with_decision(
        self,
        model: str,
        decision: RoutingDecision,
    ) -> None:
        """Stamp pending envelopes for `model` with the routing decision.

        Called after a successful preload so the next dispatch tick
        sees ``envelope.instance_url`` set and ``forward_request``
        targets the right upstream. Untagged envelopes (e.g. enqueued
        before this preload) get the decision; envelopes already tagged
        from a prior decision keep their original tag.
        """
        pending = await self.queues.pending_for_model(model)
        for env in pending:
            if env.instance_url is None:
                env.instance_url = decision.instance.url
                env.tier_label = decision.instance.tier_label
                env.routing_reason = decision.reason.value

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
        target_instance = self.memory.find_instance_for(target)
        logger.info(
            "scheduler.evicting",
            model=target,
            pending=pending_counts.get(target, 0),
            needed_for=needed_for,
            instance=target_instance,
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
                self._tag_batch_with_instance(batch, target_instance)
                await self._process_batch(batch)

        self.memory.mark_intended_unload(target, instance_url=target_instance)
        success = await self.lifecycle.unload(target, instance_url=target_instance)
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

    def _resolve_retry_attempts(self, envelope: RequestEnvelope) -> int:
        """Resolve the effective retry budget for one envelope.

        Precedence (highest first):
        1. `envelope.retry_max_override` (from `X-Marshal-Retry-Max`
           header) — explicit per-request opt-in/out.
        2. `config.retry.max_attempts` if `retry.enabled`.
        3. 1 (no retry) when retry is globally disabled.

        Streaming requests always resolve to 1 attempt (the retry
        helper also short-circuits, but resolving here makes the
        no-retry fast path explicit at the call site).
        """
        if envelope.stream:
            return 1
        if envelope.retry_max_override is not None:
            # Clamp to >= 1 — `0` from the header means "no retries"
            # which is still a single attempt.
            return max(1, envelope.retry_max_override)
        retry_cfg = self.config.retry
        if not retry_cfg.enabled:
            return 1
        return retry_cfg.max_attempts

    async def _forward_single(self, envelope: RequestEnvelope) -> None:
        """Forward a single request to Ollama and complete the envelope.

        Wraps `forward_request` in `call_with_retry` for non-streaming
        non-idempotent calls. Streaming requests are NEVER retried (the
        retry helper short-circuits), and `retry_max_override` on the
        envelope can disable or extend retry per-request.

        Maintains ``_in_flight_count`` for the admin pause endpoint's
        drain wait — increment at entry, decrement in the outer
        ``finally`` block so the counter never drifts even when the
        dispatch raises or is cancelled mid-stream.

        Args:
            envelope: The request to forward.
        """
        self._in_flight_count += 1
        try:
            await self._forward_single_inner(envelope)
        finally:
            self._in_flight_count = max(0, self._in_flight_count - 1)

    async def _forward_single_inner(self, envelope: RequestEnvelope) -> None:
        """Inner dispatch body for ``_forward_single``.

        Separated so the outer ``_forward_single`` can do clean
        ``_in_flight_count`` bookkeeping without nesting the entire
        dispatch path inside a try/finally.
        """
        max_attempts = self._resolve_retry_attempts(envelope)
        retry_cfg = self.config.retry
        # Compute whether this dispatch site allows ReadTimeout retry.
        # Embeddings are idempotent — always retryable. Other endpoints
        # only retry ReadTimeout when the operator opts in.
        allow_read_retry = retry_cfg.read_timeouts or envelope.endpoint in (
            "/api/embeddings",
            "/v1/embeddings",
        )
        # Per-envelope upstream URL set by the routing decision. Falls
        # back to the primary instance for legacy single-instance
        # paths (and the v0.4.x test fixtures that don't tag envelopes).
        ollama_host = (
            envelope.instance_url
            if envelope.instance_url is not None
            else self.config.ollama.host
        )
        try:
            if max_attempts <= 1:
                # No-retry fast path: single attempt, no helper overhead.
                result = await forward_request(
                    ollama_host=ollama_host,
                    endpoint=envelope.endpoint,
                    request_body=envelope.request_body,
                    stream=envelope.stream,
                    timeout_s=envelope.ollama_forward_timeout_s,
                )
            else:
                # request_id correlates retries in logs; envelope object
                # id is unique within the process and cheap.
                request_id = f"{envelope.model}:{id(envelope):x}"
                result, attempts_used, exhausted = await call_with_retry(
                    forward_request,
                    ollama_host=ollama_host,
                    endpoint=envelope.endpoint,
                    request_body=envelope.request_body,
                    stream=envelope.stream,
                    timeout_s=envelope.ollama_forward_timeout_s,
                    max_attempts=max_attempts,
                    base_delay_s=retry_cfg.base_delay_s,
                    max_delay_s=retry_cfg.max_delay_s,
                    retry_read_timeouts=allow_read_retry,
                    request_id=request_id,
                )
                if attempts_used > 1:
                    # The helper used at least one retry. Always count
                    # the attempts as "attempted" so operators see retry
                    # frequency. But only count "succeeded" when the
                    # final attempt actually returned a healthy response
                    # — `exhausted=True` means we burned the retry budget
                    # on a 502/503/504 and the caller is still getting
                    # the failure. Keeping that out of `retries_succeeded`
                    # prevents `marshal doctor` from misreporting Ollama
                    # health as fine when it's actually flaking.
                    self.metrics.retries_attempted += attempts_used - 1
                    if not exhausted:
                        self.metrics.retries_succeeded += 1
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
                instance_url=envelope.instance_url,
                tier_label=envelope.tier_label,
                routing_reason=envelope.routing_reason,
            )
        except Exception as exc:
            # When retry was active and exhausted, the helper raised the
            # last exception — count those attempts here so the metric
            # captures both successful-retry and exhausted-retry paths.
            #
            # BUT only when the exception was ACTUALLY retried: if a
            # non-retryable exception (e.g. ReadTimeout with
            # read_timeouts=False) propagates, call_with_retry raises it
            # on attempt 1 without consuming any retry budget. Counting
            # those would mark every transient ReadTimeout as
            # "max_attempts - 1 retries used" when zero were.
            from ollama_marshal.retry import (
                SAFE_RETRY_EXCEPTIONS,
                UNSAFE_RETRY_EXCEPTIONS,
            )

            retryable: tuple[type[Exception], ...] = SAFE_RETRY_EXCEPTIONS
            if allow_read_retry:
                retryable = retryable + UNSAFE_RETRY_EXCEPTIONS
            if max_attempts > 1 and isinstance(exc, retryable):
                self.metrics.retries_attempted += max_attempts - 1
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
                instance_url=envelope.instance_url,
                tier_label=envelope.tier_label,
                routing_reason=envelope.routing_reason,
            )
