"""Per-model request queues with skip tracking for fairness."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


class PreloadFailedError(Exception):
    """Scheduler exhausted its per-model preload retry budget.

    Set as ``RequestEnvelope.error`` by the scheduler when
    ``lifecycle.preload`` has failed
    ``scheduler.preload_max_consecutive_failures`` consecutive times for
    the same model — at which point keeping the envelope queued forever
    is worse than failing it. Surfaced to clients through the standard
    error response path (a 502 today; v0.6.5 will propagate the class
    name + reason in the response body).
    """


@dataclass
class RequestEnvelope:
    """Wraps an incoming request with scheduling metadata.

    Attributes:
        model: The model name this request targets.
        program_id: Identifier of the calling program.
        request_body: The raw request body to forward to Ollama.
        endpoint: The Ollama endpoint path (e.g., '/api/chat').
        stream: Whether this is a streaming request.
        arrived_at: Monotonic timestamp of arrival.
        skip_count: Number of times this request has been skipped by the scheduler.
        done_event: Asyncio event set when the response is ready.
        response: The response object, populated by the scheduler.
        error: Any error that occurred during processing.
    """

    model: str
    program_id: str
    request_body: dict[str, Any]
    endpoint: str
    stream: bool = False
    arrived_at: float = field(default_factory=time.monotonic)
    skip_count: int = 0
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    response: Any = None
    error: Exception | None = None
    # Per-request retry override from `X-Marshal-Retry-Max` header.
    # None means "use config default"; an explicit int (including 0)
    # overrides the config. Lets a client disable retry on a single
    # call (e.g. tool-calling agent that wants fail-fast) or extend
    # it for an idempotent embedding burst.
    retry_max_override: int | None = None
    # Multi-instance routing decision (v0.5.0+). Populated by the
    # scheduler from ``routing.pick_instance`` before the envelope is
    # dispatched. ``forward_request`` reads ``instance_url`` to pick
    # the right Ollama upstream. ``tier_label`` and ``routing_reason``
    # ride along for audit-log enrichment. None on single-instance
    # setups or before routing has decided — call sites must fall back
    # to the primary instance URL.
    instance_url: str | None = None
    tier_label: str | None = None
    routing_reason: str | None = None
    # Admin pause bypass (v0.6.0+). When the scheduler's dispatch is
    # paused via ``/api/marshal/admin/pause``, normal envelopes sit in
    # the queue waiting for resume. Envelopes flagged with
    # ``bypass_pause=True`` (set when the request carries the
    # ``X-Marshal-Test-Bypass`` header matching ``admin.test_bypass_token``)
    # dispatch immediately even during pause. Used by integration tests
    # so they can fire requests against a paused prod marshal without
    # blocking on the queue freeze.
    bypass_pause: bool = False
    # Hop 2 forward timeout (v0.6.4+). Wall-clock budget for the
    # marshal→Ollama HTTP forward in seconds. Resolved by the request
    # handler from the X-Request-Timeout header (if present) or
    # ``scheduler.ollama_forward_timeout_s`` config default. The
    # scheduler reads this when calling ``forward_request``. Default
    # 3600 (1h) matches the config default for legacy tests / direct
    # construction paths that don't set it explicitly.
    ollama_forward_timeout_s: int = 3600

    def increment_skip(self) -> None:
        """Increment the skip counter for this request."""
        self.skip_count += 1

    def is_unskippable(self, max_skips: int) -> bool:
        """Check if this request has exceeded its skip limit.

        Args:
            max_skips: Maximum allowed skips before forced service.

        Returns:
            True if the request must be served next.
        """
        return self.skip_count >= max_skips

    def complete(self, response: Any) -> None:
        """Mark this request as completed with a response.

        Args:
            response: The response to deliver to the waiting caller.
        """
        self.response = response
        self.done_event.set()

    def fail(self, error: Exception) -> None:
        """Mark this request as failed with an error.

        Args:
            error: The exception that caused the failure.
        """
        self.error = error
        self.done_event.set()

    @property
    def wait_time(self) -> float:
        """Seconds this request has been waiting."""
        return time.monotonic() - self.arrived_at


class ModelQueues:
    """Thread-safe per-model request queues.

    Organizes incoming requests into separate FIFO queues by model name,
    with global arrival-order tracking for fair scheduling.
    """

    def __init__(self) -> None:
        self._queues: dict[str, deque[RequestEnvelope]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def enqueue(self, envelope: RequestEnvelope) -> None:
        """Add a request to the appropriate model queue.

        Args:
            envelope: The request envelope to enqueue.
        """
        async with self._lock:
            self._queues[envelope.model].append(envelope)

    async def dequeue(self, model: str) -> RequestEnvelope | None:
        """Remove and return the next request for a model.

        Args:
            model: The model name to dequeue from.

        Returns:
            The next RequestEnvelope, or None if the queue is empty.
        """
        async with self._lock:
            queue = self._queues.get(model)
            if queue:
                return queue.popleft()
            return None

    async def dequeue_batch(
        self, model: str, max_count: int | None = None
    ) -> list[RequestEnvelope]:
        """Remove and return multiple requests for a model.

        Args:
            model: The model name to dequeue from.
            max_count: Maximum number of requests to dequeue. None means all.

        Returns:
            List of RequestEnvelopes, possibly empty.
        """
        async with self._lock:
            queue = self._queues.get(model)
            if not queue:
                return []
            if max_count is None:
                batch = list(queue)
                queue.clear()
            else:
                batch = []
                for _ in range(min(max_count, len(queue))):
                    batch.append(queue.popleft())
            return batch

    async def peek_models(self) -> list[str]:
        """Get list of model names that have pending requests.

        Returns:
            List of model names with non-empty queues.
        """
        async with self._lock:
            return [model for model, queue in self._queues.items() if queue]

    async def pending_count(self, model: str) -> int:
        """Get the number of pending requests for a model.

        Args:
            model: The model name.

        Returns:
            Number of pending requests.
        """
        async with self._lock:
            queue = self._queues.get(model)
            return len(queue) if queue else 0

    async def pending_for_model(self, model: str) -> list[RequestEnvelope]:
        """Snapshot of pending envelopes for a single model.

        Does NOT remove from the queue — used by the scheduler to peek
        at envelope contents (e.g. compute the max num_ctx across all
        pending requests for a model before preload).

        Args:
            model: The model name.

        Returns:
            List of pending envelopes, oldest first.
        """
        async with self._lock:
            queue = self._queues.get(model)
            return list(queue) if queue else []

    async def total_pending(self) -> int:
        """Get the total number of pending requests across all models.

        Returns:
            Total pending request count.
        """
        async with self._lock:
            return sum(len(q) for q in self._queues.values())

    async def pending_by_model(self) -> dict[str, int]:
        """Get pending request counts grouped by model.

        Returns:
            Dict mapping model name to pending count.
        """
        async with self._lock:
            return {model: len(queue) for model, queue in self._queues.items() if queue}

    async def pending_programs_by_model(self) -> dict[str, list[str]]:
        """Get unique pending-request program IDs grouped by model.

        Returns:
            Dict mapping model name to a sorted list of distinct program IDs
            that currently have queued requests for that model.
        """
        async with self._lock:
            out: dict[str, list[str]] = {}
            for model, queue in self._queues.items():
                if not queue:
                    continue
                progs = sorted({env.program_id for env in queue})
                out[model] = progs
            return out

    async def get_all_sorted_by_arrival(self) -> list[RequestEnvelope]:
        """Get all pending requests sorted by arrival time (oldest first).

        Does NOT remove requests from queues. Used by the scheduler to
        inspect the full queue state for bin-packing decisions.

        Returns:
            Sorted list of all pending requests.
        """
        async with self._lock:
            all_envelopes: list[RequestEnvelope] = []
            for queue in self._queues.values():
                all_envelopes.extend(queue)
            all_envelopes.sort(key=lambda e: e.arrived_at)
            return all_envelopes

    async def get_unskippable(self, max_skips: int) -> list[RequestEnvelope]:
        """Get all requests that have exceeded their skip limit.

        Does NOT remove requests from queues.

        Args:
            max_skips: The skip threshold.

        Returns:
            List of unskippable requests sorted by arrival time.
        """
        async with self._lock:
            unskippable: list[RequestEnvelope] = []
            for queue in self._queues.values():
                for envelope in queue:
                    if envelope.is_unskippable(max_skips):
                        unskippable.append(envelope)
            unskippable.sort(key=lambda e: e.arrived_at)
            return unskippable

    async def dequeue_bypass_for_model(self, model: str) -> list[RequestEnvelope]:
        """Pop only envelopes flagged with ``bypass_pause=True``.

        Used by the scheduler during admin pause to dispatch test
        traffic carrying ``X-Marshal-Test-Bypass`` while leaving
        non-bypass envelopes parked in the queue. Preserves arrival
        order among the popped bypass envelopes.

        Args:
            model: The model whose queue to drain bypass envelopes from.

        Returns:
            List of bypass envelopes (possibly empty). Non-bypass
            envelopes for the same model stay in the queue.
        """
        async with self._lock:
            queue = self._queues.get(model)
            if not queue:
                return []
            bypass: list[RequestEnvelope] = []
            remaining: deque[RequestEnvelope] = deque()
            for envelope in queue:
                if envelope.bypass_pause:
                    bypass.append(envelope)
                else:
                    remaining.append(envelope)
            self._queues[model] = remaining
            return bypass

    async def increment_skips_for_model(
        self,
        model: str,
        exclude_program_ids: set[str] | None = None,
    ) -> None:
        """Increment skip count for pending requests of a model.

        Args:
            model: The model whose requests were skipped.
            exclude_program_ids: Program IDs whose envelopes should
                NOT have their skip counter incremented. Used by the
                scheduler to exempt CRITICAL-priority programs from
                the fairness floor — they have a dedicated preemption
                path (``Scheduler._handle_critical_preemption``) so
                the skip-based starvation guard doesn't apply.
                Default ``None`` increments every envelope (legacy
                behavior).
        """
        excluded = exclude_program_ids or set()
        async with self._lock:
            queue = self._queues.get(model)
            if queue:
                for envelope in queue:
                    if envelope.program_id in excluded:
                        continue
                    envelope.increment_skip()
