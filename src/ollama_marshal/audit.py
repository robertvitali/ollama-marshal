"""Audit-log feature flag — buffered JSONL writer with retention/rotation.

Off by default (`audit.enabled: false`). When enabled, marshal appends
one JSON record per request lifecycle event to a file at
`audit.path`. Records contain METADATA ONLY — never prompt text or
response content — so the audit file is safe to share with compliance
teams or store long-term.

Schema (one JSON object per line):
    {"ts": "2026-04-28T03:48:50.123456+00:00",
     "event": "request.served",
     "request_id": "uuid-v4",
     "program_id": "ai-portfolio",
     "model": "qwen3.5:9b-bf16",
     "endpoint": "/api/chat",
     "wait_ms": 1234,
     "stream": false,
     "burst_size_hint": 50}

Buffered: writes are collected in memory and flushed every
`_FLUSH_INTERVAL_S` seconds OR when the buffer reaches
`_FLUSH_BATCH_SIZE`, whichever comes first. Avoids fsync-per-request
overhead while bounding worst-case data loss to one flush window.

Retention + rotation:
- `retention_days`: hourly background sweep removes lines older than N
  days. 0 disables retention.
- `max_size_mb`: when the file exceeds this size, the current file is
  rotated to `audit.jsonl.1` (and any prior `.1` to `.2`, etc., up to 5
  generations) and a fresh file is started. 0 disables rotation.

Zero new dependencies — stdlib `json`, `pathlib`, `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from ollama_marshal.config import AuditConfig

logger = structlog.get_logger()

# Per-record schema version. Embedded in every audit JSONL record so
# downstream parsers (compliance dashboards, log indexers) can detect
# schema changes and either migrate or skip incompatible records. Bump
# on field renames/removals/type changes; pure additions don't require
# a bump if parsers tolerate unknown fields.
_AUDIT_SCHEMA_VERSION = 1

# Buffered-write tuning. Flush every 100ms or 50 records, whichever first.
_FLUSH_INTERVAL_S = 0.1
_FLUSH_BATCH_SIZE = 50

# Hourly retention sweep when retention_days > 0.
_RETENTION_SWEEP_INTERVAL_S = 3600.0

# Cap on rotated-file generations to avoid filling disk if rotation runs
# faster than retention. Files older than `audit.jsonl.5` are deleted
# on rotation.
_MAX_ROTATED_GENERATIONS = 5


class AuditLogger:
    """Buffered append-only JSONL writer for request lifecycle events.

    Lifecycle:
    1. `start()` opens the buffer + spawns the flush + retention tasks.
    2. `record(...)` enqueues an event (cheap — just appends to buffer).
    3. `stop()` flushes any pending records and cancels background tasks.

    Disabled-mode shortcut: when `config.audit.enabled` is False, all
    methods are cheap no-ops so callers don't need `if audit.enabled:`
    branches everywhere.
    """

    def __init__(self, config: AuditConfig) -> None:
        self._config = config
        self._enabled = config.enabled
        self._path = Path(config.path).expanduser() if self._enabled else Path()
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._retention_task: asyncio.Task[None] | None = None
        self._stopped = False

    @property
    def enabled(self) -> bool:
        """Whether audit logging is currently active."""
        return self._enabled and not self._stopped

    @property
    def path(self) -> Path:
        """The audit file path (resolved with ~ expansion)."""
        return self._path

    async def start(self) -> None:
        """Open the audit file and spawn background flush/retention tasks."""
        if not self._enabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_task = asyncio.create_task(self._flush_loop())
        if self._config.retention_days > 0:
            self._retention_task = asyncio.create_task(self._retention_loop())
        logger.info(
            "audit.started",
            path=str(self._path),
            retention_days=self._config.retention_days,
            max_size_mb=self._config.max_size_mb,
        )

    async def stop(self) -> None:
        """Flush any pending records and stop background tasks."""
        if not self._enabled:
            return
        self._stopped = True
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        if self._retention_task is not None:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except asyncio.CancelledError:
                pass
        # Final flush.
        await self._flush_now()
        logger.info("audit.stopped", path=str(self._path))

    async def record(
        self,
        event: str,
        *,
        request_id: str | None = None,
        program_id: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        wait_ms: float | None = None,
        stream: bool | None = None,
        burst_size_hint: int | None = None,
        error_type: str | None = None,
    ) -> None:
        """Enqueue an audit record. No-op when disabled.

        Records are buffered and flushed in the background. Callers do
        NOT pass prompt text or response content — those are forbidden
        for privacy reasons.
        """
        if not self._enabled or self._stopped:
            return
        record: dict[str, Any] = {
            "schema_version": _AUDIT_SCHEMA_VERSION,
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
        }
        # Only include fields that were provided to keep records compact.
        if request_id is not None:
            record["request_id"] = request_id
        if program_id is not None:
            record["program_id"] = program_id
        if model is not None:
            record["model"] = model
        if endpoint is not None:
            record["endpoint"] = endpoint
        if wait_ms is not None:
            record["wait_ms"] = round(wait_ms, 1)
        if stream is not None:
            record["stream"] = stream
        if burst_size_hint is not None:
            record["burst_size_hint"] = burst_size_hint
        if error_type is not None:
            record["error_type"] = error_type
        async with self._lock:
            self._buffer.append(record)
            buffer_full = len(self._buffer) >= _FLUSH_BATCH_SIZE
        if buffer_full:
            await self._flush_now()

    async def _flush_loop(self) -> None:
        """Background task: flush the buffer at regular intervals."""
        while not self._stopped:
            try:
                await asyncio.sleep(_FLUSH_INTERVAL_S)
            except asyncio.CancelledError:
                return
            await self._flush_now()

    async def _flush_now(self) -> None:
        """Atomically drain the buffer and append to disk."""
        async with self._lock:
            if not self._buffer:
                return
            pending = self._buffer
            self._buffer = []
        # Disk I/O off the event loop.
        await asyncio.to_thread(self._write_lines, pending)
        # Size-based rotation check after write.
        if self._config.max_size_mb > 0:
            await asyncio.to_thread(self._maybe_rotate)

    def _write_lines(self, records: list[dict[str, Any]]) -> None:
        """Append a batch of JSONL lines. Logs (not raises) on I/O error."""
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning(
                "audit.write_failed",
                path=str(self._path),
                count=len(records),
                error=str(exc) or repr(exc),
                error_type=type(exc).__name__,
            )

    def _maybe_rotate(self) -> None:
        """Rotate the audit file if size exceeds `max_size_mb`."""
        try:
            size_bytes = self._path.stat().st_size
        except OSError:
            return
        if size_bytes < self._config.max_size_mb * 1024 * 1024:
            return
        # Shift .N → .N+1, dropping the oldest if at the cap.
        for i in range(_MAX_ROTATED_GENERATIONS, 0, -1):
            old = self._path.with_suffix(self._path.suffix + f".{i}")
            new = self._path.with_suffix(self._path.suffix + f".{i + 1}")
            if old.exists():
                if i == _MAX_ROTATED_GENERATIONS:
                    try:
                        old.unlink()
                    except OSError:
                        pass
                else:
                    try:
                        old.rename(new)
                    except OSError:
                        pass
        # Move current to .1 and start fresh.
        try:
            self._path.rename(self._path.with_suffix(self._path.suffix + ".1"))
        except OSError as exc:
            logger.warning(
                "audit.rotate_failed",
                path=str(self._path),
                error=str(exc) or repr(exc),
            )

    async def _retention_loop(self) -> None:
        """Background task: hourly sweep deleting lines past retention.

        Held under `self._lock` for the read+rewrite+rename window so a
        concurrent flush can't append records to the original file
        between the read and the rename — those appends would be lost
        when tmp.replace() unlinks the original. For an audit log
        targeting compliance/forensics, lost records compromise the
        log's value.
        """
        while not self._stopped:
            try:
                await asyncio.sleep(_RETENTION_SWEEP_INTERVAL_S)
            except asyncio.CancelledError:
                return
            # Drain the buffer FIRST (outside the sweep lock window) so
            # any in-flight records make it to the file before the sweep
            # snapshots it.
            await self._flush_now()
            # Now hold the lock for the entire sweep — appenders block
            # briefly, but no writes are lost.
            async with self._lock:
                await asyncio.to_thread(self._sweep_retention)

    def _sweep_retention(self) -> None:
        """One pass of the retention sweep. Best-effort.

        Caller must hold `self._lock` to prevent concurrent appends to
        the original file (which would be lost when tmp.replace
        unlinks it). The lock is acquired by `_retention_loop`; direct
        unit-test callers run with no concurrent appender so the lack
        of lock there is safe.
        """
        if self._config.retention_days <= 0 or not self._path.exists():
            return
        cutoff = datetime.now(UTC) - timedelta(days=self._config.retention_days)
        try:
            kept: list[str] = []
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        ts = datetime.fromisoformat(record["ts"])
                        if ts >= cutoff:
                            kept.append(line)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Corrupt line — drop it (defensive).
                        continue
            # Atomic-ish replace via tmp file + rename.
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.writelines(kept)
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning(
                "audit.retention_sweep_failed",
                path=str(self._path),
                error=str(exc) or repr(exc),
            )


# Module-level NULL_AUDIT helper so callers can write
#   audit = audit_logger or NULL_AUDIT
# rather than `if audit_logger: ...` everywhere.
class _NullAuditLogger:
    """No-op AuditLogger used when audit is disabled or unset."""

    enabled: bool = False
    path: Path = Path()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None


# Singleton — safe to reuse since it has no state.
NULL_AUDIT = _NullAuditLogger()
