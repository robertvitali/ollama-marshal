"""Tests for the audit log feature flag and JSONL writer."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ollama_marshal.audit import NULL_AUDIT, AuditLogger
from ollama_marshal.config import AuditConfig


@pytest.fixture
def disabled_config() -> AuditConfig:
    return AuditConfig(enabled=False)


@pytest.fixture
def enabled_config(tmp_path: Path) -> AuditConfig:
    return AuditConfig(
        enabled=True,
        path=str(tmp_path / "audit.jsonl"),
        retention_days=0,  # disable retention sweep in unit tests
        max_size_mb=0,  # disable rotation in unit tests
    )


# ---------------------------------------------------------------------------
# Disabled-mode behavior — should be cheap no-ops
# ---------------------------------------------------------------------------


class TestDisabledMode:
    async def test_record_is_noop_when_disabled(self, disabled_config, tmp_path):
        audit = AuditLogger(disabled_config)
        await audit.start()
        await audit.record("request.served", model="m", program_id="p")
        await audit.stop()
        # No file should have been created.
        assert not (tmp_path / "audit.jsonl").exists()
        assert audit.enabled is False

    async def test_start_stop_safe_when_disabled(self, disabled_config):
        audit = AuditLogger(disabled_config)
        # Multiple start/stop cycles should be safe.
        await audit.start()
        await audit.stop()
        await audit.start()
        await audit.stop()


# ---------------------------------------------------------------------------
# Enabled mode — write, buffer, flush
# ---------------------------------------------------------------------------


class TestEnabledMode:
    async def test_record_writes_jsonl(self, enabled_config):
        audit = AuditLogger(enabled_config)
        await audit.start()
        try:
            await audit.record(
                "request.served",
                program_id="ai-portfolio",
                model="qwen3.5:9b",
                endpoint="/api/chat",
                wait_ms=1234.5,
                stream=False,
            )
            # Force flush by stopping (which final-flushes).
        finally:
            await audit.stop()

        path = Path(enabled_config.path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "request.served"
        assert record["program_id"] == "ai-portfolio"
        assert record["model"] == "qwen3.5:9b"
        assert record["wait_ms"] == 1234.5
        assert record["stream"] is False
        assert "ts" in record  # ISO timestamp present

    async def test_optional_fields_omitted_when_none(self, enabled_config):
        audit = AuditLogger(enabled_config)
        await audit.start()
        try:
            await audit.record("request.served", program_id="p", model="m")
        finally:
            await audit.stop()

        record = json.loads(
            Path(enabled_config.path).read_text().strip().split("\n")[0]
        )
        # Compact records — fields not provided should be absent.
        assert "endpoint" not in record
        assert "wait_ms" not in record
        assert "stream" not in record

    async def test_burst_size_hint_recorded(self, enabled_config):
        audit = AuditLogger(enabled_config)
        await audit.start()
        try:
            await audit.record(
                "request.enqueued",
                program_id="ai-portfolio",
                model="qwen3.5:9b",
                burst_size_hint=50,
            )
        finally:
            await audit.stop()
        record = json.loads(
            Path(enabled_config.path).read_text().strip().split("\n")[0]
        )
        assert record["burst_size_hint"] == 50

    async def test_multiple_records_each_on_own_line(self, enabled_config):
        audit = AuditLogger(enabled_config)
        await audit.start()
        try:
            for i in range(10):
                await audit.record("request.served", model=f"m-{i}", program_id="p")
        finally:
            await audit.stop()
        lines = Path(enabled_config.path).read_text().strip().split("\n")
        assert len(lines) == 10
        # Each line must be valid JSON.
        for line in lines:
            assert json.loads(line)["event"] == "request.served"

    async def test_creates_parent_dirs(self, tmp_path):
        config = AuditConfig(
            enabled=True,
            path=str(tmp_path / "deep" / "nested" / "audit.jsonl"),
            retention_days=0,
            max_size_mb=0,
        )
        audit = AuditLogger(config)
        await audit.start()
        try:
            await audit.record("request.served", model="m", program_id="p")
        finally:
            await audit.stop()
        assert Path(config.path).exists()


# ---------------------------------------------------------------------------
# Privacy: prompt content must NEVER appear in audit records
# ---------------------------------------------------------------------------


class TestPrivacy:
    async def test_record_signature_does_not_accept_prompt(self, enabled_config):
        # The record() method should not have any parameter that could
        # accidentally smuggle prompt text. This is enforced by signature
        # — if someone tries audit.record(..., prompt="..."), Python will
        # reject the kwarg.
        audit = AuditLogger(enabled_config)
        await audit.start()
        try:
            with pytest.raises(TypeError):
                # `prompt` is not a valid kwarg.
                await audit.record(  # type: ignore[call-arg]
                    "request.served", prompt="secret data"
                )
        finally:
            await audit.stop()


# ---------------------------------------------------------------------------
# Retention sweep
# ---------------------------------------------------------------------------


class TestRetention:
    def test_sweep_drops_old_entries(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        # Write a mix of old + fresh entries.
        old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        fresh_ts = datetime.now(UTC).isoformat()
        with path.open("w") as f:
            f.write(json.dumps({"ts": old_ts, "event": "old"}) + "\n")
            f.write(json.dumps({"ts": fresh_ts, "event": "fresh"}) + "\n")
            f.write(json.dumps({"ts": fresh_ts, "event": "also fresh"}) + "\n")

        config = AuditConfig(
            enabled=True,
            path=str(path),
            retention_days=30,
            max_size_mb=0,
        )
        audit = AuditLogger(config)
        # Run sweep synchronously (helper is sync).
        audit._sweep_retention()

        lines = path.read_text().strip().split("\n")
        events = [json.loads(line)["event"] for line in lines]
        assert "old" not in events
        assert events.count("fresh") == 1
        assert events.count("also fresh") == 1

    def test_sweep_disabled_when_retention_zero(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        old_ts = (datetime.now(UTC) - timedelta(days=999)).isoformat()
        path.write_text(json.dumps({"ts": old_ts, "event": "ancient"}) + "\n")

        config = AuditConfig(
            enabled=True, path=str(path), retention_days=0, max_size_mb=0
        )
        audit = AuditLogger(config)
        audit._sweep_retention()
        # File untouched.
        assert "ancient" in path.read_text()

    def test_sweep_handles_corrupt_lines(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        fresh_ts = datetime.now(UTC).isoformat()
        with path.open("w") as f:
            f.write("not json at all\n")
            f.write(json.dumps({"ts": fresh_ts, "event": "fresh"}) + "\n")
            f.write(json.dumps({"event": "missing ts"}) + "\n")

        config = AuditConfig(
            enabled=True, path=str(path), retention_days=30, max_size_mb=0
        )
        audit = AuditLogger(config)
        audit._sweep_retention()
        # Only the well-formed fresh entry survives.
        survivors = [line for line in path.read_text().strip().split("\n") if line]
        assert len(survivors) == 1
        assert json.loads(survivors[0])["event"] == "fresh"


# ---------------------------------------------------------------------------
# Size-based rotation
# ---------------------------------------------------------------------------


class TestRotation:
    async def test_rotates_when_over_size_limit(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        # 1 MB cap; rotation happens AFTER a write that takes the file
        # over the threshold. So the live file is renamed to .1 and any
        # subsequent record() lands in a fresh file.
        config = AuditConfig(
            enabled=True,
            path=str(path),
            retention_days=0,
            max_size_mb=1,
        )
        # Pre-populate the file just over 1 MB so the next write triggers.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write("x" * (1024 * 1024 + 100))

        audit = AuditLogger(config)
        await audit.start()
        try:
            await audit.record("request.served", model="m", program_id="p")
        finally:
            await audit.stop()
        # The pre-populated oversized content has been moved to .1.
        # (Records from this test were flushed BEFORE the size check, so
        # they're inside the .1 file; the live file is fresh-empty until
        # the next flush, which the test doesn't trigger.)
        assert (tmp_path / "audit.jsonl.1").exists()
        # .1 contains the original 1 MB padding plus our test record.
        rotated = (tmp_path / "audit.jsonl.1").read_text()
        assert "request.served" in rotated


# ---------------------------------------------------------------------------
# NULL_AUDIT singleton
# ---------------------------------------------------------------------------


class TestNullAudit:
    async def test_null_audit_is_disabled(self):
        assert NULL_AUDIT.enabled is False

    async def test_null_audit_record_safe(self):
        # Accepts any kwargs without raising.
        await NULL_AUDIT.record("anything", program_id="p", model="m", whatever="x")

    async def test_null_audit_start_stop(self):
        await NULL_AUDIT.start()
        await NULL_AUDIT.stop()


# ---------------------------------------------------------------------------
# Background flush loop + properties + error paths
# ---------------------------------------------------------------------------


class TestBackgroundFlush:
    async def test_path_property_when_enabled(self, enabled_config):
        audit = AuditLogger(enabled_config)
        assert audit.path == Path(enabled_config.path)

    async def test_path_property_when_disabled(self, disabled_config):
        audit = AuditLogger(disabled_config)
        # Disabled config returns a default empty path (not the configured one).
        assert audit.path == Path()

    async def test_buffer_size_triggers_flush(self, enabled_config):
        # Writing 60 records — the in-record FLUSH_BATCH_SIZE=50 path
        # fires once at record 50; remaining 10 land at stop().
        audit = AuditLogger(enabled_config)
        await audit.start()
        try:
            for i in range(60):
                await audit.record("request.served", model=f"m-{i}", program_id="p")
        finally:
            await audit.stop()
        lines = Path(enabled_config.path).read_text().strip().split("\n")
        assert len(lines) == 60

    async def test_record_after_stop_is_noop(self, enabled_config):
        audit = AuditLogger(enabled_config)
        await audit.start()
        await audit.stop()
        # After stop, records should be ignored without raising.
        await audit.record("request.served", model="m", program_id="p")
        # File has nothing in it (no records were written before stop).
        path = Path(enabled_config.path)
        if path.exists():
            assert path.read_text().strip() == ""


class TestRotationMultiGen:
    """Multiple rotation cycles: .1 → .2 → ... up to the cap."""

    def test_rotates_through_generations(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        # Pre-populate .1 .. .5 to simulate prior rotations.
        for i in range(1, 6):
            (tmp_path / f"audit.jsonl.{i}").write_text(f"gen{i}\n")
        path.write_text("x" * (1024 * 1024 + 10))

        config = AuditConfig(
            enabled=True, path=str(path), retention_days=0, max_size_mb=1
        )
        audit = AuditLogger(config)
        audit._maybe_rotate()

        # Current rotated to .1; previous .5 dropped (over cap).
        assert (tmp_path / "audit.jsonl.1").exists()
        assert (tmp_path / "audit.jsonl.5").exists()
        assert not (tmp_path / "audit.jsonl.6").exists()


class TestSweepMissingFile:
    def test_sweep_safe_when_file_missing(self, tmp_path):
        """No file → sweep is a clean no-op (first-run scenario)."""
        config = AuditConfig(
            enabled=True,
            path=str(tmp_path / "never-existed.jsonl"),
            retention_days=30,
            max_size_mb=0,
        )
        audit = AuditLogger(config)
        # Should not raise.
        audit._sweep_retention()


class TestRetentionTaskLifecycle:
    """Cover the retention background-task spawn + cancel path."""

    async def test_retention_task_starts_when_retention_days_positive(self, tmp_path):
        config = AuditConfig(
            enabled=True,
            path=str(tmp_path / "audit.jsonl"),
            retention_days=30,
            max_size_mb=0,
        )
        audit = AuditLogger(config)
        await audit.start()
        try:
            # Internal: retention task should be spawned.
            assert audit._retention_task is not None
        finally:
            await audit.stop()


class TestErrorPaths:
    def test_write_lines_logs_on_oserror(self, tmp_path, monkeypatch):
        # Force open() to raise — the writer logs and returns rather than crash.
        config = AuditConfig(
            enabled=True,
            path=str(tmp_path / "audit.jsonl"),
            retention_days=0,
            max_size_mb=0,
        )
        audit = AuditLogger(config)

        def raising_open(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "open", raising_open)
        # Must not raise.
        audit._write_lines([{"event": "x"}])

    def test_maybe_rotate_handles_missing_file(self, tmp_path):
        # No file → stat() raises OSError → rotate is a clean no-op.
        config = AuditConfig(
            enabled=True,
            path=str(tmp_path / "never.jsonl"),
            retention_days=0,
            max_size_mb=1,
        )
        audit = AuditLogger(config)
        audit._maybe_rotate()  # must not raise

    def test_sweep_handles_oserror(self, tmp_path, monkeypatch):
        # Force open() to raise during the sweep — handler logs and returns.
        path = tmp_path / "audit.jsonl"
        path.write_text(
            json.dumps({"ts": "2026-04-28T00:00:00+00:00", "event": "x"}) + "\n"
        )
        config = AuditConfig(
            enabled=True, path=str(path), retention_days=30, max_size_mb=0
        )
        audit = AuditLogger(config)

        def raising_open(self, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "open", raising_open)
        audit._sweep_retention()  # must not raise
