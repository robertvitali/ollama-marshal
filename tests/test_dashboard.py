"""Tests for the live TUI dashboard.

Covers the pure render functions, the log line parser, and the status
fetcher (with the Ollama-marshal HTTP boundary mocked at the module's
import location, per CLAUDE.md bright-line rule #1).
"""

from __future__ import annotations

from collections import deque
from unittest.mock import patch

import httpx
import pytest

from ollama_marshal.dashboard import (
    LogEvent,
    StatusSnapshot,
    _delta_str,
    _event_color,
    _format_uptime,
    fetch_status,
    make_layout,
    parse_log_line,
    render_events,
    render_footer,
    render_header,
    render_memory,
    render_metrics,
    render_models,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestFormatUptime:
    def test_seconds_only(self):
        assert _format_uptime(45) == "0m 45s"

    def test_minutes(self):
        assert _format_uptime(125) == "2m 5s"

    def test_with_hours(self):
        assert _format_uptime(3725) == "1h 2m 5s"

    def test_zero(self):
        assert _format_uptime(0) == "0m 0s"


class TestDeltaStr:
    def test_zero(self):
        assert "Δ0" in _delta_str(10, 10)

    def test_positive_int(self):
        assert "Δ+5" in _delta_str(15, 10)
        assert "green" in _delta_str(15, 10)

    def test_negative(self):
        assert "Δ-3" in _delta_str(7, 10)
        assert "yellow" in _delta_str(7, 10)

    def test_float_rounded(self):
        result = _delta_str(10.55, 10.0)
        assert "Δ+0.6" in result or "Δ+0.55" in result


class TestEventColor:
    def test_eviction_is_red(self):
        assert _event_color("scheduler.evicting") == "red"

    def test_bin_pack_is_green(self):
        assert _event_color("scheduler.bin_pack_load") == "green"

    def test_critical_is_magenta(self):
        assert _event_color("scheduler.critical_preemption") == "magenta"

    def test_forced_is_yellow(self):
        assert _event_color("scheduler.forced_load") == "yellow"

    def test_request_error_is_red(self):
        assert _event_color("server.request_error") == "red"

    def test_default_is_cyan(self):
        assert _event_color("server.request_enqueued") == "cyan"
        assert _event_color("scheduler.tick_error") == "cyan"


# ---------------------------------------------------------------------------
# parse_log_line
# ---------------------------------------------------------------------------


class TestParseLogLine:
    def test_parses_clean_console_line(self):
        line = (
            "2026-04-27T05:35:12.514Z [info  ] "
            "scheduler.bin_pack_load model=qwen3.5:4b size_gb=2.1"
        )
        ev = parse_log_line(line)
        assert ev is not None
        assert ev.event_name == "scheduler.bin_pack_load"
        assert ev.level == "info"
        assert "model=qwen3.5:4b" in ev.rest
        assert ev.timestamp.startswith("05:35:12")

    def test_strips_ansi_color_codes(self):
        # structlog console-mode wraps level + event_name in ANSI sequences
        line = (
            "\x1b[2m2026-04-27T05:35:12.514Z\x1b[0m "
            "[\x1b[32m\x1b[1minfo     \x1b[0m] "
            "\x1b[1mscheduler.evicting\x1b[0m "
            "\x1b[36mmodel\x1b[0m=\x1b[35mqwen3.5:4b\x1b[0m"
        )
        ev = parse_log_line(line)
        assert ev is not None
        assert ev.event_name == "scheduler.evicting"

    def test_filters_out_unrelated_events(self):
        # Only scheduler.* / request_* / model_registry.* should pass
        line = "2026-04-27T05:35:12.514Z [info  ] memory.budget_calculated total_gb=256"
        assert parse_log_line(line) is None

    def test_keeps_request_lifecycle(self):
        line = (
            "2026-04-27T05:35:12.514Z [info  ] "
            "server.request_enqueued program=ai-email-triage"
        )
        assert parse_log_line(line) is not None

    def test_keeps_model_registry(self):
        line = (
            "2026-04-27T05:35:12.514Z [info  ] "
            "model_registry.benchmark_starting count=5"
        )
        assert parse_log_line(line) is not None

    def test_unparseable_returns_none(self):
        assert parse_log_line("garbage") is None
        assert parse_log_line("") is None
        assert parse_log_line("   ") is None

    def test_invalid_iso_timestamp_falls_back(self):
        # Should not crash; should still produce an event using last 12 chars
        line = "not-a-timestamp [info  ] scheduler.bin_pack_load model=foo"
        ev = parse_log_line(line)
        assert ev is not None
        assert ev.event_name == "scheduler.bin_pack_load"


# ---------------------------------------------------------------------------
# fetch_status
# ---------------------------------------------------------------------------


class _MockResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom",
                request=None,
                response=None,  # type: ignore[arg-type]
            )


class TestFetchStatus:
    @patch("ollama_marshal.dashboard.httpx.get")
    def test_happy_path(self, mock_get):
        mock_get.return_value = _MockResponse(
            200,
            {
                "uptime_seconds": 1234,
                "memory": {
                    "total": 256 * 1024**3,
                    "available": 100 * 1024**3,
                    "used_by_models": 50 * 1024**3,
                },
                "loaded_models": [
                    {
                        "name": "qwen3.5:4b",
                        "size_vram": 22 * 1024**3,
                        "pending_requests": 0,
                    }
                ],
                "queue": {"total_pending": 0, "by_model": {}},
                "metrics": {
                    "requests_served": 30,
                    "model_swaps": 6,
                    "evictions": 0,
                    "average_wait_ms": 1500.5,
                },
            },
        )

        snap = fetch_status("http://localhost:11435")
        assert snap.error is None
        assert snap.uptime_s == 1234
        assert snap.requests_served == 30
        assert snap.model_swaps == 6
        assert snap.loaded_models[0]["name"] == "qwen3.5:4b"
        assert snap.fetched_at > 0

    @patch("ollama_marshal.dashboard.httpx.get")
    def test_returns_error_snapshot_on_http_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        snap = fetch_status("http://localhost:11435")
        assert snap.error is not None
        assert "refused" in snap.error
        assert snap.uptime_s == 0

    @patch("ollama_marshal.dashboard.httpx.get")
    def test_returns_error_on_non_200(self, mock_get):
        mock_get.return_value = _MockResponse(500, {})
        snap = fetch_status("http://localhost:11435")
        assert snap.error is not None

    @patch("ollama_marshal.dashboard.httpx.get")
    def test_handles_partial_response_gracefully(self, mock_get):
        # Marshal might respond with partial fields if some metrics aren't initialized
        mock_get.return_value = _MockResponse(200, {"uptime_seconds": 5})
        snap = fetch_status("http://localhost:11435")
        assert snap.error is None
        assert snap.uptime_s == 5
        assert snap.total_bytes == 0
        assert snap.requests_served == 0


# ---------------------------------------------------------------------------
# Renderers — smoke-test that they don't crash on edge inputs
# ---------------------------------------------------------------------------


def _full_status() -> StatusSnapshot:
    return StatusSnapshot(
        uptime_s=3600,
        total_bytes=256 * 1024**3,
        available_bytes=200 * 1024**3,
        used_bytes=50 * 1024**3,
        loaded_models=[
            {
                "name": "qwen3.5:4b-bf16",
                "size_vram": 22 * 1024**3,
                "pending_requests": 0,
            },
            {"name": "qwen3-vl:32b", "size_vram": 28 * 1024**3, "pending_requests": 2},
        ],
        pending_total=2,
        pending_by_model={"qwen3-vl:32b": 2},
        requests_served=30,
        model_swaps=6,
        evictions=1,
        avg_wait_ms=1500.5,
        fetched_at=1700000000.0,
    )


class TestRenderers:
    def test_render_header_happy(self):
        panel = render_header(_full_status(), "http://localhost:11435")
        assert panel is not None
        # rich.Panel renders to console; we're just checking no crash.

    def test_render_header_with_error(self):
        s = StatusSnapshot(error="Connection refused", fetched_at=1700000000.0)
        panel = render_header(s, "http://localhost:11435")
        assert panel is not None

    def test_render_memory_no_data(self):
        # total_bytes=0 → "no data" branch
        panel = render_memory(StatusSnapshot())
        assert panel is not None

    def test_render_memory_high_usage(self):
        s = _full_status()
        s.used_bytes = int(s.total_bytes * 0.95)  # >90% → red bar
        panel = render_memory(s)
        assert panel is not None

    def test_render_models_empty(self):
        panel = render_models(StatusSnapshot())
        assert panel is not None

    def test_render_models_full(self):
        panel = render_models(_full_status())
        assert panel is not None

    def test_render_metrics_no_baseline(self):
        panel = render_metrics(_full_status(), None)
        assert panel is not None

    def test_render_metrics_with_baseline(self):
        baseline = StatusSnapshot(
            requests_served=10, model_swaps=2, evictions=0, avg_wait_ms=500
        )
        panel = render_metrics(_full_status(), baseline)
        assert panel is not None

    def test_render_events_empty(self):
        panel = render_events(deque(), 10)
        assert panel is not None

    def test_render_events_with_entries(self):
        events = deque(
            [
                LogEvent(
                    timestamp="05:35:12.514",
                    level="info",
                    event_name="scheduler.bin_pack_load",
                    rest="model=qwen3.5:4b size_gb=2.1",
                ),
                LogEvent(
                    timestamp="05:35:12.515",
                    level="info",
                    event_name="scheduler.evicting",
                    rest="model=qwen3.5:0.8b pending=0",
                ),
            ]
        )
        panel = render_events(events, 5)
        assert panel is not None

    def test_render_events_clamps_rows(self):
        events = deque(
            [
                LogEvent(
                    timestamp=f"{i:02d}:00:00.000",
                    level="info",
                    event_name="x",
                    rest="",
                )
                for i in range(20)
            ]
        )
        panel = render_events(events, 5)
        assert panel is not None

    def test_render_events_zero_rows(self):
        # Should not crash even if rows=0 is passed
        panel = render_events(deque(), 0)
        assert panel is not None

    def test_render_footer(self):
        panel = render_footer()
        assert panel is not None


class TestMakeLayout:
    def test_full_layout(self):
        layout = make_layout(
            _full_status(),
            None,
            deque([LogEvent("05:35:12.514", "info", "scheduler.bin_pack_load", "")]),
            "http://localhost:11435",
            console_height=40,
        )
        assert layout is not None

    def test_layout_with_tiny_console(self):
        # When the console is shorter than the fixed-size panels, events
        # should still render with minimum 5 height.
        layout = make_layout(
            _full_status(),
            None,
            deque(),
            "http://localhost:11435",
            console_height=10,
        )
        assert layout is not None

    def test_layout_with_error_status(self):
        layout = make_layout(
            StatusSnapshot(error="boom"),
            None,
            deque(),
            "http://localhost:11435",
            console_height=40,
        )
        assert layout is not None


# ---------------------------------------------------------------------------
# Async coordinators — light coverage so the 95% coverage gate passes.
# Full e2e is exercised by the smoke test (manual: `ollama-marshal dashboard`).
# ---------------------------------------------------------------------------


class TestAsyncCoordinators:
    @pytest.mark.asyncio
    async def test_status_poller_updates_state(self):
        from ollama_marshal.dashboard import status_poller

        state: dict = {"status": StatusSnapshot(), "baseline": None, "stopped": False}

        with patch("ollama_marshal.dashboard.httpx.get") as mock_get:
            mock_get.return_value = _MockResponse(
                200,
                {
                    "uptime_seconds": 5,
                    "memory": {"total": 1, "available": 1, "used_by_models": 0},
                    "loaded_models": [],
                    "queue": {"total_pending": 0, "by_model": {}},
                    "metrics": {
                        "requests_served": 1,
                        "model_swaps": 0,
                        "evictions": 0,
                        "average_wait_ms": 0,
                    },
                },
            )
            import asyncio

            task = asyncio.create_task(status_poller(state, "http://x", 0.05))
            await asyncio.sleep(0.15)
            state["stopped"] = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert state["status"].uptime_s == 5
        assert state["baseline"] is not None
