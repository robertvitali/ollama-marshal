"""Live TUI dashboard for ollama-marshal — single-window observability.

Btop-style. Polls /api/marshal/status at the configured rate, follows
~/.ollama-marshal/marshal.out.log continuously, renders header + memory
bars + loaded models + metrics + scrolling event tail, all in one
Rich-driven layout.

Designed to be the canonical "is marshal doing the right thing right now?"
view. Pair with `scripts/dryrun.py` (in scripts/) to fire scenarios while
watching the dashboard react.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG_PATH = Path.home() / ".ollama-marshal" / "marshal.out.log"
DEFAULT_MARSHAL_URL = "http://localhost:11435"
DEFAULT_REFRESH_HZ = 2.0  # render rate, clamped to [0.5, 10.0]
# Cap how often we hit /api/marshal/status regardless of render rate.
# A user passing --refresh-hz 100 should not DOS marshal's status endpoint.
_MAX_POLL_HZ = 5.0
_MIN_REFRESH_HZ = 0.5
_MAX_REFRESH_HZ = 10.0
# Status-poll request timeout. Distinct from the inference Hop 2
# budget (``scheduler.ollama_forward_timeout_s``) — this is a UI-side
# budget for "is marshal alive right now?" snapshot fetches. 2s is
# generous for a localhost JSON GET; beyond that we'd rather show a
# stale snapshot than block the render loop.
_STATUS_POLL_TIMEOUT_S = 2.0

# Match the lifecycle and scheduling events worth showing.
# Superset of the legacy `tail | grep "scheduler\.|request_(enqueued|served|
# timeout|error)"` recipe — adds `lifecycle.*` (model load/unload) and
# `model_registry.*` (size discovery) so the events panel is the full
# observability story, not a strict subset.
_EVENT_FILTER = re.compile(
    r"scheduler\."
    r"|server\.request_(enqueued|served|timeout|error)"
    r"|lifecycle\."
    r"|model_registry\."
)
# Strip ANSI color sequences from structlog console output.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Parse "TIMESTAMP [level] event.name key=value key=value" lines.
_LOG_LINE = re.compile(
    r"^(?P<ts>\S+)\s+\[(?P<level>\w+)\s*\]\s+(?P<event>\S+)\s*(?P<rest>.*)$"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StatusSnapshot:
    """Reduced view of /api/marshal/status used by the dashboard panels."""

    uptime_s: float = 0.0
    # Marshal budget (model-VRAM only, kept for compat).
    total_bytes: int = 0
    available_bytes: int = 0
    used_bytes: int = 0
    # Actual host RAM (NEW in v0.2.0 status payload).
    sys_total: int = 0
    sys_used: int = 0
    sys_available: int = 0
    sys_percent: float = 0.0
    # Host swap.
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0
    loaded_models: list[dict[str, Any]] = field(default_factory=list)
    pending_total: int = 0
    pending_by_model: dict[str, int] = field(default_factory=dict)
    requests_served: int = 0
    model_swaps: int = 0
    evictions: int = 0
    avg_wait_ms: float = 0.0
    fetched_at: float = 0.0
    error: str | None = None


@dataclass
class LogEvent:
    """A parsed scheduler/request lifecycle event from marshal.out.log."""

    timestamp: str  # HH:MM:SS.fff for display
    level: str
    event_name: str
    rest: str  # remaining key=value pairs


# ---------------------------------------------------------------------------
# Data fetching and parsing
# ---------------------------------------------------------------------------


def fetch_status(url: str) -> StatusSnapshot:
    """Fetch /api/marshal/status and reduce to StatusSnapshot.

    Returns a snapshot with .error set if the request failed (so callers
    don't need a try/except — the renderer handles error display).
    """
    try:
        r = httpx.get(f"{url}/api/marshal/status", timeout=_STATUS_POLL_TIMEOUT_S)
        r.raise_for_status()
        d = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return StatusSnapshot(error=str(exc), fetched_at=time.time())

    if not isinstance(d, dict):
        return StatusSnapshot(
            error=f"unexpected response type: {type(d).__name__}",
            fetched_at=time.time(),
        )

    return _build_snapshot(d)


async def fetch_status_async(client: httpx.AsyncClient, url: str) -> StatusSnapshot:
    """Async variant of fetch_status.

    Used by the async poller so the event loop is not blocked while
    waiting on marshal.
    """
    try:
        r = await client.get(
            f"{url}/api/marshal/status", timeout=_STATUS_POLL_TIMEOUT_S
        )
        r.raise_for_status()
        d = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return StatusSnapshot(error=str(exc), fetched_at=time.time())

    if not isinstance(d, dict):
        return StatusSnapshot(
            error=f"unexpected response type: {type(d).__name__}",
            fetched_at=time.time(),
        )

    return _build_snapshot(d)


def _build_snapshot(d: dict[str, Any]) -> StatusSnapshot:
    """Reduce raw /api/marshal/status JSON into a StatusSnapshot."""
    mem = d.get("memory", {})
    sys_mem = mem.get("system", {})
    swap = mem.get("swap", {})
    return StatusSnapshot(
        uptime_s=d.get("uptime_seconds", 0.0),
        total_bytes=mem.get("total", 0),
        available_bytes=mem.get("available", 0),
        used_bytes=mem.get("used_by_models", 0),
        sys_total=sys_mem.get("total", 0),
        sys_used=sys_mem.get("used", 0),
        sys_available=sys_mem.get("available", 0),
        sys_percent=sys_mem.get("percent", 0.0),
        swap_total=swap.get("total", 0),
        swap_used=swap.get("used", 0),
        swap_percent=swap.get("percent", 0.0),
        loaded_models=d.get("loaded_models", []),
        pending_total=d.get("queue", {}).get("total_pending", 0),
        pending_by_model=d.get("queue", {}).get("by_model", {}),
        requests_served=d.get("metrics", {}).get("requests_served", 0),
        model_swaps=d.get("metrics", {}).get("model_swaps", 0),
        evictions=d.get("metrics", {}).get("evictions", 0),
        avg_wait_ms=d.get("metrics", {}).get("average_wait_ms", 0.0),
        fetched_at=time.time(),
    )


def parse_log_line(line: str) -> LogEvent | None:
    """Parse a structlog console-format line.

    Returns None if the line doesn't match the structlog console format or
    doesn't pass the event filter (only scheduler.* / request_* /
    model_registry.* events are kept).
    """
    clean = _ANSI.sub("", line.rstrip())
    m = _LOG_LINE.match(clean)
    if not m:
        return None
    event = m.group("event")
    if not _EVENT_FILTER.search(event):
        return None

    ts_raw = m.group("ts")
    try:
        dt = datetime.fromisoformat(ts_raw.rstrip("Z"))
        ts_short = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
    except ValueError:
        ts_short = ts_raw[-12:]

    return LogEvent(
        timestamp=ts_short,
        level=m.group("level").strip(),
        event_name=event,
        rest=m.group("rest").strip(),
    )


# ---------------------------------------------------------------------------
# Renderers (pure functions — easy to test)
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    # Coerce negatives (shouldn't happen, but if marshal returns nonsense
    # we want clean display rather than "-1h 59m 55s" from divmod).
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s"


def format_wait_ms(ms: float) -> str:
    """Format an average wait duration for human-readable display.

    Wait times can range from a few hundred ms (cached / passthrough)
    up to multiple minutes (cold-load of a 70B model). Render adaptively:

    - < 1s   → "123ms"   (preserves resolution for fast paths)
    - < 1min → "5.2s"    (one decimal, easy to compare)
    - else   → "1m 23s"  (minute format requested for legibility)
    """
    if ms < 0:
        return "0ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, sec = divmod(int(seconds), 60)
    return f"{m}m {sec}s"


def render_header(status: StatusSnapshot, marshal_url: str) -> Panel:
    """Top bar: marshal URL + uptime + last refresh time."""
    if status.error:
        body: Text = Text(
            f"  ✗  Cannot reach {marshal_url}  —  {status.error}",
            style="red bold",
        )
    else:
        body = Text.assemble(
            ("  ollama-marshal", "bold cyan"),
            ("  →  ", "dim"),
            (marshal_url, ""),
            ("   uptime ", "dim"),
            (_format_uptime(status.uptime_s), "bold"),
            ("   refreshed ", "dim"),
            (datetime.fromtimestamp(status.fetched_at).strftime("%H:%M:%S"), "dim"),
        )
    return Panel(body, border_style="cyan", padding=(0, 1))


def _bar_color(pct: float) -> str:
    """Color thresholds for memory bars: green <70%, yellow <90%, red otherwise."""
    if pct < 70:
        return "green"
    if pct < 90:
        return "yellow"
    return "red"


def render_memory(status: StatusSnapshot) -> Panel:
    """Memory panel with system RAM, marshal budget, swap, and per-model bars."""
    if status.total_bytes == 0 and status.sys_total == 0:
        return Panel(
            Text("(no data)", style="dim"),
            title="Memory",
            border_style="cyan",
        )

    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(ratio=3)
    grid.add_column(ratio=1, justify="right")

    # Marshal budget bar (model-VRAM-only, kept for cluster-vs-budget visibility)
    if status.total_bytes > 0:
        budget_total_gb = status.total_bytes / (1024**3)
        budget_used_gb = status.used_bytes / (1024**3)
        budget_pct = (status.used_bytes / status.total_bytes) * 100
        budget_bar = ProgressBar(
            total=100,
            completed=budget_pct,
            complete_style=_bar_color(budget_pct),
        )
        grid.add_row(
            budget_bar,
            Text.assemble(
                ("budget ", "dim"),
                (f"{budget_used_gb:.1f}", "bold"),
                (" / ", "dim"),
                (f"{budget_total_gb:.1f} GB", "bold"),
            ),
        )

    # System RAM bar (actual host)
    if status.sys_total > 0:
        sys_total_gb = status.sys_total / (1024**3)
        sys_used_gb = status.sys_used / (1024**3)
        sys_pct = status.sys_percent
        sys_bar = ProgressBar(
            total=100, completed=sys_pct, complete_style=_bar_color(sys_pct)
        )
        grid.add_row(
            sys_bar,
            Text.assemble(
                ("system ", "dim"),
                (f"{sys_used_gb:.1f}", "bold"),
                (" / ", "dim"),
                (f"{sys_total_gb:.1f} GB", "bold"),
                (f"  ({sys_pct:.0f}%)", "dim"),
            ),
        )

    # Swap (only if any is in use)
    if status.swap_total > 0 and status.swap_used > 0:
        swap_total_gb = status.swap_total / (1024**3)
        swap_used_gb = status.swap_used / (1024**3)
        swap_pct = status.swap_percent
        swap_bar = ProgressBar(total=100, completed=swap_pct, complete_style="red")
        grid.add_row(
            swap_bar,
            Text.assemble(
                ("swap ", "dim"),
                (f"{swap_used_gb:.1f}", "bold red"),
                (" / ", "dim"),
                (f"{swap_total_gb:.1f} GB", "bold"),
            ),
        )

    # Per-loaded-model bars (capped at 5 to keep panel compact)
    for m in status.loaded_models[:5]:
        sz_gb = m["size_vram"] / (1024**3)
        denom = status.sys_total or status.total_bytes or 1
        mpct = (m["size_vram"] / denom) * 100
        name_bar = Table.grid(padding=(0, 1))
        name_bar.add_column(no_wrap=True, ratio=2)
        name_bar.add_column(ratio=3)
        name_bar.add_row(
            Text("  " + m["name"][:28], style="cyan"),
            ProgressBar(total=100, completed=mpct, complete_style="cyan"),
        )
        grid.add_row(name_bar, Text(f"{sz_gb:.1f} GB", style="dim"))

    if status.sys_total:
        title_used = status.sys_used / (1024**3)
        title_total = status.sys_total / (1024**3)
    else:
        title_used = status.used_bytes / (1024**3)
        title_total = status.total_bytes / (1024**3)
    return Panel(
        grid,
        title=f"Memory  [{title_used:.1f} / {title_total:.1f} GB]",
        border_style="cyan",
    )


def render_models(status: StatusSnapshot) -> Panel:
    """Loaded models table."""
    t = Table(expand=True, show_header=True, header_style="bold cyan")
    t.add_column("Model", style="cyan", no_wrap=True)
    t.add_column("VRAM", justify="right")
    t.add_column("Pending", justify="right")
    t.add_column("Programs", overflow="fold")

    if not status.loaded_models:
        t.add_row("[dim](no models loaded)[/dim]", "—", "—", "—")
    else:
        for m in status.loaded_models:
            sz = f"{m['size_vram'] / (1024**3):.1f} GB"
            pending = m.get("pending_requests", 0)
            pending_text = (
                f"[yellow]{pending}[/yellow]" if pending > 0 else f"{pending}"
            )
            progs = m.get("programs") or []
            progs_text = ", ".join(progs) if progs else "[dim]—[/dim]"
            t.add_row(m["name"], sz, pending_text, progs_text)

    title = (
        f"Loaded Models ({len(status.loaded_models)})"
        f"  ·  Queue {status.pending_total} pending"
    )
    return Panel(t, title=title, border_style="cyan")


def _delta_str(current: int | float, base: int | float) -> str:
    d = current - base
    if d == 0:
        return "[dim](Δ0)[/dim]"
    sign = "+" if d > 0 else ""
    color = "green" if d > 0 else "yellow"
    val = d if isinstance(d, int) else round(d, 1)
    return f"[{color}](Δ{sign}{val})[/{color}]"


def render_metrics(status: StatusSnapshot, baseline: StatusSnapshot | None) -> Panel:
    """Metrics panel with delta-since-dashboard-started."""
    base = baseline or StatusSnapshot()
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        f"Requests served: [bold]{status.requests_served}[/bold] "
        f"{_delta_str(status.requests_served, base.requests_served)}",
        f"Model swaps: [bold]{status.model_swaps}[/bold] "
        f"{_delta_str(status.model_swaps, base.model_swaps)}",
    )
    grid.add_row(
        f"Evictions: [bold]{status.evictions}[/bold] "
        f"{_delta_str(status.evictions, base.evictions)}",
        f"Avg wait: [bold]{format_wait_ms(status.avg_wait_ms)}[/bold]",
    )
    return Panel(grid, title="Metrics", border_style="cyan")


def _event_color(event_name: str) -> str:
    """Pick a color for the event name based on its kind."""
    if event_name.startswith("scheduler.evict"):
        return "red"
    if event_name.startswith("scheduler.bin_pack"):
        return "green"
    if event_name.startswith("scheduler.critical"):
        return "magenta"
    if event_name.startswith("scheduler.forced"):
        return "yellow"
    if event_name.startswith(("server.request_error", "server.request_timeout")):
        return "red"
    if event_name == "lifecycle.preloaded":
        return "green"
    if event_name == "lifecycle.unloaded":
        return "yellow"
    if event_name.endswith(("_failed", "_timeout", "_error")):
        return "red"
    return "cyan"


def render_events(events: deque[LogEvent], rows: int) -> Panel:
    """Scrolling event log (tail-style — newest at the bottom)."""
    rows = max(1, rows)
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column("ts", style="dim", no_wrap=True)
    t.add_column("event", overflow="fold")

    visible = list(events)[-rows:]
    if not visible:
        t.add_row("", Text("(no events yet — fire some traffic)", style="dim"))
    else:
        for e in visible:
            t.add_row(
                e.timestamp,
                Text.assemble(
                    (e.event_name, _event_color(e.event_name)),
                    ("  ", ""),
                    (e.rest, "dim"),
                ),
            )

    return Panel(
        t,
        title=f"Events ({len(events)} buffered, filter: scheduler.* + request_*)",
        border_style="cyan",
        padding=(0, 0),
    )


def render_footer() -> Panel:
    """Bottom keybindings hint."""
    return Panel(
        Text("  Ctrl+C to quit", style="dim"),
        border_style="cyan",
        padding=(0, 1),
    )


def make_layout(
    status: StatusSnapshot,
    baseline: StatusSnapshot | None,
    events: deque[LogEvent],
    marshal_url: str,
    console_height: int,
) -> Layout:
    """Compose the full dashboard layout."""
    # Reserve heights for fixed-size panels; events panel takes whatever's left.
    header_h = 3
    memory_h = 10
    models_h = 10
    metrics_h = 5
    footer_h = 3
    fixed = header_h + memory_h + models_h + metrics_h + footer_h
    events_height = max(5, console_height - fixed)
    # Approximate row count for events (panel borders + padding cost ~3 rows)
    event_rows = max(1, events_height - 3)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=header_h),
        Layout(name="body"),
        Layout(name="footer", size=footer_h),
    )
    layout["body"].split_column(
        Layout(name="memory", size=memory_h),
        Layout(name="models", size=models_h),
        Layout(name="metrics", size=metrics_h),
        Layout(name="events"),
    )
    layout["header"].update(render_header(status, marshal_url))
    layout["memory"].update(render_memory(status))
    layout["models"].update(render_models(status))
    layout["metrics"].update(render_metrics(status, baseline))
    layout["events"].update(render_events(events, event_rows))
    layout["footer"].update(render_footer())
    return layout


# ---------------------------------------------------------------------------
# Async coordinators
# ---------------------------------------------------------------------------


async def status_poller(
    state: dict[str, Any],
    marshal_url: str,
    interval: float,
) -> None:
    """Poll /api/marshal/status at `interval` seconds. Updates state in place.

    Uses an async httpx client so the event loop stays responsive — render_loop
    and log_follower can keep running while a status fetch is in flight.
    """
    async with httpx.AsyncClient() as client:
        while not state.get("stopped"):
            snap = await fetch_status_async(client, marshal_url)
            if state.get("baseline") is None and snap.error is None:
                state["baseline"] = snap
            state["status"] = snap
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return


async def log_follower(state: dict[str, Any], log_path: Path) -> None:
    """Tail the marshal log file like `tail -f`. Updates state["events"].

    Best-effort: if the file is unreadable (permissions), missing, or the
    open/read fails for any other OS reason, surface the error in
    state["log_error"] and exit cleanly instead of crashing the dashboard.
    """
    # Wait for the file to exist (marshal might not be running yet).
    while not log_path.exists() and not state.get("stopped"):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            return

    if state.get("stopped"):
        return

    # File I/O on this path is normally fast (local disk, line-buffered
    # marshal log), but on a slow filesystem (NFS, throttled SSD) seek/
    # readline can block the event loop. Wrap them in asyncio.to_thread
    # so the render loop and status poller stay responsive. Avoids adding
    # aiofiles as a new dependency.
    try:
        f = await asyncio.to_thread(log_path.open, "r")
    except OSError as exc:
        state["log_error"] = f"cannot read {log_path}: {exc}"
        return

    try:
        await asyncio.to_thread(f.seek, 0, 2)  # start at end
        while not state.get("stopped"):
            line = await asyncio.to_thread(f.readline)
            if line:
                event = parse_log_line(line)
                if event is not None:
                    state["events"].append(event)
            else:
                try:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    return
    finally:
        await asyncio.to_thread(f.close)


async def render_loop(
    state: dict[str, Any], marshal_url: str, refresh_hz: float
) -> None:
    """Drive the rich.Live display."""
    interval = 1.0 / refresh_hz
    console = Console()

    initial = make_layout(
        StatusSnapshot(),
        None,
        state["events"],
        marshal_url,
        console.height,
    )
    with Live(
        initial,
        console=console,
        refresh_per_second=refresh_hz,
        screen=True,  # use alternate screen so we don't clobber scrollback
        transient=False,
    ) as live:
        while not state.get("stopped"):
            live.update(
                make_layout(
                    state.get("status", StatusSnapshot()),
                    state.get("baseline"),
                    state["events"],
                    marshal_url,
                    console.height,
                )
            )
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return


async def run_dashboard_async(
    marshal_url: str,
    log_path: Path,
    refresh_hz: float,
) -> None:
    """Coordinate the three async tasks: status polling, log following, rendering."""
    refresh_hz = max(_MIN_REFRESH_HZ, min(_MAX_REFRESH_HZ, refresh_hz))
    poll_hz = min(refresh_hz, _MAX_POLL_HZ)
    poll_interval = 1.0 / poll_hz

    state: dict[str, Any] = {
        "status": StatusSnapshot(),
        "baseline": None,
        "events": deque(maxlen=500),
        "stopped": False,
    }
    poller_task = asyncio.create_task(status_poller(state, marshal_url, poll_interval))
    follower_task = asyncio.create_task(log_follower(state, log_path))
    try:
        await render_loop(state, marshal_url, refresh_hz)
    finally:
        state["stopped"] = True
        poller_task.cancel()
        follower_task.cancel()
        await asyncio.gather(poller_task, follower_task, return_exceptions=True)


def run_dashboard(
    marshal_url: str = DEFAULT_MARSHAL_URL,
    log_path: Path = DEFAULT_LOG_PATH,
    refresh_hz: float = DEFAULT_REFRESH_HZ,
) -> None:
    """Public entry point. Runs the dashboard until Ctrl+C."""
    try:
        asyncio.run(run_dashboard_async(marshal_url, log_path, refresh_hz))
    except KeyboardInterrupt:
        pass
