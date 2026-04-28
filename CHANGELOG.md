# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-04-27

### Added

- **Programs per loaded model in `/api/marshal/status`** — each entry in
  `loaded_models[*]` now has a `programs: [...]` field listing every
  program ID with currently-pending requests for that model plus every
  program that has dispatched against it since it was loaded. Sorted,
  deduped, cleared on eviction. Surfaces as a "Programs" column in the
  TUI dashboard's loaded-models table and in `ollama-marshal status`.
- **Time-based idle eviction** — new `scheduler.idle_eviction_minutes`
  config field (default 15). Loaded models with no activity for that many
  minutes are evicted, regardless of memory pressure. Models with pending
  requests are never time-evicted. Set to `0` to disable. Behavior change
  vs v0.1.x — set to 0 in marshal.yaml to preserve old behavior.
- **Configurable request timeout** — new `proxy.request_timeout_s` config
  field (default 3600 = 1 hour, was hardcoded 300s in v0.1.x). Clients
  can override per-request via the `X-Request-Timeout: <seconds>` header
  so different programs can set their own SLAs.
- **System RAM and swap in `/api/marshal/status`** — `memory.system` and
  `memory.swap` blocks added (via psutil). The pre-existing top-level
  `total`/`available`/`used_by_models` keys are preserved for backward
  compatibility — they still report marshal's *budget* (model VRAM only).
  CLI `status` command and the new TUI dashboard render all three.
- `ollama-marshal dashboard` — live single-window TUI observability,
  btop-style. Polls `/api/marshal/status` and tails the structured log
  in one auto-refreshing layout: header (uptime), memory bar with
  per-model breakdown, loaded-models table, metrics with delta-
  since-dashboard-started, scrolling event log filtered to scheduler
  decisions and request lifecycle. Color-coded events (green for
  bin-pack, red for evict, magenta for critical preemption, yellow for
  forced-load). Built on `rich.live` + `rich.layout` — zero new
  dependencies.
- `scripts/dryrun.py` — load harness for validating marshal under
  realistic concurrent traffic. 12 named scenarios across single-shot,
  pattern (same-model-burst, same-model-loop, parallel-all, thrash-test,
  priority-test), and boundary (passthrough allowed/blocked, bad-model)
  groups. Each scenario polls `/api/marshal/status` before/after and
  asserts expected metric deltas (e.g. `same-model-burst` checks
  `model_swaps` Δ is 0).
- `scripts/README.md` — documents the 3-pane fallback layout (`watch
  ollama-marshal status` + `tail -f marshal.out.log` + dryrun) for cases
  where the integrated TUI dashboard isn't preferred. Lists expected
  dashboard behavior per scenario.
- `make dryrun-dashboard` — Makefile target that prints the three pane
  commands ready to paste into iTerm splits.
- `GET /status` — short alias for `/api/marshal/status` so a quick
  `curl localhost:11435/status` works without remembering the longer path.

### Fixed

- `scheduler.evicting` log entry used an f-string for the `reason` field
  (`reason=f"making room for {needed_for}"`), violating CLAUDE.md
  bright-line #9 ("no f-strings in log messages"). Replaced with a
  structured `needed_for=needed_for` key. Pre-existing on main.
- `_apply_env_overrides` int-coercion list was missing the two v0.2.0
  config fields (`request_timeout_s`, `idle_eviction_minutes`). Pydantic
  coerced them at validation time, so no runtime bug — but the dict
  shape was inconsistent with `port` / `poll_interval` / etc.
- `ollama-marshal start` crashed on launch with `AttributeError: module
  structlog has no attribute get_level_from_name`. The function never
  existed in any structlog version. Replaced with `getattr(logging,
  level.upper(), logging.INFO)` from the stdlib.
- `tests/test_cli.py::TestSetupLogging` mocked `structlog` so completely
  that `mock_structlog.get_level_from_name` resolved to a MagicMock and
  the bug never surfaced. Added unmocked parametrized tests that exercise
  the real codepath for each standard log level.
- `scheduler.request_failed` log entry was emitting `error=` (empty
  string) for httpx exceptions whose `str()` is empty. Now includes
  `error_type=<ExceptionClassName>` and falls back to `repr(exc)` when
  `str(exc)` is empty.
- `dashboard.fetch_status` could crash on a non-dict JSON response (e.g.
  if a misconfigured intermediary returned a list). Now returns a
  StatusSnapshot with a clear `.error` instead of raising.
- `scripts/dryrun.py:get_status` did direct `data["metrics"][...]` access
  that would `KeyError` on a partial response (e.g. during marshal
  startup). Switched to `.get(...)` defaults.

### Changed

- Average wait time renders adaptively in `ollama-marshal status` and
  the dashboard metrics panel. Sub-second waits show as `123ms`, 1-60s
  as `5.2s`, and longer waits as `1m 30s`. Previously always rendered
  as `50493 ms`. The `/api/marshal/status` JSON payload still returns
  raw `average_wait_ms` (unchanged) so consumers aren't broken.
- Dashboard's `status_poller` now uses `httpx.AsyncClient` instead of
  blocking `httpx.get` so the event loop stays responsive (previously
  the render loop and log follower could freeze for up to the 2s status
  timeout per tick).
- `_format_uptime(-N)` now clamps to `0m 0s` instead of producing
  ugly negative output (defensive — shouldn't happen, but doesn't hurt).
- Dashboard refresh rate is now clamped to [0.5, 10.0] Hz, and status
  polling is decoupled from refresh rate (capped at 5 Hz) so a user
  passing `--refresh-hz 100` cannot DOS marshal's status endpoint.
- Dashboard `log_follower` now catches `OSError` on file open/read and
  surfaces the error in `state["log_error"]` instead of silently dying.

## [0.1.0] - 2026-04-24

### Added

- Initial release
- FIFO + bin-packing model-aware scheduler with per-request fairness limits
- Memory auto-detection and VRAM budget management
- Model size registry with background benchmarking
- Full Ollama API proxy (native + OpenAI-compatible endpoints)
- Streaming response passthrough (NDJSON)
- Per-program priority configuration via `X-Program-ID` header
- YAML configuration with env var and CLI flag overrides
- Typer CLI: `start`, `status`, `stop` commands
- JSON status endpoint at `/api/marshal/status`
- Configurable shutdown behavior (drain vs immediate)
- Structured logging via structlog (console + JSON modes)
- 95%+ unit test coverage

[Unreleased]: https://github.com/robertvitali/ollama-marshal/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robertvitali/ollama-marshal/releases/tag/v0.1.0
