# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-04-27

### Added

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
  pattern (email-burst, portfolio-loop, parallel-all, thrash-test,
  priority-test), and boundary (passthrough allowed/blocked, bad-model)
  groups. Each scenario polls `/api/marshal/status` before/after and
  asserts expected metric deltas (e.g. `email-burst` checks `model_swaps`
  Δ is 0).
- `scripts/README.md` — documents the 3-pane fallback layout (`watch
  ollama-marshal status` + `tail -f marshal.out.log` + dryrun) for cases
  where the integrated TUI dashboard isn't preferred. Lists expected
  dashboard behavior per scenario.
- `make dryrun-dashboard` — Makefile target that prints the three pane
  commands ready to paste into iTerm splits.
- `GET /status` — short alias for `/api/marshal/status` so a quick
  `curl localhost:11435/status` works without remembering the longer path.

### Fixed

- `ollama-marshal start` crashed on launch with `AttributeError: module
  structlog has no attribute get_level_from_name`. The function never
  existed in any structlog version. Replaced with `getattr(logging,
  level.upper(), logging.INFO)` from the stdlib.
- `tests/test_cli.py::TestSetupLogging` mocked `structlog` so completely
  that `mock_structlog.get_level_from_name` resolved to a MagicMock and
  the bug never surfaced. Added unmocked parametrized tests that exercise
  the real codepath for each standard log level.

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
