# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Per-instance state in `/api/marshal/status`** — the status payload
  now includes a top-level `instances` array with one entry per
  configured Ollama instance: `url`, `kv_cache_type`, `tier_label`,
  `reachable` (boolean — true after a successful `/api/ps` poll, false
  on poll error), `loaded_models` (list of model names on that
  instance), and `used_vram` (sum of model size_vram on that
  instance). Each entry in the existing `loaded_models` array is now
  tagged with `instance_url` and `tier_label` so consumers can
  correlate model → tier without cross-referencing the new array.
  Legacy single-instance configs still produce a one-entry `instances`
  list (the validator-backfilled primary), giving operator tooling a
  consistent shape regardless of multi-instance setup. Backed by new
  `MemoryManager.is_instance_reachable(url)` API; reachability flips
  per-poll (no time-decay logic) and starts False until the first
  successful poll.
- **Three new multi-instance integration tests**:
  - `test_fault_proxy_per_instance_unexpected_unload` — fault-proxy
    based, runs on every integration suite invocation. Verifies that
    when one Ollama instance drops a model unexpectedly while another
    keeps it loaded, marshal counts the unload exactly once and
    attributes it to the correct instance (no double-count, no
    false-positive on the wrong tier).
  - `test_a_rule_strict_q8_to_q4_fallback` — gated on three real
    daemons (`_REQUIRES_THREE_INSTANCES`). Verifies the routing tree
    walks through f16 → q8 → q4 under tight memory pressure and
    chooses q4 with `routing_reason=fallback_no_fit` only when q8
    strictly cannot fit.
  - `test_q4_only_promotes_to_higher_tier_when_room` — gated on three
    daemons. Pre-loads model on q4 only; verifies routing returns
    `routing_reason=promoting_from_last_resort` with `unload_from=[q4]`
    so the scheduler cleans up the stale copy after promotion.
- **`scheduler.benchmark_on_startup` config flag** (default `true`) —
  controls whether marshal runs the model-size benchmark sweep on
  lifespan startup. Production behavior unchanged. Set to `false` in
  integration test harnesses that front Ollama with a fault-injection
  proxy: with per-test temp registry paths the cache is always empty,
  and the benchmark would otherwise load every installed model
  through the proxy on every startup, saturating the upstream
  daemon and starving the test's own request behind 10s+ per model
  load. The integration suite's `make_test_app()` helper now
  force-disables this regardless of what the test's inline
  `SchedulerConfig` says.

### Changed

- **README "Multi-instance setup" walkthrough corrected.** Previous
  versions of the bootstrap walkthrough instructed operators to run
  `ollama pull` against every Ollama instance separately, claiming
  daemons don't share model files. This was wrong — Ollama daemons
  read `~/.ollama/models/` by default (no `OLLAMA_MODELS` env
  override), so a single `ollama pull <name>` is enough; every daemon
  on the box sees the model immediately. The walkthrough now uses
  this happy path with a verification snippet (`curl /api/tags | jq
  '.models | length'` from each daemon should match). The
  per-daemon-pull instruction is preserved as a caveat for operators
  who deliberately set `OLLAMA_MODELS` to isolate per-tier model
  stores.
- **Example launchd plists updated** with explicit comment confirming
  the shared-store behavior — `examples/com.user.ollama-serve-q8.plist`
  and `…-q4.plist` both note "no `OLLAMA_MODELS` env override → this
  daemon shares `~/.ollama/models/` with the primary, no need to
  repull."

- **Multi-instance routing — Stage 2 plumbing (Track 2 stage 2)** —
  the Stage 1 routing decision is now wired through the actual
  request path. `MemoryManager` polls every configured Ollama
  instance independently and tracks loaded models per-instance;
  `ModelLifecycle.preload` / `unload` / `unload_all` accept an
  `instance_url` parameter; `RequestEnvelope` carries `instance_url`
  + `tier_label` + `routing_reason` set by the scheduler from the
  `routing.pick_instance` decision; `forward_request` targets the
  envelope's instance URL (falls back to the primary on legacy
  single-instance setups). VRAM budget stays GLOBAL across instances
  (Mac unified memory is a single pool — partitioning would double-
  count). New `MemoryManager.probe_fit(instance, size, non_idle)`
  returns the routing-aware fit answer (fits / would-evict-non-idle
  / only-idle). New `ModelRegistry.get_kv_per_slot_scaled(model,
  kv_cache_type)` and `get_total_footprint(model, num_ctx,
  kv_cache_type)` apply precision multipliers (f16=1.0, q8_0=0.5,
  q4_0=0.25) so routing's fit math reflects real per-instance cost.
  Audit-log records (`request.served`, `request.failed`) now include
  `instance_url`, `tier_label`, and `routing_reason` so operators
  can answer "why did this request run on q8?" by reading the JSONL
  alone. 35+ new unit tests cover the per-instance memory state,
  routing-decision wiring, and audit-field plumbing. New integration
  test file `tests/integration/test_multi_instance.py` (4 tests)
  validates the cross-component path against a real two-instance
  setup; tests skip cleanly when only :11434 is up.

- **Multi-instance routing foundation (Track 2 stage 1)** — config
  schema + pure decision logic for routing requests across multiple
  Ollama instances at different KV cache precisions. New types in
  `ollama_marshal.config`: `KVCacheType` (f16 / q8_0 / q4_0),
  `OllamaInstance` (frozen Pydantic model with url + kv_cache_type
  + tier_label), `MarshalConfig.instances` (list, auto-derived from
  the legacy singular `ollama.host` form so existing configs work
  unchanged). New `ollama_marshal.routing` module with
  `pick_instance(state, fit_probe)` — pure function implementing the
  decision tree (memory-pressure failover only, never per-program
  "tier preference"). 15 unit tests in `tests/test_routing.py`. No
  consumer reads the new types yet — production behavior unchanged
  in this stage. Stage 2 (plumbing through MemoryManager / Scheduler
  / Lifecycle / Server) ships in a follow-up commit.
- **Integration test suite** — opt-in pytest suite under
  `tests/integration/` (24 tests across 7 files) that validates
  end-to-end behavior against the user's real Ollama. Runs locally
  via `make test-integration`; never in CI. Covers memory handling
  (preload, bin-packing, drain-before-evict, slot allocation,
  reload-on-need, failed-preload sentinel, unexpected-unload
  detection, idle eviction), the v0.4.0 surfaces (retry, fail-fast,
  num_ctx injection, audit log, marshal doctor), and basic smoke
  tests. Also ships a small fault-injection HTTP proxy
  (`tests/integration/_fault_proxy.py`) that lets tests simulate
  Ollama failures (502/503, ConnectError, malformed responses,
  fake `/api/ps`) without restarting the real daemon.
- **`asgi-lifespan>=2.1.0`** dev dependency — runs FastAPI's
  lifespan hooks from async test code without spinning up a uvicorn
  subprocess. Required only for tests under `tests/integration/`;
  production code never imports it.
- **App component handles on `app.state`** — `_scheduler`,
  `_memory`, `_registry`, `_lifecycle`, `_queues` are now stashed
  on the FastAPI `app.state` at end of lifespan startup so
  integration tests (and any future consumer that wants in-process
  introspection) can read them without poking module globals.
  Additive only; production request handlers continue using the
  module globals as before.
- **`pytest` pre-push hook** in `.pre-commit-config.yaml` — full
  unit suite gates every `git push`. Integration tests are excluded
  (they need a running Ollama). Install via the existing
  `make install-dev` target which now runs both
  `pre-commit install` and `pre-commit install --hook-type pre-push`.

### Removed

- **Claude review + security workflows** (`.github/workflows/claude-review.yml`,
  `.github/workflows/claude-security.yml`) and the `CLAUDE_API_KEY`
  GitHub Actions secret. Cost-control measure: the local `/review`
  skill (gstack) catches more bugs anyway — empirically it found 3
  P0/P1 correctness bugs the CI bot missed on PR #6. Default PR CI
  keeps `lint`, `test (...)` matrix, and `guard-workflow-changes`
  (defense-in-depth: blocks accidental future re-adds of the
  workflow files unless on a `chore/ci-*` branch). Replacement
  review path documented in CLAUDE.md.

## [0.4.0] - 2026-04-28

### Added

- **Fail-fast 404 on unknown models** — `/api/chat`, `/api/generate`,
  `/api/embeddings`, and the OpenAI-compat paths now return 404 in
  milliseconds when the requested model isn't installed in Ollama,
  instead of letting the request sit in the queue for up to
  `proxy.request_timeout_s` (1h default) while marshal repeatedly
  tries to preload a non-existent model. Response includes a
  `Run \`ollama pull <model>\`` hint. The check is cached and
  refreshed opportunistically (rate-limited at 5s) so a freshly-pulled
  model is recognized within a few seconds.
- **Marshal-side retry on transient Ollama failures** — new
  `retry` config section (default ON, max 3 attempts). When Ollama
  briefly flaps (daemon recycling, transient 502/503), marshal
  absorbs the blip via in-process retry with exponential backoff +
  full jitter, so the client never sees the failure. Conservative by
  default: streaming requests are never retried, ReadTimeout is not
  retried (risk of re-executing partial generation), and only
  `ConnectError`/`ConnectTimeout` + HTTP 502/503/504 trigger retry.
  Embeddings endpoints opt into ReadTimeout retry automatically since
  they're idempotent.
- **`X-Marshal-Retry-Max` header** — per-request retry override.
  `0` disables retry on a single call (e.g. tool-calling agents that
  want fail-fast); higher values opt into more aggressive retry for
  known-idempotent burst workloads. Capped server-side at 10.
- **`retries_attempted`, `retries_succeeded`, `unexpected_unloads`
  counters** on `SchedulerMetrics`. Persisted across restarts in
  `metrics.json`. `unexpected_unloads` is wired by Surface C2 later
  in this release.
- **Per-program context profiles** (`context.programs.<id>`) with
  `typical_num_ctx` (floor) and `max_num_ctx` (ceiling). Lets a
  tool-calling program declare "round 1's allocation must already
  fit round 5's growth" without exposing every program to the
  pessimal max-context allocation.
- **Load-time slot management + reload-on-need** (Surface C1 Dim 4).
  `lifecycle.preload(model, num_ctx=N)` now passes `options.num_ctx`
  to Ollama at load time so KV cache slots are allocated at the
  right size. The scheduler tracks `_allocated_num_ctx_per_model`
  and, before dispatching any envelope whose computed `num_ctx`
  exceeds the current allocation, drains pending requests for that
  model, unloads, and preloads at the larger size. New
  `reload_count` metric on `SchedulerMetrics` (persisted) lets the
  dashboard surface frequent-reload warnings.
- **Detection of Ollama-side memory-pressure evictions** (Surface
  C2). When marshal observes a model leave `/api/ps` without having
  called `lifecycle.unload()` itself, it logs
  `memory.unexpected_unload` (warning) and increments the
  `unexpected_unloads` counter on `SchedulerMetrics`. Persistent
  non-zero values indicate Ollama-side memory tuning is needed
  (e.g. lower `OLLAMA_NUM_PARALLEL`, set
  `OLLAMA_KV_CACHE_TYPE=q8_0`). The `marshal doctor` CLI (next)
  surfaces specific recommendations.
- **`marshal doctor` CLI subcommand** (Surface C3). Diagnostic
  command that reads `/api/tags`, `/api/show`, and `/api/ps`,
  computes per-model KV cache demand, and recommends specific
  `OLLAMA_*` env vars to set in the launchd plist or systemd unit:
  `OLLAMA_KV_CACHE_TYPE=q8_0` (halves KV cache size),
  `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_NUM_PARALLEL` (computed from
  worst-case KV demand vs system RAM, capped at 4),
  `OLLAMA_MAX_LOADED_MODELS` (computed from mean KV demand). When
  marshal is reachable, the report includes the live
  `unexpected_unloads` counter so the user can confirm tuning
  worked: a non-zero value drops to 0 after applying the
  recommendations.

### Changed

- **BEHAVIOR CHANGE: dynamic `num_ctx` sizing.** v0.3.0 injected
  `num_ctx = model_max_context` unconditionally — a 4B model with a
  262K context window would pre-allocate ~17 GB of KV cache per slot
  on every request, causing Ollama to thrash co-resident models. v0.4.0
  estimates prompt tokens (chars/4 + 20% buffer), adds the completion
  budget + safety buffer, rounds up to a power-of-2 boundary
  (2K…262K), clamps to the program's profile if set, then to the
  model's max. **Marshal NEVER silently truncates a real prompt** —
  when a request needs more context than the model has allocated,
  marshal will reload the model at the larger size (Surface C1 Dim 4,
  shipping in this release). To restore v0.2.x behavior (Ollama
  default + silent truncation), set `context.injection_enabled:
  false`.

### Fixed

Issues found in local `/review` after the initial CI bot review
landed clean — caught real correctness bugs that would have shipped
otherwise.

- **CRITICAL: reload-on-need no longer silently truncates the
  triggering request.** v0.4.0's first-cut drained pending requests
  for the model BEFORE unload — including the very request whose
  `num_ctx > allocated` triggered the reload — so it dispatched
  against the OLD smaller slot and Ollama silently truncated it.
  This defeated the entire stated principle of Dim 4 ("Marshal NEVER
  silently truncates a real prompt"). Fixed: skip the drain;
  unload + preload, then let the next tick dispatch against the new
  larger slot.
- **CRITICAL: client-supplied `options.num_ctx` is now clamped to
  the model's max — UNCONDITIONALLY.** Without this, a request with
  `options.num_ctx: 999_999_999` triggered reload-on-need, failed
  preload, infinite-looped the scheduler, and unboundedly grew
  `metrics.reload_count`. One bad request would brick the proxy for
  everyone. The clamp is a trust-boundary safety check, not part of
  prompt-driven sizing — it runs whether or not
  `context.injection_enabled` is true. Operators who opt out of
  prompt-driven sizing (`injection_enabled: false`) are still
  protected. Non-positive client values are dropped: when injection
  is enabled they fall through to prompt-driven sizing; when
  disabled the request goes out with no `num_ctx` (Ollama default).
- **CRITICAL: failed-preload-after-unload no longer leaves the
  scheduler unable to detect oversized requests.** When unload
  succeeds but preload fails, `_allocated_num_ctx` is now written
  with sentinel `0` instead of being left None. `needs_reload` treats
  `0` as "always reload" so the scheduler retries instead of silently
  dispatching against an unknown slot.
- **`retries_succeeded` is honest now.** `call_with_retry` returns
  `(result, attempts_used, exhausted)`. The scheduler only bumps
  `retries_succeeded` when `not exhausted`. Previously, exhausting
  retries on 502/503/504 (returning the failed response without
  raising) counted as a success — `marshal doctor` would report
  Ollama healthy when every retry actually failed.
- **`metrics.reload_count` only bumps on successful reloads.**
  Previously it incremented before preload was attempted, so failed
  preloads still bumped the counter — combined with the unvalidated
  `num_ctx` bug, an adversarial value produced unbounded counter
  growth across restarts.
- **`X-Program-ID` header is sanitized.** Truncated to 64 chars and
  restricted to `[A-Za-z0-9_.-]`. Without this, an adversarial
  client cycling 10MB header values would inflate burst-hint dicts,
  `_active_programs`, and `audit.jsonl` by 10MB per distinct value.
  Newlines and control chars in the value also corrupted structlog
  console output (log injection).
- **`/api/ps` shape is validated defensively.** Each model entry is
  now type-checked (must be a dict with a string `name`). A
  malformed response (string entries, null `models`, bad
  `size_vram`) used to crash the polling loop and broad-except into
  a single warning, leaving `_loaded_models` stale — a false negative
  on the very signal Surface C2 was meant to catch.
- Removed dead `_RetryableStatusError` class from `retry.py`
  (defined but never raised or caught).

## [0.3.0] - 2026-04-28

### Added

- **`X-Burst-Size` header** — programs that submit work sequentially
  (e.g. tool-calling loops where each call blocks on the previous
  one's response) can now declare expected total demand via
  `X-Burst-Size: N` on each request. Marshal's eviction scorer treats
  the program-model pair as if it had `actual_pending + N` queued
  requests, protecting the model from being evicted mid-burst even
  when only one envelope is currently visible. Hint expires 30s after
  the last refresh; capped server-side at `max_skips × 4` so an
  adversarial header can't starve other programs.
- **Per-model `parallel_per_model` dispatch limit** — new
  `scheduler.parallel_per_model` config field (default 1, preserves
  v0.2.x behavior). When raised — and Ollama is started with
  `OLLAMA_NUM_PARALLEL >= N` — same-model envelopes fan out through
  an `asyncio.Semaphore` and Ollama's pre-allocated KV cache slots
  serve them in parallel. Adapts naturally to queue depth: a 1-deep
  queue still serves 1 at a time.
- **Per-model architecture metadata probe** — on first sight of a
  model, marshal calls `/api/show` to extract `context_length`,
  `block_count`, `embedding_length`, `attention.head_count`,
  `attention.head_count_kv`. Cached at
  `~/.ollama-marshal/model_metadata.json`. Provides the foundation
  for context-window enforcement and parallelism math.
- **Context window enforcement** — marshal now injects
  `options.num_ctx = model_max_context` on every proxied inference
  request that doesn't already set one. Stops Ollama from silently
  truncating context to fit its slot allocation, which previously
  caused a model that supports 32K to quietly run with 4K with no
  error or warning. Client-set `num_ctx` is preserved. Embeddings
  endpoints (`/api/embeddings`, `/v1/embeddings`) are skipped since
  they aren't bitten by the truncation bug.

  **Behavior change / migration note:** loading a model now reserves
  KV cache at its full architectural context length, not Ollama's
  default 2048. VRAM per loaded model can rise 4-16x for models with
  large max contexts. A loadout that fit in v0.2.x may no longer
  co-resident the same set of models — bin-packing will evict
  smaller co-residents to make room. To opt out per-request, set
  `options.num_ctx` explicitly. To opt out globally for one model,
  set its expected `num_ctx` in the client request body.
- **Persisted scheduler metrics** — `requests_served`, `model_swaps`,
  `evictions`, `total_wait_ms` now survive marshal restarts via
  `~/.ollama-marshal/metrics.json`. Loaded on startup, saved on
  shutdown + every 60s. Schema-version-aware (refuses to load
  mismatched versions, falls back to fresh).
- **Audit log feature flag** — new `audit.enabled` config (off by
  default). When enabled, marshal appends one JSON record per
  request lifecycle event to a configurable JSONL file. Records
  contain metadata only — never prompt text or response content.
  Configurable retention (`retention_days`, default 30) and
  size-based rotation (`max_size_mb`, default 100). Suitable for
  compliance / forensics / per-program usage analytics.

### Changed

- `_process_batch` no longer serializes non-embedding requests through
  `_forward_single`. Same-model envelopes are now gated through
  `InflightTracker.semaphore_for(model)` whose size matches
  `scheduler.parallel_per_model`. With the default `parallel_per_model:
  1` this is observationally identical to v0.2.x behavior.
- `scheduler._tick` adds a sixth step: prune expired X-Burst-Size hints.

### Fixed

- `model_metadata.json` cache now drops entries when the source model
  is deleted from Ollama, matching the existing `model_sizes.json`
  cleanup behavior.

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

[Unreleased]: https://github.com/robertvitali/ollama-marshal/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robertvitali/ollama-marshal/releases/tag/v0.1.0
