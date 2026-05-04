# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.5] - 2026-05-03

P1 observability + correctness release. Closes out the
`BUGFIX-NOTES.txt` fixes started in v0.6.4. None of these are active
production failures; they improve diagnostic clarity, scheduler
reactivity, and confidence in the v0.6.4 Hop 1 unbounded design.

### Fixed

- **Error responses now propagate the actual exception class and
  message** instead of a generic `{"error": "Internal proxy error"}`
  502 for every failure. When marshal's call to Ollama fails:
  - **Ollama-native endpoints** (`/api/chat`, `/api/generate`,
    `/api/embeddings`) return `{"error": "<actual message>",
    "error_type": "<class>", "model": "<name>"}`.
  - **OpenAI-compat endpoints** (`/v1/chat/completions`,
    `/v1/completions`, `/v1/embeddings`) return `{"error":
    {"message": "...", "type": "proxy_error", "code": "proxy_error",
    "exception_type": "<class>"}}` — the OpenAI-spec `type` slug stays
    `"proxy_error"` for backwards compatibility with clients matching
    on it, and the actual Python exception class moves to a new
    `exception_type` field.
  - **Status codes** map by exception class hierarchy:
    `httpx.TimeoutException` (parent of `ReadTimeout`,
    `ConnectTimeout`, `WriteTimeout`, `PoolTimeout`) → **504**;
    `httpx.NetworkError` (parent of `ConnectError`, `ReadError`,
    `WriteError`) and `PreloadFailedError` → **503**; everything else
    (`RemoteProtocolError`, unknown exceptions) → **502**.
  - **Operators triaging from a client error body alone** can now
    distinguish "Ollama took too long" (504) from "can't reach Ollama
    or model couldn't be loaded" (503) from "protocol or unknown
    failure" (502).

  Localhost-only deployments today: httpx exception messages may
  include the upstream URL (e.g., `http://localhost:11434`) — this
  is non-sensitive in the local-only architecture but is documented
  here so any future remote-Ollama support triggers a re-review of
  the error-message exposure path.
- **Scheduler reacts to Ollama-side LRU evictions within ~100 ms**
  (one scheduler tick) instead of waiting up to 5 s for the next
  `/api/ps` poll. When Ollama silently evicts a loaded model
  (memory pressure, internal cleanup), marshal previously kept
  dispatching requests to a model that no longer existed in VRAM,
  triggering surprise cold-start loads. Now the memory poller surfaces
  evictions for the scheduler's next tick to consume; dispatch finds
  the model marked-for-reload and forces a fresh preload before
  forwarding. Per `BUGFIX-NOTES.txt`: 84 `memory.unexpected_unload`
  events in one experiment run — each previously imposed up to 5 s of
  stale-state scheduling.
- **`test_unexpected_unload_detection` integration flake eliminated.**
  Previous assertion read the process-wide `scheduler.metrics.
  unexpected_unloads` counter, which (a) collided with the prod
  marshal's eviction noise (cross-model counter pollution) and (b)
  raced the scheduler-tick drain (counter is the drained-into metric;
  test's `wait_for(not is_loaded)` returned before the next tick had
  drained). The new per-model `_recent_unexpected_unloads` record
  (added for the reactivity feature above) gives the test a
  per-model handle that doesn't collide and is populated synchronously
  inside `_update_from_ps`, so no tick-drain race.

### Added

- **Load test suite** at `tests/integration/test_load.py` marked with
  `@pytest.mark.load` (excluded from default `make test-integration`
  and `make pre-pr`). Run with `make load-test`. Verifies marshal's
  Hop 1 unbounded behavior is healthy under sustained queue pressure
  (50 concurrent clients), mixed-model contention, latency
  percentiles (p50/p95/p99), and pause/resume recovery under load.
  Designed as a pre-release sanity check rather than a per-commit
  gate.

## [0.6.4] - 2026-05-03

P0 stability release. Two production-impacting fixes identified
during the v0.6.2 retrospective and verified against
`BUGFIX-NOTES.txt` evidence (qwen3.5 experiment logs, 1531 threads,
67 errors).

### Changed (BREAKING)

- **`proxy.request_timeout_s` removed.** Marshal no longer caps the
  client→marshal wait. Async clients wait indefinitely for a response
  or error. Configs that still set this field will fail validation
  with a clear pydantic error — remove it. The motivation: the field
  was misnamed and surface-confusing. It gated Hop 1 (the queued
  request's wait time inside marshal) but never propagated to Hop 2
  (the actual marshal→Ollama HTTP call), so a 5-min ReadTimeout on
  the forward could fire even when `request_timeout_s=3600`. Splitting
  the surface into "Hop 1 unbounded, Hop 2 explicit knob" makes the
  semantics matched to the actual failure modes.
- **`X-Request-Timeout` header repurposed.** Previously meant "max
  wait for marshal to start serving the request" (Hop 1). Now means
  "Ollama forward call budget" (Hop 2) — the wall-clock timeout
  marshal applies to the httpx call to Ollama. Most callers will
  want the new behavior; if you depended on the old semantics, set a
  client-side socket timeout instead. Pre-1.0 break documented under
  the v0.6.4 versioning policy.

### Fixed

- **Hop 2 forward timeout is now configurable + per-request
  overridable.** `src/ollama_marshal/stream.py` and
  `src/ollama_marshal/lifecycle.py` previously hardcoded
  `timeout=300` (5 min) for marshal→Ollama HTTP calls. For long-
  context inference on bigger models (e.g. 70B with 32K+ tokens),
  Ollama can legitimately need 5+ minutes; marshal would kill the
  connection at 5 min and return 502 to the client. Per
  `BUGFIX-NOTES.txt`: 477 ReadTimeout occurrences in one experiment
  run (some amplified by Bug 2 below; remainder is real big-model
  long-context behavior). The hardcoded constants are gone — the
  scheduler threads `scheduler.ollama_forward_timeout_s` (default
  3600 s = 1 h) through `forward_request` and
  `lifecycle.preload`. Per-request override via the repurposed
  `X-Request-Timeout` header.
- **Per-model preload backoff prevents the 313-failures-in-30s
  cascade.** When Ollama crashes/restarts, `lifecycle.preload`
  returns False on every attempt while Ollama is unreachable. The
  `_SCHEDULER_TICK = 0.1` loop kept calling `preload` immediately
  on the next tick, generating ~10 preload attempts per second,
  hammering the recovering Ollama, spamming logs, and likely slowing
  recovery. Per `BUGFIX-NOTES.txt`: 313 `lifecycle.preload_failed`
  log entries in ~30 s during one Ollama crash recovery. Now: each
  model has independent failure tracking
  (`Scheduler._preload_failures`), exponential backoff with full
  jitter (1 s → 2 s → 4 s → ... up to 30 s by default), and after
  5 consecutive failures the queued envelopes for that model fail
  with `PreloadFailedError` (surfaced as 502) rather than waiting
  forever. Failure state clears on the next successful preload — the
  giveup is per-batch, not permanent.

### Added

- `scheduler.ollama_forward_timeout_s` (default 3600) — wall-clock
  budget for a single marshal→Ollama HTTP forward, in seconds.
  Threaded through `forward_request` and `lifecycle.preload` so both
  the inference call and the load call share the same configurable
  budget. Override per-request via `X-Request-Timeout`.
- `scheduler.preload_backoff_base_s` (default 1.0) — initial backoff
  delay after a preload failure.
- `scheduler.preload_backoff_max_s` (default 30.0) — cap on a single
  preload backoff sleep.
- `scheduler.preload_max_consecutive_failures` (default 5) — give up
  + fail queued envelopes after N consecutive failures for the same
  model.
- `PreloadFailedError` exception (in `ollama_marshal.queue`) —
  surfaced as `RequestEnvelope.error` when the scheduler exhausts
  its per-model preload retry budget. Currently maps to a generic
  502 in the response body; v0.6.5's Bug 3 will propagate the
  class name and reason into the error body for clearer client
  diagnostics.
- `scheduler.preload_failure_recorded`,
  `scheduler.preload_failure_cleared`, and
  `scheduler.preload_giving_up` log events — emitted by the new
  per-model backoff state machine. Pair with
  `scheduler.preload_giving_up` audit-log entries when audit is
  enabled.

### Documentation

- `marshal.example.yaml`: removed the `proxy.request_timeout_s`
  entry (replaced with a migration note pointing at
  `scheduler.ollama_forward_timeout_s`); added annotated entries
  for the four new `scheduler.*` fields.
- `CHANGELOG.md`: documented the breaking-change rationale at
  the top of v0.6.4 so operators upgrading from v0.6.3 can see
  the migration path before reading the field docs.

## [0.6.3] - 2026-05-03

### Fixed

- **Cross-suite contamination of integration tests resolved.** The
  integration suite shares Ollama at `:11434` with any local prod
  marshal at `:11435`. When prod had a model loaded (e.g.
  `qwen3.5:2b-q8_0` for ai-email-triage), test marshals racing
  against prod's load state caused 3 reproducible failures in
  `test_multi_instance.py`, a documented flake on
  `test_bin_packing_keeps_multiple_models_loaded`, and real prod
  `server.request_error` events during the v0.6.2 integration test
  run on 2026-05-03. New session-scoped autouse pytest fixture
  `pause_local_prod_marshal` (in `tests/integration/conftest.py`)
  pauses the local prod marshal via the v0.6.0 admin/pause endpoint
  for the duration of the suite, then resumes.
  - Pause is sent with `drain_timeout_s=0` (set the pause flag
    immediately, don't wait for prod's in-flight inferences to
    drain). In-flight inferences complete naturally while the
    suite runs. Empirically: waiting on drain (e.g. `=60`) blocks
    fixture setup AND introduced new flakes in multi-instance
    routing tests when prod's in-flight finished mid-test. The
    zero-drain path is the right semantic for "don't compete with
    test scheduling" and shrinks suite wall-clock from ~6 min to
    ~2:30 on the maintainer's M3 Ultra. Override with
    `MARSHAL_TEST_PAUSE_DRAIN_S` for stricter semantics (e.g.
    before destructive admin work).
  - Pause endpoint returns 200 (drain complete) OR 409 (drain
    timeout) — both leave the pause flag in effect server-side per
    `Scheduler.pause()` contract. Fixture treats both as "paused"
    and always calls `resume` on teardown.
  - Auto-resume failsafe defaults to **1 hour**
    (`MARSHAL_TEST_PAUSE_TIMEOUT_S=3600`) so a long integration run
    can't trip the failsafe mid-suite and re-introduce the
    contamination it exists to prevent. Up from the prior 10 min
    `auto_resume_after_seconds` value the unused `prod_marshal_pause`
    fixture used (raised here because the autouse path covers the
    full suite, not a small subset).
- **Token discovery without sourcing the env file.** New
  `tests/integration/_admin_token.py` helper reads
  `MARSHAL_TEST_ADMIN_TOKEN` / `MARSHAL_TEST_BYPASS_TOKEN` from the
  environment first, then falls back to parsing
  `~/.ollama-marshal/admin-tokens.env` directly (with mode-600
  enforcement — refuses to read a world/group-readable token file).
  Contributors no longer need to `source` the file before running
  `make test-integration`.

### Changed

- **Pre-push integration test hook restored** in
  `.pre-commit-config.yaml`. Was deferred from v0.6.2 because the
  same contamination would have hit on every push. With the
  autouse pause fixture in place, the hook is back as
  `pytest-integration` at `stages: [pre-push]`, restoring "pre-PR
  enforcement of integration tests" that was the original v0.6.2
  intent.
- **`prod_marshal_pause` fixture refactored** to depend on the new
  autouse `pause_local_prod_marshal`. Tests that hit prod marshal
  directly via `prod_marshal_client` now skip cleanly when the
  autouse pause didn't take effect, rather than each path
  re-implementing the discovery + skip logic.

### Documentation

- `CONTRIBUTING.md`: integration tests now run on `git push`
  automatically; documented the
  `MARSHAL_INTEGRATION_SKIP_PROD_PAUSE=1` escape hatch.
- `marshal.example.yaml`: added a note in the `admin:` section
  pointing integration test contributors at
  `~/.ollama-marshal/admin-tokens.env` and the autouse pause
  fixture.

## [0.6.2] - 2026-05-03

### Fixed

- **`make lint` now agrees with the pre-commit ruff hook** by
  invoking ruff via `uv run --extra dev` instead of the bare
  `ruff` binary on `$PATH`. Previously, on systems with an older
  system-wide ruff (e.g. pyenv shim 0.13.x), `make lint` and the
  pre-commit ruff hook (pinned 0.15.12) disagreed on rules like
  S603. The other Makefile targets (`format`, `typecheck`, `test`,
  `test-integration`) were updated to the same `uv run --extra dev`
  pattern for consistency, but note that the corresponding
  pre-commit hooks for mypy and pytest still use `language: system`
  and resolve from `$PATH` (only the ruff hook is pinned).
- **v0.1.0 release date corrected** in CHANGELOG.md from
  `2026-04-24` (initial commit date) to `2026-04-27` (PR #1 merge
  date — the actual ship date). v0.1.0 was also retroactively
  tagged at the PR #1 merge commit (`3ba7133`); previously the
  repo's tag history started at v0.2.0, leaving v0.1.0 untagged.
- **`warn_unused_configs` mypy flag flipped to `false`** in
  `pyproject.toml`. The flag previously fired on every `make
  typecheck` run because the `[[tool.mypy.overrides]] module =
  "tests"` block doesn't match anything when only `src/` is being
  analyzed (the Makefile + pre-commit invocation). The override
  module pattern was also corrected to `["tests", "tests.*"]` so
  it actually applies to test submodules under ad-hoc
  `mypy tests/` (the bare `module = "tests"` form only matched the
  package `__init__`, not its submodules). With the override
  applied, `mypy tests/` produces ~600 errors instead of ~1.7K —
  tests are not strict-clean (and not part of the typecheck
  contract), but the override cuts the noisiest categories.
  Tradeoff: lose the unused-config diagnostic globally to keep the
  override functional across both invocation paths.

### Changed

- **Hook reorganization** in `.pre-commit-config.yaml`:
  - Unit tests (`pytest tests/ --ignore=tests/integration`) now
    run on **every commit** (was: pre-push only). Adds ~10s per
    commit but catches regressions before they reach the branch
    tip.
  - Hook invokes via `uv run --extra dev pytest` to match the
    Makefile's pinned-version pattern. The `mypy` pre-commit hook
    still uses `language: system` (out of scope for v0.6.2).
  - A pre-push **integration** test hook is intentionally NOT
    added in this release. The cross-suite contamination caused
    by sharing Ollama with a local prod marshal at `:11435` makes
    every push hit false-failures (and can spike real prod
    errors). The fix lands in v0.6.3 as a pytest session-scoped
    fixture that pauses prod marshal via the v0.6.0
    `admin/pause` endpoint during the suite. Once that lands, the
    pre-push hook returns. Until then, use `make pre-pr` as the
    manual gate.
- **New `make pre-pr` target** — runs `check` (lint + typecheck +
  unit tests) plus `test-integration`. The interim manual gate for
  pre-PR integration test enforcement until v0.6.3's prod-pause
  fixture lands.

### Documentation

- `CONTRIBUTING.md`: clarified that `uv` is **required** (not
  "recommended") for the `make` targets after the `uv run` switch.
- `CLAUDE.md`: updated the "Development Commands" section to
  reflect the `uv run` invocation pattern.
- v0.6.1's `### Deferred to v0.6.2` section renamed to `### Deferred
  to a future release` since v0.6.2 ships as a build-tooling patch
  and does not address those items (memory_behavior tests,
  multi_instance, sentinel test, removing `_marshal_internals`).
- Added missing CHANGELOG link references for `[Unreleased]` and
  every released version (`[0.4.0]` through `[0.6.2]`) so the
  markdown anchors resolve.

## [0.6.1] - 2026-05-02

### Added

- **`/api/marshal/debug` endpoint extension** — now exposes
  `memory.allocated_num_ctx_per_model` (flattened across instances,
  first-instance-wins to match
  `MemoryManager.get_allocated_num_ctx(instance_url=None)`),
  `memory.loaded_per_instance` (per-instance loaded model lists),
  and `registry.metadata_per_model` (the architecture/max_ctx/
  kv_per_slot cache). Lets integration tests assert on marshal-
  internal state via HTTP rather than `app.state._marshal_internals`.
### Changed

- **12 of 27 success integration tests migrated to the v0.6.0
  subprocess pattern**: `test_smoke` (3), `test_fail_fast` (2),
  `test_audit` (2 happy-path; failure path stays ASGI),
  `test_num_ctx` (3), `test_memory_behavior::test_preload_populates_loaded_models`,
  plus `test_doctor::test_doctor_cli_produces_recommendations` was
  already a CLI subprocess test (counted as migrated for
  consistency).
  Internal-state reads (`app.state._marshal_internals.memory.*`,
  `.scheduler.metrics.*`, `.registry.*`) replaced with GETs against
  `/api/marshal/status` and the extended `/api/marshal/debug`
  endpoint. Tests now exercise the same wire format prod operators
  see — sockets, headers, HTTP parsing — instead of the in-process
  ASGI shortcut.
- `test_fail_fast::test_unknown_model_returns_404_fast` budget
  relaxed from 500ms to 750ms. The 500ms target documents the
  production guarantee; subprocess tests share Ollama with whatever
  else is running, so the /api/tags probe occasionally takes longer
  than ideal. 750ms gives that headroom without losing
  regression-detection power for the "request sat in queue waiting
  for non-existent model" bug class.

### Deferred to a future release

(Originally labeled "Deferred to v0.6.2" — v0.6.2 shipped as a
build-tooling patch and did not address these items. They remain
deferred and will likely land alongside the v0.7.0 onboarding
work.)

The remaining 15 tests still use the in-process ASGI pattern. Each
needs additional infrastructure that would have ballooned v0.6.1's
review surface:

- **`test_memory_behavior` (8 tests)** — most need either
  custom subprocess configs (memory budget, idle eviction interval)
  or stay ASGI as failure-path tests using `fault_proxy`.
  `test_bin_packing_keeps_multiple_models_loaded` was attempted but
  surfaced cross-suite contamination flakes in BOTH ASGI and
  subprocess patterns (the user's prod marshal at :11435 competes
  for VRAM since both share the same Ollama at :11434). Requires
  test-isolated Ollama daemon to fix cleanly.
- **`test_multi_instance` (6 daemon-gated tests)** — need
  multi-instance subprocess config helper (3 separate Ollama
  subprocesses + per-tier marshal config). Significant new
  infrastructure.
- **`test_failed_preload_writes_sentinel_allocation`** — needs
  conversion from monkey-patching `_marshal_internals.lifecycle.preload`
  to `fault_proxy.fail_next("/api/generate")`. Then joins the ASGI
  failure-path bucket per the v0.6.0 plan Path C.
- **Removing `app.state._marshal_internals` SimpleNamespace** —
  blocked on all 27 migrations being complete. Currently 12 done.

## [0.6.0] - 2026-05-02

### Added

- **Admin pause/resume + debug config foundation** (v0.6.0 Stage 1).
  New `AdminConfig` section (`pause_endpoints_enabled`, `admin_token`,
  `test_bypass_token`) with a Pydantic validator that rejects
  enabled-without-token configs to prevent accidentally exposing
  unauthenticated endpoints. New `DebugConfig` section
  (`endpoint_enabled`) for the upcoming `/api/marshal/debug`. Both
  default OFF; production marshal stays lean. Both env vars
  (`MARSHAL_ADMIN_*`, `MARSHAL_DEBUG_*`) parse to real Python
  booleans via the explicit bool coercion list.
- **Scheduler pause/resume state machine** (v0.6.0 Stage 1, no
  behavior change yet). New public API: `Scheduler.pause(drain_timeout_s)`,
  `resume()`, `is_paused()`, `in_flight_count()`. Soft-pause
  semantics: pause flips a flag and waits for in-flight dispatches
  to drain before returning True; never rejects new requests, never
  affects the queue. The HTTP endpoints + dispatch loop integration
  ship in the next chunk.
- **`RequestEnvelope.bypass_pause` field** (default `False`).
  Carried through the queue so scheduler dispatch can identify test
  traffic that bypasses the admin pause guard.
- **Admin pause/resume HTTP endpoints** at
  `POST /api/marshal/admin/pause` and
  `POST /api/marshal/admin/resume`. Both gated by
  `admin.pause_endpoints_enabled=true` (return 404 otherwise) and
  require an `X-Marshal-Admin-Token` header matching
  `admin.admin_token` (return 401 otherwise). Pause accepts a JSON
  body with `drain_timeout_s` (default 60) and
  `auto_resume_after_seconds` (default 300); returns 200 with
  `{drained_in_flight, queued_at_pause, auto_resume_at}` on success
  or 409 if drain timed out (with the same fields plus current
  `in_flight` count). Resume returns 200 with the current
  `queue_depth`. Auth uses `secrets.compare_digest` for
  constant-time comparison. Audit events
  (`admin.dispatch_paused`, `admin.dispatch_resumed`,
  `admin.drain_timeout_exceeded`, `admin.auto_resumed`) record on
  every state change.
- **Auto-resume failsafe** in the scheduler. `Scheduler.pause` now
  schedules an `_auto_resume_after` background task that flips
  `_dispatch_paused` back after the configured delay. Defends
  against test-session crashes that would otherwise leave prod
  paused indefinitely. Cancelled by explicit `resume`; reset by
  re-`pause`; cleaned up by `Scheduler.stop`.
- **`GET /api/marshal/debug` endpoint** gated by
  `debug.endpoint_enabled=true` (return 404 otherwise). Returns
  scheduler metrics + pause state + in-flight count for integration
  tests that assert on marshal-internal state via HTTP rather than
  reaching into `app.state._marshal_internals`. Production marshal
  keeps this endpoint disabled to avoid leaking internals.
- **Bypass-token detection on inference requests.** The
  `_enqueue_inference` and `_enqueue_and_wait` request handlers
  now read `X-Marshal-Test-Bypass`; matching the configured
  `admin.test_bypass_token` flips `RequestEnvelope.bypass_pause` to
  `True` (constant-time compare). The scheduler dispatch loop
  changes that respect the flag during pause ship in v0.6.0
  Stage 2.

### Changed

- **CRITICAL-priority requests are exempt from the `max_skips`
  fairness floor.** Previously, the scheduler's bin-pack step
  incremented `skip_count` on every pending envelope of a model that
  didn't fit in VRAM, regardless of priority. CRITICAL programs have
  a dedicated preemption path (`_handle_critical_preemption`) so the
  fairness floor doesn't apply to them — incrementing their skip
  counter would surface spurious `scheduler.forced_load` log noise
  and double-handle requests already covered by preemption. The new
  `ModelQueues.increment_skips_for_model(exclude_program_ids=...)`
  parameter lets the scheduler exempt CRITICAL-priority program IDs.
  Default behavior unchanged for callers that don't pass the new
  kwarg.

### Deferred to v0.6.1

The plan called for migrating all 27 success-path integration tests
to either Path A (live prod marshal) or Path B (subprocess on
ephemeral port). The fixture infrastructure to do so is fully shipped
in v0.6.0 and verified working end-to-end via 8 meta-tests in
`tests/integration/test_infra_subprocess.py`. The actual per-test
migrations are mechanical but high-volume; deferring them to a
focused v0.6.1 PR keeps v0.6.0's review surface manageable. The
existing tests continue to use the in-process ASGI pattern in the
meantime — they pass in isolation; cross-suite contamination flakes
remain a known issue that the migrated subprocess pattern will fix
incrementally.

Also deferred: removing `app.state._marshal_internals` (depends on
all tests being migrated) and the `make_test_app` helper (same).

### Operator action required

Update `~/.ollama-marshal/marshal.yaml` admin section once you've
generated tokens (use `openssl rand -hex 32` for each):

```yaml
admin:
  pause_endpoints_enabled: true
  admin_token: "<long-random-token>"
  test_bypass_token: "<another-long-random-token>"
debug:
  endpoint_enabled: true   # OFF in production unless integration tests need it
```

Then export the matching tokens for the test session:

```bash
export MARSHAL_TEST_ADMIN_TOKEN="<same as admin.admin_token>"
export MARSHAL_TEST_BYPASS_TOKEN="<same as admin.test_bypass_token>"
```

Restart prod marshal so the new config takes effect:

```bash
launchctl unload ~/Library/LaunchAgents/com.user.ollama-marshal.plist
launchctl load ~/Library/LaunchAgents/com.user.ollama-marshal.plist
```

If your prod marshal.yaml has `ai-portfolio-rebalance` (or similar
ai-portfolio programs) under `programs:`, flip them to
`priority: critical` so they get the dedicated preemption path AND
the new max_skips exemption.

## [0.5.1] - 2026-05-01

### Fixed

- **Lifespan task cleanup is now exception-safe.** The pre-yield setup
  is wrapped in `try/finally` so the `benchmark_task` and
  `metrics_persister` background tasks always get cancelled and awaited
  on shutdown, even if startup raised partway through (e.g. between
  task creation and the yield). Previously a startup exception would
  leak both tasks because the cleanup branch was sequenced after the
  yield. New helper `_shutdown_task` always awaits the task (instead
  of skipping on `done()`) so a benchmark sweep that fails by raising
  surfaces its exception via structlog's `server.benchmark_task_failed`
  / `server.metrics_persister_failed` events instead of riding silently
  as `Task exception was never retrieved` at GC time.
- **`MARSHAL_SCHEDULER_BENCHMARK_ON_STARTUP` env var now coerces to
  bool correctly.** The v0.5.0 release added the field but forgot to
  list it in `_apply_env_overrides`'s explicit bool list, so
  `MARSHAL_SCHEDULER_BENCHMARK_ON_STARTUP=false` was passed through as
  the literal string `"false"` and relied on Pydantic's lax-mode
  coercion. Now parses to a real Python bool with the same
  `true|1|yes` truthiness rules as the other bool env overrides.

### Tests

- **Regression test for the v0.5.0 benchmark gate** —
  `test_lifespan_skips_benchmark_when_disabled` asserts
  `benchmark_unknown` is NOT called when `benchmark_on_startup=False`,
  guarding against a future flip of the production default or
  accidental removal of the gate.
- **Regression tests for `MARSHAL_SCHEDULER_BENCHMARK_ON_STARTUP`
  env override** — verifies both `false` and `true` parse correctly
  and produce real Python booleans matching the documented contract
  in `marshal.example.yaml`.
- **Inline guard in `tests/integration/conftest.py::make_test_app`** —
  asserts the `model_copy` belt-and-suspenders override actually
  produced `benchmark_on_startup=False`. Catches future Pydantic
  semantics changes loudly during test setup instead of silently
  saturating the upstream Ollama through the test fault_proxy.

### Build

- **`uv.lock` is now tracked in version control** per the comment in
  `.gitignore` (`# Similar to Pipfile.lock, it is generally
  recommended to include uv.lock in version control.`). Locks the
  dependency tree for reproducible local + CI builds across the
  v0.5.x series.

## [0.5.0] - 2026-05-01

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

## [0.1.0] - 2026-04-27

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

[Unreleased]: https://github.com/robertvitali/ollama-marshal/compare/v0.6.3...HEAD
[0.6.3]: https://github.com/robertvitali/ollama-marshal/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/robertvitali/ollama-marshal/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/robertvitali/ollama-marshal/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/robertvitali/ollama-marshal/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robertvitali/ollama-marshal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robertvitali/ollama-marshal/releases/tag/v0.1.0
