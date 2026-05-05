# CLAUDE.md — ollama-marshal

## What This Is

A model-aware scheduling proxy for Ollama. Sits between client programs and
a single Ollama instance. Eliminates model thrashing with FIFO + bin-packing
scheduling that maximizes VRAM utilization.

## Tech Stack

- Python 3.11+ / FastAPI / uvicorn (async HTTP proxy)
- httpx (async client to Ollama)
- Typer (CLI) / Pydantic (config validation) / structlog (logging)
- Hatchling (build) / Ruff (lint+format) / mypy strict (type check)

## Project Layout

```
src/ollama_marshal/    — package source (13 modules)
tests/                 — pytest unit + integration tests
```

Modules in dependency order: config → routing → audit → queue → registry →
memory → lifecycle → scheduler → stream → openai_compat → server →
dashboard → cli

(`routing.py` is pure decision logic — no I/O, no async, no
MemoryManager. Imports only `config.py`. Easy to test in isolation.
See `pick_instance()` for the multi-instance failover decision tree.)

(`audit.py` only depends on `config.py` — listed near the top to reflect
the actual import graph. The scheduler intentionally avoids importing
audit directly; it gets a duck-typed audit instance via `Scheduler.audit`
attribute injection from the server lifespan.)

## Development Commands

```bash
make install-dev    # install editable + dev deps + pre-commit hooks
make test           # uv run --extra dev pytest (with 95% coverage gate)
make lint           # uv run --extra dev ruff check
make format         # uv run --extra dev ruff format
make typecheck      # uv run --extra dev mypy src/
make check          # lint + typecheck + test
```

## Code Conventions

- Async throughout — FastAPI + httpx async + asyncio
- Type hints on all function signatures (mypy strict)
- Google-style docstrings on all public functions
- No hardcoded values — everything via config (YAML → env → CLI)
- Structured logging with structlog (never use print())
- Line length: 88 chars (ruff default)
- Test naming: test_<feature>_<condition>_<expected_result>

## Python Conventions

Idiomatic patterns beyond the basics above:

**Async correctness**
- Never `time.sleep()` in async code — use `await asyncio.sleep()`
- Never run sync I/O (file reads, blocking HTTP) inside `async def` —
  use `aiofiles` or `httpx.AsyncClient`
- Fire-and-forget tasks: store the reference (`task = asyncio.create_task(...)`),
  don't drop it on the floor
- HTTP clients: `async with httpx.AsyncClient() as client:` — never bare
  `httpx.AsyncClient()` without context manager

**Exception handling**
- No bare `except:` and no `except Exception:` without re-raise or
  structured log entry
- Catch specific exception types (`httpx.HTTPError`,
  `pydantic.ValidationError`, etc.)
- Validate at trust boundaries (incoming HTTP, file I/O, env vars);
  trust internal calls

**Type hints**
- Prefer `Protocol` over `ABC` for interfaces (duck-typing friendly)
- `from __future__ import annotations` at top of every module
- Avoid `Any` — if you need it, leave a one-line comment explaining why
- Use `X | None` (PEP 604) over `Optional[X]`

**Pydantic v2**
- `model_config = ConfigDict(...)` — not the legacy `class Config:`
- `model_validate()` over `parse_obj()`; `model_dump()` over `.dict()`
- Field validators: `@field_validator("name")` decorator

**Filesystem and IO**
- `pathlib.Path` over `os.path` string manipulation
- Default mutable arguments are forbidden:
  `def f(x: list = [])` → `def f(x: list | None = None)`

**Logging (structlog)**
- Key-value pairs: `logger.info("event.name", key=value)`
- No f-strings in log messages: `logger.info(f"failed: {x}")` is wrong
- Event names use dotted namespaces: `scheduler.tick`,
  `memory.budget_exceeded`

## Config Precedence

CLI flags > env vars (MARSHAL_*) > marshal.yaml > defaults

Config discovery: --config flag → MARSHAL_CONFIG env → ./marshal.yaml
→ ~/.ollama-marshal/marshal.yaml → built-in defaults

## Endpoint Routing Rules

Queued (through scheduler): /api/chat, /api/generate, /api/embeddings,
/v1/chat/completions, /v1/completions, /v1/embeddings

Pass-through (no scheduling): /api/tags, /api/ps, /api/show, /api/version

Blocked (return 403 — manage models via Ollama directly): /api/pull,
/api/delete, /api/copy

Local (served by proxy): /api/marshal/status, /status (alias)

Admin (gated by `admin.pause_endpoints_enabled` + bearer token):
POST /api/marshal/admin/pause, POST /api/marshal/admin/resume

Debug (gated by `debug.endpoint_enabled`, OFF in production):
GET /api/marshal/debug

## Scheduling Algorithm

1. FIFO baseline — respect arrival order
2. Bin-pack — fill VRAM by loading smaller models that fit alongside current
3. Skip limit — per-request counter; after max_skips, force-load the model.
   CRITICAL programs are EXEMPT from skip increment (their dedicated
   preemption path in step 5 already guarantees forced load).
4. Eviction — least disruptive: fewest pending, lowest priority, oldest
5. Priority — normal (drain-then-evict) vs critical (can preempt)
6. Immediate — if model already loaded, forward without queuing
7. Pause — when `admin.pause` flips `_dispatch_paused`, the scheduler
   suspends queue draining. Bypass-flagged envelopes
   (`X-Marshal-Test-Bypass` header) still dispatch. Resume picks up
   where it left off; auto-resume failsafe fires after the configured
   timeout if no explicit resume arrives.

## Testing Rules

- Unit tests: mock Ollama responses via httpx, target 95%+ coverage
- Integration tests: live under `tests/integration/`, MUST use the
  `integration` pytest marker, MUST never run in default `make test`
  or default CI (excluded via `--ignore=tests/integration`), MUST
  skipif `localhost:11434` is unreachable so they fail cleanly when
  Ollama is down
- Never mock internal modules — only mock the Ollama HTTP boundary
- conftest.py has shared fixtures for all Ollama API responses
- Use `pytest` fixtures over `setUp`/`tearDown`; `@pytest.mark.parametrize`
  over loops; `pytest-asyncio` for async tests

### Integration suite layout (v0.5.0+)

- `tests/integration/conftest.py` — shared fixtures (`marshal_app`,
  `marshal_config`, `tmp_marshal_paths`) and a `make_test_app(cfg,
  paths)` helper for tests that build their own MarshalConfig (so
  registry-cache isolation isn't bypassed by inline `create_app(cfg)`
  calls). Tests that need fault injection import the
  `fault_proxy` async context manager directly from
  `_fault_proxy.py` rather than as a pytest fixture.
- `tests/integration/_fault_proxy.py` — bare `asyncio.start_server`
  HTTP/1.1 proxy in front of Ollama. Hooks: `fail_next`,
  `disconnect_next`, `delay_next`, `fake_response`. Used for retry
  tests and the unexpected-unload test
- `tests/integration/_admin_token.py` — token discovery for the
  prod-pause fixtures. Reads `MARSHAL_TEST_ADMIN_TOKEN` /
  `MARSHAL_TEST_BYPASS_TOKEN` from env, then falls back to parsing
  `~/.ollama-marshal/admin-tokens.env` (mode 600 enforced). Used
  by the autouse `pause_local_prod_marshal` fixture (v0.6.3+) so
  the integration suite doesn't compete for Ollama VRAM with a
  local prod marshal at `:11435`. Set
  `MARSHAL_INTEGRATION_SKIP_PROD_PAUSE=1` to opt out.
- Per-area test files: `test_smoke.py`, `test_memory_behavior.py`
  (the main thrust — model loading/unloading correctness),
  `test_fail_fast.py`, `test_num_ctx.py`, `test_retry.py`,
  `test_audit.py`, `test_doctor.py`
- **NO pytest-xdist parallelism inside `tests/integration/`** —
  marshal uses module-level globals (`_scheduler`, `_memory`,
  `_config`); two parallel marshal apps would stomp on each other.
  Default pytest is serial, so no actual change needed; just don't
  add `-n` flags
- Test envelope priority: every test fires with
  `X-Program-ID: integration-test` (CRITICAL priority by default).
  The `marshal_config` fixture also defines `integration-test-normal`
  for the few tests that specifically need normal-priority paths
  (drain-before-evict)
- When adding a new feature, consider whether it needs an integration
  test (cross-component path against real Ollama) or unit test alone
  is sufficient. Cross-component memory-handling features (anything
  touching scheduler ↔ memory ↔ lifecycle) generally NEED integration
  coverage — the unit suite mocks at the httpx boundary, missing
  exactly the bug class /review caught on PR #6 (v0.4.0)

### Integration test infrastructure design decisions

Settled trade-offs that should NOT be re-litigated without new
evidence. Recorded here so future investigators don't redo the
analysis from scratch.

1. **`PAUSE_DRAIN_TIMEOUT_S = 0`** (v0.6.3, reaffirmed v0.6.6).
   The autouse `pause_local_prod_marshal` fixture sets the pause
   flag immediately and does NOT wait for prod's in-flight
   inferences to drain. Reasons: a 60s drain on a busy machine
   added 60s of cold start AND introduced new flakes in
   multi-instance routing tests (suite went 6:13 → 2:34 with
   `=0`, fixing 4 multi-instance failures). The cost — in-flight
   prod inferences keep using Ollama VRAM until they naturally
   finish — is absorbed by item 2 below.

2. **`INTEGRATION_FORWARD_TIMEOUT_S = 900`** (15 min, v0.6.6 raised
   from 120s). The Hop 2 budget (test marshal → Ollama) is large
   enough to absorb VRAM-contention waits opened by the
   `drain_timeout=0` choice above. A tiny qwen3.5:0.8b chat with
   `num_predict=4` should complete in <5s; if it takes >900s
   something is genuinely wrong and worth surfacing as a real bug.
   Test-side `httpx.AsyncClient(timeout=...)` calls match this so
   the test client doesn't ReadTimeout before the marshal does.

3. **Test program priority is CRITICAL but doesn't help VRAM**
   (informational, v0.6.6). Every test request carries
   `X-Program-ID: integration-test` which `marshal_config` maps to
   `Priority.CRITICAL`. CRITICAL preempts other programs in the
   marshal queue, but the TEST marshal has no other traffic — so
   priority is moot at that layer. Contention between test marshal
   and prod marshal happens at OLLAMA's VRAM, where Ollama doesn't
   have a priority concept (FIFO). Don't try to "fix" the
   integration suite by tuning priority.

4. **Pause state is visible on `/api/marshal/status`** (v0.6.6).
   The `paused: bool` field on the canonical (token-free) status
   payload lets the autouse fixture verify the dispatch flag
   actually took effect after `admin/pause` returns success, and
   lets operators see pause state without holding the admin token.
   The fixture's `_verify_paused` step polls this for up to 5s and
   warns loudly if the flag never propagates.

## Documentation Rules

Every PR keeps these docs in sync with code changes (no drift allowed):

- **CHANGELOG.md** — every user-visible change goes under `## [Unreleased]`
  with the right subsection (Added, Changed, Deprecated, Removed, Fixed,
  Security). Pure internal refactors with no user impact may be omitted.
- **README.md** — user-facing API, CLI flag, or config change updates the
  Configuration Reference, Quick Start, or relevant section
- **CLAUDE.md** — new `/api/*` route adds a row to "Endpoint Routing
  Rules"; new convention or invariant added to the right section
- **marshal.example.yaml** — every new config field added in
  `src/ollama_marshal/config.py` must have a corresponding annotated
  entry here (default + purpose comment)
- **CONTRIBUTING.md** — only when contributor workflow changes (rare)

If a doc references a function, file, or pattern that's been renamed or
removed, update or delete the reference in the same PR.

## Safety Rules

- Never commit secrets (.env, API keys, credentials)
- Never hardcode ports, hosts, timeouts, or thresholds
- Override keep_alive on EVERY proxied request (prevents Ollama auto-eviction)
- Drain-before-evict for normal priority (no mid-batch model unloading)
- Graceful shutdown must respect the configured mode (drain vs immediate)

## Code Review — Local Only

This repo previously used `anthropics/claude-code-action@v1` for
automated Claude reviews on every PR. Both `claude-review.yml` and
`claude-security.yml` workflows have been **removed** (v0.5.0), along
with the `CLAUDE_API_KEY` secret, to control Anthropic API costs.

**Current review path** — local only:

1. **`/review`** (gstack skill) — multi-specialist adversarial review
   of the working diff. Runs ruff + mypy + pytest, dispatches
   testing/maintainability/security/performance specialists, captures
   ~24 review categories. This caught 3 P0/P1 correctness bugs the
   CI bot missed on PR #6 — empirically the strongest path.
2. **`/codex`** (gstack skill) — cross-model second opinion via
   OpenAI Codex CLI. Useful for high-stakes PRs where you want
   independent ground truth. Requires `codex` CLI installed.
3. **`/cso`** (gstack skill) — security audit, OWASP-style.
4. **`pre-commit`** — fast checks on every commit (ruff check,
   ruff format, mypy strict, pre-commit-hooks). Add `pytest` via
   `pre-push` if you want test coverage on push (see
   `.pre-commit-config.yaml`).

**Workflow file discipline** still applies for any future cloud review
tooling: changes to security-sensitive workflows (anything that
references a secret) ride on a dedicated `chore/ci-*` branch straight
to `main` — one file, one commit, no other changes. Feature branches
inherit via `git fetch origin main && git rebase origin/main`. The
`guard-workflow-changes` CI job remains as defense-in-depth: it
blocks feature PRs from re-adding any `claude-*.yml` file unless on
a `chore/ci-*` branch.

## Per-Issue Dev Workflow

For multi-issue releases (a v0.X.Y with several sub-bugs under one
parent task), iterate **per issue** rather than batching every change
into one sweep. Each issue gets its own commit on the release feature
branch and its own pass through review. The cadence catches
regressions early and produces a reviewable history.

### For code or test-coverage issues

1. Read the affected code paths first; understand existing behavior
   before changing it.
2. Implement the fix or new test.
3. Update unit tests (`tests/test_*.py`) to cover the new behavior.
4. Author new integration tests where the change crosses the
   scheduler ↔ memory ↔ lifecycle boundary or hits a wire-level path
   not exercised by the unit suite (`tests/integration/test_*.py`).
5. Run `make test` + `make test-integration`. Fix anything that
   fails.
6. Run `/review` (gstack skill — runs Claude structured + Claude
   adversarial + Codex adversarial passes, plus Codex structured
   review automatically when the diff is ≥ 200 LOC). For high-stakes
   changes under 200 LOC, ask `/review` for "full review" / "P1
   gate" to force the Codex structured pass.
7. Fix any issues `/review` surfaced.
8. Re-run `make test` + `make test-integration` to confirm review
   fixes didn't regress anything.
9. Commit on the release feature branch (e.g. `feat/v0.6.6`). Add a
   CHANGELOG.md entry under `[Unreleased]` for any user-visible
   change.
10. Update the corresponding Asana subtask: mark complete, and
    append any review-discovered follow-ups as new subtasks at the
    end of the parent release task.

### For verify-only items (load tests, workload re-runs)

Verify items don't produce a code diff. They produce artifacts —
logs, percentile numbers, error counts.

1. Run the verification.
2. Capture artifacts to the Asana subtask body (timestamps, key
   numbers, before/after deltas).
3. If clean, mark the subtask complete.
4. If the verification surfaces an active issue, file it as a new
   subtask under the same release parent task and run the
   code/test-coverage workflow above on the new subtask.

### Ordering inside a release

Sequence subtasks so test infrastructure and coverage land before
production code changes. A stable green test suite is a precondition
for trustworthy verification of subsequent changes. General order:

1. Test-isolation / flake fixes
2. New integration test coverage for shipped-but-uncovered features
3. Production code changes
4. Verification runs (load tests, workload re-runs)

### Release branch and shipping

Per-issue commits accumulate on the release feature branch (e.g.
`feat/v0.6.6`). The branch ships as a single PR via `/ship`.
"Doing this as v0.6.X" claims the slot but does not ship — commits
stay on the branch until the user gives the explicit ship verb.

## Versioning

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Single source of truth: `pyproject.toml` (`[project] version = "X.Y.Z"`).
Referenced from `src/ollama_marshal/__init__.py` (`__version__`) and
CHANGELOG.md release headings. All three must agree at release time.

**Pre-1.0 (current state):** the public API is not yet stable. Minor
bumps may include breaking changes.

| Bump | Pre-1.0 (0.X.Y) | Post-1.0 (X.Y.Z) |
|---|---|---|
| Major (X.0.0) | Reserved for the 1.0 stable-API commitment | Breaking change to public API, config schema, or CLI |
| Minor (0.X.0 / X.Y.0) | New feature, may include breaking changes | New backwards-compatible feature or notable behavior change |
| Patch (0.0.X / X.Y.Z) | Backwards-compatible bug fix or doc-only change | Backwards-compatible bug fix or doc-only change |

**1.0.0 commitment.** Cutting 1.0.0 means the public Python API
(`from ollama_marshal import ...`), the HTTP endpoint paths and shapes,
the YAML config schema, and the CLI flags are stable. Breaking any of
those post-1.0 requires a major-version bump and a deprecation cycle.

**Per-PR workflow.** Don't bump the version in feature PRs. Add your
change under `## [Unreleased]` in CHANGELOG.md (Added / Changed /
Deprecated / Removed / Fixed / Security). Version bumps happen at
release time: `[Unreleased]` moves to `## [X.Y.Z] - YYYY-MM-DD`,
`pyproject.toml` and `__init__.py` get updated in the same release
commit, and the commit gets tagged `vX.Y.Z`.

**What counts as breaking:**
- Removing or renaming a public function, class, or method
- Removing or renaming a config field, CLI flag, or env var
- Changing the YAML config schema in backwards-incompatible ways
- Changing HTTP endpoint paths, request shapes, or response shapes
- Raising the minimum Python version
- Removing a previously-stable behavior users depended on

## Bright-line Bug Patterns

The local `/review` skill flags these aggressively (the
`anthropics/claude-code-action@v1` PR-review path was removed in
v0.5.0 — see "Code Review — Local Only" above). These are
project-specific anti-patterns we want caught every time, no
false-positive concerns:

1. **Mocking internal logic** — tests must not `unittest.mock.patch`
   internal `ollama_marshal.*` functions, classes, or methods (the
   logic being tested). External dependencies imported into a module
   ARE legitimate patch targets — Python's mock system requires patching
   at the import location, so `patch("ollama_marshal.lifecycle.httpx.
   AsyncClient", ...)` is the *correct* way to mock the httpx boundary.
   Same for `psutil`, `os.environ`, etc. Use shared fixtures in
   `tests/conftest.py` for Ollama HTTP responses where possible.
2. **Endpoint not registered** — new `/api/*` or `/v1/*` route handler
   in `server.py` without a corresponding row in "Endpoint Routing
   Rules" above.
3. **Config field not exposed** — new field in a `config.py` Pydantic
   model without an annotated entry in `marshal.example.yaml`.
4. **Missing CHANGELOG entry** — user-visible change (feature, fix,
   config addition, breaking change, deprecation) without an entry
   under `## [Unreleased]` in CHANGELOG.md.
5. **Hardcoded values** — port numbers, hostnames, timeouts, retry
   counts, queue sizes, file paths. All come from config.
6. **`time.sleep()` in async code** — must be `await asyncio.sleep()`.
7. **Bare exception handlers** — `except:` or `except Exception:`
   without re-raise OR structured log entry.
8. **`print()` calls** — only `structlog` is allowed for output.
9. **f-strings in log messages** — `logger.info(f"failed: {x}")` is
   wrong; use `logger.info("event.failed", reason=x)`.
10. **New module without tests** — every new file in
    `src/ollama_marshal/` needs a corresponding `tests/test_<name>.py`.
11. **Forgotten `keep_alive`** — every new code path that proxies a
    request to Ollama must override `keep_alive` (prevents
    auto-eviction).
12. **Doc drift** — code change that contradicts an existing statement
    in README.md, CLAUDE.md, or marshal.example.yaml without
    updating the doc in the same PR.
13. **Version drift** — at release time, `pyproject.toml`,
    `src/ollama_marshal/__init__.py`, and the latest `## [X.Y.Z]`
    heading in CHANGELOG.md must all agree. Mismatches are blockers.
14. **Wrong bump for the change** — a release PR that adds a feature
    must be at least a minor bump; a release PR that breaks public
    API/CLI/config must be a major bump (post-1.0) or call out the
    break clearly (pre-1.0). See "Versioning" above.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. The
skill has multi-step workflows, checklists, and quality gates that produce better
results than an ad-hoc answer. When in doubt, invoke the skill. A false positive is
cheaper than a false negative.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke /office-hours
- Strategy, scope, "think bigger", "what should we build" → invoke /plan-ceo-review
- Architecture, "does this design make sense" → invoke /plan-eng-review
- Design system, brand, "how should this look" → invoke /design-consultation
- Design review of a plan → invoke /plan-design-review
- Developer experience of a plan → invoke /plan-devex-review
- "Review everything", full review pipeline → invoke /autoplan
- Bugs, errors, "why is this broken", "wtf", "this doesn't work" → invoke /investigate
- Test the site, find bugs, "does this work" → invoke /qa (or /qa-only for report only)
- Code review, check the diff, "look at my changes" → invoke /review
- Visual polish, design audit, "this looks off" → invoke /design-review
- Developer experience audit, try onboarding → invoke /devex-review
- Ship, deploy, create a PR, "send it" → invoke /ship
- Merge + deploy + verify → invoke /land-and-deploy
- Configure deployment → invoke /setup-deploy
- Post-deploy monitoring → invoke /canary
- Update docs after shipping → invoke /document-release
- Weekly retro, "how'd we do" → invoke /retro
- Second opinion, codex review → invoke /codex
- Safety mode, careful mode, lock it down → invoke /careful or /guard
- Restrict edits to a directory → invoke /freeze or /unfreeze
- Upgrade gstack → invoke /gstack-upgrade
- Save progress, "save my work" → invoke /context-save
- Resume, restore, "where was I" → invoke /context-restore
- Security audit, OWASP, "is this secure" → invoke /cso
- Make a PDF, document, publication → invoke /make-pdf
- Launch real browser for QA → invoke /open-gstack-browser
- Import cookies for authenticated testing → invoke /setup-browser-cookies
- Performance regression, page speed, benchmarks → invoke /benchmark
- Review what gstack has learned → invoke /learn
- Tune question sensitivity → invoke /plan-tune
- Code quality dashboard → invoke /health
