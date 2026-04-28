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
src/ollama_marshal/    — package source (10 modules)
tests/                 — pytest unit + integration tests
```

Modules in dependency order: config → queue → registry → memory → lifecycle
→ scheduler → stream → openai_compat → server → cli

## Development Commands

```bash
make install-dev    # install editable + dev deps + pre-commit hooks
make test           # pytest with 95% coverage gate
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy --strict src/
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

Pass-through (no scheduling): /api/tags, /api/ps, /api/show, /api/version,
/api/pull, /api/delete, /api/copy

Local (served by proxy): /api/marshal/status, /status (alias)

## Scheduling Algorithm

1. FIFO baseline — respect arrival order
2. Bin-pack — fill VRAM by loading smaller models that fit alongside current
3. Skip limit — per-request counter; after max_skips, force-load the model
4. Eviction — least disruptive: fewest pending, lowest priority, oldest
5. Priority — normal (drain-then-evict) vs critical (can preempt)
6. Immediate — if model already loaded, forward without queuing

## Testing Rules

- Unit tests: mock Ollama responses via httpx, target 95%+ coverage
- Integration tests: @pytest.mark.integration, require running Ollama
- Never mock internal modules — only mock the Ollama HTTP boundary
- conftest.py has shared fixtures for all Ollama API responses
- Use `pytest` fixtures over `setUp`/`tearDown`; `@pytest.mark.parametrize`
  over loops; `pytest-asyncio` for async tests

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

The Claude PR review action flags these aggressively — these are
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
