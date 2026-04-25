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

## Config Precedence

CLI flags > env vars (MARSHAL_*) > marshal.yaml > defaults

Config discovery: --config flag → MARSHAL_CONFIG env → ./marshal.yaml
→ ~/.ollama-marshal/marshal.yaml → built-in defaults

## Endpoint Routing Rules

Queued (through scheduler): /api/chat, /api/generate, /api/embeddings,
/v1/chat/completions, /v1/completions, /v1/embeddings

Pass-through (no scheduling): /api/tags, /api/ps, /api/show, /api/version,
/api/pull, /api/delete, /api/copy

Local (served by proxy): /api/marshal/status

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

## Safety Rules

- Never commit secrets (.env, API keys, credentials)
- Never hardcode ports, hosts, timeouts, or thresholds
- Override keep_alive on EVERY proxied request (prevents Ollama auto-eviction)
- Drain-before-evict for normal priority (no mid-batch model unloading)
- Graceful shutdown must respect the configured mode (drain vs immediate)

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
