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
