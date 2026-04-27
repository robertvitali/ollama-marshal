# ollama-marshal

[![CI](https://github.com/robertvitali/ollama-marshal/actions/workflows/ci.yml/badge.svg)](https://github.com/robertvitali/ollama-marshal/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Model-aware scheduling proxy for Ollama** — eliminates model thrashing
with intelligent scheduling.

```
program A ──┐    ┌──────────────────┐
program B ──┼───>│  ollama-marshal   │───> Ollama :11434
program C ──┘    │  (:11435)         │
                 └──────────────────┘
```

## The Problem

When multiple programs on the same machine use different Ollama models, the
default FIFO queue with LRU eviction causes **model thrashing** — constant
loading and unloading of models.

If requests arrive as `[modelA, modelB, modelA, modelB]`, Ollama performs
**4 model load cycles**. An intelligent proxy reorders to
`[modelA, modelA, modelB, modelB]` — only **2 loads**, and can go further
by fitting both models in memory simultaneously for **0 unnecessary loads**.

## Why this proxy exists

Ollama serves requests with a simple FIFO queue. When VRAM pressure
forces a model swap, it evicts via LRU. Neither decision considers
what's queued behind the current request, so models load and unload
reactively as different programs hit the proxy.

The [QLM paper](https://dl.acm.org/doi/10.1145/3698038.3698523) (Patke
et al., ACM SoCC 2024) introduces a **global scheduler** for LLM
serving that orchestrates request pulling, eviction, load balancing,
and model swapping together, instead of treating each as an
independent decision. They report 40–90% SLO improvement and 20–400%
throughput gains in their evaluation.

ollama-marshal applies that *global-scheduler* insight to a single
Ollama instance. We don't implement QLM's full machinery (no Request
Waiting Time Estimator), but we do replace Ollama's reactive
per-request behavior with batch-aware decisions:

| Decision | Ollama natively | ollama-marshal |
|---|---|---|
| Order of dispatch | FIFO arrival order | Group by already-loaded model, dispatch all at once |
| Eviction trigger | LRU on VRAM pressure | Score by pending request count + program priority |
| When to evict | Whenever a different model is requested | After draining the in-flight model's pending batch |
| Bin-packing | None | Load smaller queued models into remaining VRAM |
| Fairness | None | `max_skips` forces service after N bin-pack passes |
| Priority | All requests equal | `critical` priority can preempt loaded models |

In the `[modelA, modelB, modelA, modelB]` example above, Ollama does 4
model loads. ollama-marshal drops that to 2, or 0 if both fit in VRAM
together.

## Features

- **FIFO + bin-packing scheduler** — respects request arrival order, but
  intelligently packs VRAM by loading smaller models that fit alongside
  current ones. Configurable fairness limits prevent starvation.
- **Automatic memory management** — detects system RAM, benchmarks model
  sizes, and makes all loading/eviction decisions. No manual tuning needed.
- **Model size registry** — measures actual VRAM footprint of each model on
  first use, caches results for instant future scheduling decisions.
- **Per-program priority** — programs identify via `X-Program-ID` header.
  Configure `critical` priority for real-time tools that can preempt batch
  jobs.
- **Full API compatibility** — proxies both Ollama-native (`/api/chat`,
  `/api/generate`, `/api/embeddings`) and OpenAI-compatible endpoints
  (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`).
- **Streaming passthrough** — full support for Ollama's NDJSON streaming.
- **Drop-in transparent** — programs just change `OLLAMA_HOST` to the proxy
  port. No other code changes required.
- **Structured logging** — colored console output for development, JSON for
  production monitoring.

## Quick Start

### Install

```bash
# With pip
pip install ollama-marshal

# With uv (recommended)
uv add ollama-marshal
```

### Run

```bash
# Start with defaults (proxy on :11435, Ollama on :11434)
ollama-marshal start

# Check status
ollama-marshal status

# Point your programs at the proxy
export OLLAMA_HOST=http://localhost:11435
```

### Configure

Create a `marshal.yaml` (see
[`marshal.example.yaml`](marshal.example.yaml) for all options):

```yaml
proxy:
  port: 11435

memory:
  safety_margin: "4GB"

scheduler:
  max_skips: 3

programs:
  default:
    priority: normal
  coding-assistant:
    priority: critical

shutdown:
  mode: drain
  unload_models: true
```

All values can be overridden via environment variables:

```bash
MARSHAL_PROXY_PORT=11435
MARSHAL_MEMORY_SAFETY_MARGIN=4GB
MARSHAL_SCHEDULER_MAX_SKIPS=5
```

## How It Works

### Scheduling Algorithm

1. **FIFO baseline** — requests are ordered by arrival time
2. **Bin-packing** — VRAM is treated like a glass to fill. If the next
   request's model doesn't fit alongside what's loaded, but a smaller model
   later in the queue does, load that first
3. **Fairness ceiling** — each request tracks how many times it's been
   skipped. After `max_skips` (default: 3), it must be served next — even
   if that means evicting a loaded model
4. **Intelligent eviction** — when a model must be evicted, choose the one
   with fewest pending requests, lowest priority, and oldest last-request
   time
5. **Priority preemption** — `critical` programs can preempt: the current
   in-flight request finishes, then the critical model is loaded immediately
6. **Immediate forwarding** — if a model is already loaded, requests are
   forwarded instantly without queuing

### Proxy Control

ollama-marshal takes **full control** of model lifecycle:

- Overrides `keep_alive` on every request to prevent Ollama's auto-eviction
- Preloads models via the Ollama API
- Unloads models via `keep_alive: "0"`
- Auto-detects available RAM and model sizes

> **Important**: Remove any `OLLAMA_MAX_LOADED_MODELS` or
> `OLLAMA_NUM_PARALLEL` settings you may have set. Common locations:
> `~/.zshrc`, `~/.bashrc`, launchd plists, systemd units, Docker env vars.
> The proxy manages these decisions.

## Comparison

| Feature                         | ollama-marshal | ollamaMQ | llama-swap | LiteLLM | Raw Ollama |
|---------------------------------|:--------------:|:--------:|:----------:|:-------:|:----------:|
| Model-aware scheduling          | Yes            | No       | No         | No      | No         |
| VRAM bin-packing                | Yes            | No       | No         | No      | No         |
| Single-instance optimization    | Yes            | No       | Yes        | No      | N/A        |
| Per-program priority            | Yes            | No       | No         | Yes     | No         |
| Automatic memory management     | Yes            | No       | No         | No      | LRU only   |
| OpenAI-compatible endpoints     | Yes            | No       | No         | Yes     | Yes        |
| Multi-backend load balancing    | No             | Yes      | Yes        | Yes     | No         |
| Drop-in transparent             | Yes            | No       | Yes        | No      | N/A        |
| Streaming support               | Yes            | Yes      | Yes        | Yes     | Yes        |

**ollama-marshal** is purpose-built for the single-machine, multi-program
use case. If you need to load-balance across multiple Ollama backends, use
ollamaMQ or LiteLLM. If you need to optimize model scheduling on a single
instance, use ollama-marshal.

## API

ollama-marshal proxies the full Ollama API surface. Endpoints fall into
three categories — choose the right path for what you're doing.

### Inference (queued — go through the scheduler)

| Method | Path | What it does | Notes |
|---|---|---|---|
| POST | `/api/chat` | Ollama-native chat | Per-program priority via `X-Program-ID` header |
| POST | `/api/generate` | Ollama-native completion | Same |
| POST | `/api/embeddings` | Ollama-native embeddings | Embeddings batched concurrently |
| POST | `/v1/chat/completions` | OpenAI-compatible chat | Auto-translated to Ollama; same scheduling |
| POST | `/v1/completions` | OpenAI-compatible completion | Same |
| POST | `/v1/embeddings` | OpenAI-compatible embeddings | Same |

These requests **always** go through the scheduler. Streaming (`stream: true`)
is passed through transparently. Set `X-Program-ID: <your-app>` to identify
your program for per-program priority and metrics.

### Pass-through (read-only; no scheduling)

| Method | Path | What it does |
|---|---|---|
| GET | `/api/version` | Ollama version |
| GET | `/api/tags` | List models in Ollama |
| POST | `/api/show` | Model details (params, template, modelfile) |
| GET | `/api/ps` | Currently-loaded models (raw from Ollama) |

Allowlisted to keep the proxy boundary tight. Forwarded to Ollama unchanged.

### Blocked (destructive — manage models via Ollama directly)

| Method | Path | Why |
|---|---|---|
| POST | `/api/pull` | Download a new model — bypasses marshal |
| POST | `/api/delete` | Remove a model |
| POST | `/api/copy` | Copy a model |

These return `403 Forbidden` with a message pointing to direct Ollama use.

### Marshal-specific

| Method | Path | What it does |
|---|---|---|
| GET | `/api/marshal/status` | Marshal's runtime state — see schema below |
| GET | `/status` | Short alias for `/api/marshal/status` |

A live TUI dashboard renders this same data in real time:

```bash
ollama-marshal dashboard
```

### Interactive API docs

FastAPI auto-generates Swagger UI and the OpenAPI spec:

- `http://localhost:11435/docs` — interactive UI, try requests in-browser
- `http://localhost:11435/openapi.json` — machine-readable spec

### Status endpoint schema

```json
{
  "uptime_seconds": 3600,
  "loaded_models": [
    {"name": "llama3:latest", "size_vram": 4500000000, "pending_requests": 2}
  ],
  "memory": {
    "total": 274877906944,
    "available": 240000000000,
    "used_by_models": 4500000000
  },
  "queue": {
    "total_pending": 5,
    "by_model": {"llama3:latest": 2, "codellama:latest": 3}
  },
  "metrics": {
    "requests_served": 142,
    "model_swaps": 8,
    "evictions": 3,
    "average_wait_ms": 250
  }
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and
PR guidelines.

## License

[Apache License 2.0](LICENSE)
