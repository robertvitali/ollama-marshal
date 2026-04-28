# ollama-marshal — Dry-Run Dashboard

Validate marshal under realistic concurrent load without writing custom
tooling. Pair the existing `ollama-marshal status` CLI with `tail -f` on
the structured log, and fire scenarios with `scripts/dryrun.py`.

## The 3-pane layout

In iTerm2 (or your terminal of choice), split into three panes. From the
marshal repo root:

**Pane 1 — refreshing status snapshot:**

```bash
watch -n 1 ollama-marshal status
```

Shows loaded models with their VRAM size, memory budget (used / available
/ total), pending queue per model, and the running metrics (requests
served, model swaps, evictions, average wait time).

**Pane 2 — live event stream:**

```bash
tail -f ~/.ollama-marshal/marshal.out.log \
  | grep --line-buffered -E "scheduler\.|request_(enqueued|served|timeout|error)"
```

Filters the structured log to scheduling decisions and request lifecycle
events. Watch for:

- `scheduler.bin_pack_load` — marshal loaded a smaller model alongside
  what was already there
- `scheduler.evicting` — picked a model to unload
- `scheduler.drain_before_evict` — finished pending requests for a model
  before evicting it (normal-priority safety)
- `scheduler.critical_preemption` — a critical request bumped the queue
- `scheduler.forced_load` — a request hit `max_skips` and forced its
  model to load

**Pane 3 — fire scenarios:**

```bash
.venv/bin/python scripts/dryrun.py --help
.venv/bin/python scripts/dryrun.py parallel-all
```

Or from the marshal repo: `make dryrun-dashboard` echoes all three
commands ready to paste.

## Scenarios at a glance

`scripts/dryrun.py` ships 12 scenarios in three groups.

### Single-shot (~1-3s, fast feedback)

One request each, with the right `X-Program-ID` header for an example
program. The four placeholder programs (`chat-a`, `chat-b`, `generate-a`,
`generate-b`) simulate a deployment with two chat clients and two generate
clients — adjust to your own programs in `scripts/dryrun.py`.

```bash
python scripts/dryrun.py single chat-a
python scripts/dryrun.py single chat-b
python scripts/dryrun.py single generate-a
python scripts/dryrun.py single generate-b
```

Use these to confirm baseline routing and that marshal is up.

### Pattern scenarios (test specific marshal behaviors)

| Command | What it tests | What you should see |
|---|---|---|
| `python scripts/dryrun.py same-model-burst --k 5` | Model-affinity grouping (e.g. K-vote self-consistency). | Pane 1: pending spikes to 5 then drains. Pane 2: 5 enqueued, 5 served, **no `evicting`**. Assertion: `model_swaps` Δ = 0. |
| `python scripts/dryrun.py same-model-loop --rounds 10` | Sequential same-model loop (e.g. tool-calling rounds). | Pane 1: `requests_served` ticks up by 10. Assertion: 0 swaps, 0 evictions. |
| `python scripts/dryrun.py parallel-all` | **Headline test.** One request from each program in parallel, each with a different small model. | Pane 1: pending grows briefly across 4 models, drains in 4 batches grouped by model. Pane 2: at most a few `bin_pack_load` events. **No `request_failed`.** |
| `python scripts/dryrun.py thrash-test --rounds 10` | Alternating models — provokes bin-packing and (with `--big`) evictions. | Pane 2: visible `bin_pack_load` for each new model. With production-size models: `evicting` events with `pending=0` (drain-before-evict working). |
| `python scripts/dryrun.py priority-test` | Critical-priority preemption. **Requires** at least one program in `marshal.yaml` set to `priority: critical`. | Pane 2: `scheduler.critical_preemption` event. Critical request served before queued normal-priority work. |

### Boundary scenarios (verify proxy guarantees)

| Command | Must |
|---|---|
| `python scripts/dryrun.py passthrough allowed` | GET `/api/tags` returns 200 with the model list. |
| `python scripts/dryrun.py passthrough blocked` | POST `/api/pull` returns 403 with "not proxied" in the message. |
| `python scripts/dryrun.py bad-model` | Request for a non-existent model errors fast (under 15s), not a silent wedge. |

## Default vs production-size models

Scenarios default to small models that load fast (`qwen3.5:4b-bf16` and
similar) so you can iterate. Pass `--model NAME` for a single scenario, or
`--big` on `parallel-all` to switch to realistic-size models. Use `--big`
to actually provoke memory pressure and eviction events.

## What "marshal is working" looks like

If you fire `parallel-all` and see all 4 programs return 200, with model
swaps bounded by the number of distinct models (not 4× the number), then
inter-program scheduling is working. If `same-model-burst --k 5` completes
with **0 swaps**, model-affinity grouping is working. If `thrash-test
--big` shows `evicting` events that come AFTER `drain_before_evict`,
graceful eviction is working.

If any scenario fails its assertion or produces unexpected dashboard
behavior, that's a real marshal bug — log it.
