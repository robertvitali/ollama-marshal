# ollama-marshal — Boot Resilience, Graceful Drain, and Dry-Run Program-ID Convention

- **Date:** 2026-06-25
- **Repo:** `ollama-marshal` (currently `feat/v0.6.7`, version 0.6.6)
- **Status:** REVISED after review (codex + OMC fan-out, 2026-06-25). For re-review. No code written yet; Asana tasks updated only on approval.
- **Spec role:** Keystone. The other 5 fleet repos route inference through marshal at `127.0.0.1:11435`; their resumability is moot if marshal is down or drops work across the weekly Saturday reboot. This spec makes marshal reliably come back after the reboot, drain cleanly on a graceful stop, expose a readiness + re-submit contract, and defines the marshal side of the fleet-wide **dry-run program-id** convention.

> **Review note (2026-06-25):** v1 of this spec was reviewed by `/codex review` + the OMC fan-out (code-reviewer, security-reviewer, critic). The review found v1's boot-autostart premise was factually wrong about this machine, that the chosen LaunchDaemon was the wrong artifact, that `Priority.LOW` and the metrics work were under-scoped, and several security/operational gaps. All findings were verified against live machine state and are folded in below. §10 records the provenance.

---

## 1. Context & problem

This Mac Studio auto-restarts every **Saturday 02:00** via a `personal-machines` LaunchDaemon (`system/com.robertvitali.weekly-restart.plist` + `weekly-restart.sh`): it curls marshal `/status`, **skips the week if `queue.total_pending > 0`**, else `pkill -TERM`s marshal (waits up to **60s** for the process to disappear via `pgrep`), runs `softwareupdate -i -a -R`, then `shutdown -r now`. The machine can also lose power unexpectedly.

**Verified machine state (this is what the spec must actually work against):**
- **Auto-login is ON** (`autoLoginUser = robadmin`), **FileVault OFF**. After the unattended reboot the Mac auto-logs-in, so **gui-domain LaunchAgents start without a login screen**.
- A marshal **LaunchAgent already exists and runs**: `~/Library/LaunchAgents/com.ollama-marshal.plist` (loaded, PID seen in `launchctl list` as `com.ollama-marshal`), `RunAtLoad=true`, `KeepAlive=true` (boolean), `ProgramArguments` → the **dev venv** `/Users/robadmin/workspace/ollama-marshal/.venv/bin/ollama-marshal`. It is hand-installed, not version-controlled, points at a deletable dev checkout, and uses the boolean-KeepAlive footgun (below).
- **Ollama runs only as gui LaunchAgents** (`com.user.ollama-serve`, `-q4`, `-q8`, plus `com.user.restart-ollama-on-upgrade`). Nothing starts Ollama as a boot daemon.
- `~/.ollama-marshal/marshal.out.log` is **270 MB**, unrotated.

**The real gaps (re-grounded after review):**
- **(a) The autostart artifact is hand-installed, fragile, and footgunned — not "absent".** It is not in version control, points at the dev venv (which can be `git clean`ed/moved), and uses **`KeepAlive=true` (boolean)**. The fleet's own `com.colima.daemon.plist` documents this exact footgun: *"KeepAlive must be the dict form, NOT the boolean form … the dict with `SuccessfulExit=false` means restart only on crash, not on clean exit."* With boolean KeepAlive, marshal's clean drain-and-exit on SIGTERM is treated as a crash and **immediately respawned** — so `weekly-restart.sh`'s `pgrep` still finds it after `pkill -TERM`, hits "didn't exit after 60s", and the **reboot is skipped every week**. This silently defeats the keystone goal.
- **(b) Graceful drain is queue-only and uncoordinated.** `uvicorn.run` installs uvicorn's SIGTERM/SIGINT handlers, which DO trigger the lifespan `finally` drain (`server.py:196-226`) — so a SIGTERM stop drains. But the drain waits only on the **queue** emptying (`_queues.total_pending()`), not on the scheduler's **in-flight** forwards (`_scheduler._in_flight_count`), so an in-flight Ollama generation can be cut at `drain_timeout`. Default `drain_timeout=30s` (`config.py:416`) is uncoordinated with the restart's 60s budget, and `uvicorn.run` sets no `timeout_graceful_shutdown` (`cli.py:130-135`), so the relationship is unpinned.
- **(c) Queue + in-flight are in-memory; no readiness endpoint; no client resume contract.** `ModelQueues` is an in-memory `defaultdict(deque)` (`queue.py:135`); each `RequestEnvelope` holds an `asyncio.Event` + the awaiting client's live connection (`queue.py:51`; client blocks at `server.py:1217/1315`). On crash the queue + in-flight vanish. The only liveness signal is the heavy `/status`; there is no cheap readiness endpoint and no documented re-submit contract.
- **(d) Dry-run vs prod are indistinguishable.** Consumers run dry-run and prod through the same marshal with the same `X-Program-ID`, so dry-run load can't be separated or deprioritized.

---

## 2. Goals / non-goals

**Goals**
1. Marshal reliably restarts after the weekly reboot with no human action, via a **version-controlled, hardened LaunchAgent** that *replaces* the hand-installed one and does not defeat the restart (correct KeepAlive form).
2. A graceful (SIGTERM) stop drains queue **and** in-flight within a bounded timeout coordinated with the restart budget and uvicorn's graceful-shutdown timeout, then unloads marshal-owned models — verified by a real-subprocess SIGTERM test.
3. A cheap **readiness endpoint** (`/healthz`) reporting marshal-process readiness, with the marshal-vs-Ollama distinction made explicit.
4. An explicit **re-submit / no-cross-crash-durability** contract.
5. A defined **dry-run program-id convention** marshal recognizes for observability **and** priority-demotion (`dryrun-*` yields to prod), implemented through every dispatch/eviction site that actually decides scheduling — not just an enum member.
6. **Raised test coverage** on touched + previously-untested paths, holding the 95% gate.
7. Security preconditions enforced fail-closed by the installer (file ownership/permissions, no XML/command injection, no secrets in the plist).

**Non-goals / explicitly rejected**
- **LaunchDaemon for marshal — rejected (D1-redux).** Auto-login makes a LaunchAgent reliable post-reboot; a daemon would start before Ollama's agents (false-healthy race) and add a root-runs-user-binary privesc surface.
- **Daemonizing Ollama for boot — out of scope.** Auto-login covers the weekly case. A non-auto-login / major-OS-Setup-Assistant scenario is a separate cross-repo effort; noted as a risk in §7, not built here.
- **Durable on-disk request queue — rejected.** The awaiting client connection + `done_event` (`queue.py:51`) are unserializable; marshal could never deliver a replayed result to the original (now-disconnected) client. Consumers re-derive their work; clients re-submit.
- Catching SIGKILL / power-loss to drain (uncatchable).
- Changing routing/tier/instance selection (program-id plays no part there; leave it).

---

## 3. Current state (grounded, file:line — corrected in review)

| Area | Today | Evidence |
|---|---|---|
| Server launch | foreground `uvicorn.run(app, host, port, log_level)`, single-process, **no `timeout_graceful_shutdown`** | `cli.py:130-135` |
| Graceful drain | lifespan `finally`: if `DRAIN`, wait `total_pending()==0` up to `drain_timeout`, then `scheduler.stop()`, stop polls, unload marshal-owned models | `server.py:196-226` |
| In-flight counter | `_in_flight_count` inc/dec around `_forward_single`; `_wait_for_drain` polls it; **not coupled to admin-pause** (safe to reuse in lifespan) | `scheduler.py:532,669,2050-2054` |
| Drain config | `shutdown.mode=DRAIN`, `drain_timeout=30`, `unload_models=True` | `config.py:412-423` |
| Boot autostart | **hand-installed LaunchAgent** (dev venv, `KeepAlive=true` boolean), not version-controlled | `~/Library/LaunchAgents/com.ollama-marshal.plist` (live) |
| KeepAlive correct form | fleet precedent: dict `{SuccessfulExit=false}` | `com.colima.daemon.plist:42-50` |
| Console entrypoint | `ollama-marshal = ollama_marshal.cli:app`; `start` subcommand | `pyproject.toml:72-73`, `cli.py:75-135` |
| Status/health | `/status`+`/api/marshal/status` heavy payload (route `server.py:480`, payload built `server.py:369`); **no `/healthz`**; no `_ready` flag (`_started_at` is set at lifespan entry, before startup completes) | `server.py:369,480,49,60` |
| Program-id read | `X-Program-ID` → `_normalize_program_id` (keeps `[A-Za-z0-9_.-]`, ≤64; hyphen OK) | `server.py:742-754,1152,1262` |
| Program-id priority | **exact-match** `get_program_config` → fallback `default`→`NORMAL`; `ProgramConfig` only has `priority`; **no prefix/glob anywhere** | `config.py:789-793,400-406` |
| Priority type | `StrEnum {NORMAL, CRITICAL}`, compared only by `==`/`!=` (no ordinal); eviction scorer string-maps `{"critical":1,"normal":0}` | `config.py:20-24`; `scheduler.py:1243,1343`; `memory.py:679` |
| Priority map | `_priority_map_from_pending` emits only `"critical"`/`"normal"` | `scheduler.py:1227-1249` |
| Dispatch | FIFO + bin-pack by arrival; **no priority-ordered dequeue**; only CRITICAL has special handling | `scheduler.py:1143,1383`; `queue.py:276` |
| Per-program metrics | **none** — `SchedulerMetrics` is global counters; per-program data exists only in `/status` + audit | `scheduler.py:275-339`; `server.py:422`; `audit.py:138` |
| Metrics schema | `_METRICS_SCHEMA_VERSION=1`; `from_json_dict` raises on mismatch; live `metrics.json` present | `scheduler.py:272,349-354` |
| Burst-hints / ctx profiles | keyed on **raw** `program_id` (full `dryrun-…`), exact | `server.py:780,894,1179,1185` |
| Existing drain test | **mocked** drain-mode test exists; only real-subprocess SIGTERM is missing | `tests/test_server.py:1677` |
| Test suite | `make test` (unit, 95% gate) + `make test-integration` (live Ollama) | `Makefile:8-15` |

---

## 4. Design overview

Four workstreams. **WS3 `/healthz` lands before the installer** (the installer verifies via it).

### WS1 — Replace the hand-installed LaunchAgent with a hardened, version-controlled one (D1-redux: LaunchAgent)
- Add `examples/com.ollama-marshal.plist` (+ ship via `dist/launchd/`) as a **gui LaunchAgent**:
  - `Label = com.ollama-marshal`.
  - **`KeepAlive` = dict `{SuccessfulExit = false}`** (restart on crash, NOT on clean drain-exit). This is the single most load-bearing fix — boolean KeepAlive defeats the weekly reboot.
  - `RunAtLoad = true`; add `ThrottleInterval` (≥30) to bound crash-loops on a bad config/port-bind.
  - `ProgramArguments` → a **stable, non-dev** install path of the `ollama-marshal` console script + `start --config /Users/robadmin/.ollama-marshal/marshal.yaml` (absolute). Canonical install location TBD in task (NOT the editable dev `.venv`).
  - `EnvironmentVariables`: set `PATH` (and `HOME` if marshal relies on `~` expansion at agent start) — a launchd job has a minimal environment. No tokens, no secrets.
  - `StandardOutPath`/`StandardErrorPath` under `~/.ollama-marshal/` (see WS1 log-rotation).
- Add `scripts/install-launchd.sh` / `uninstall-launchd.sh`, **idempotent against the CURRENT machine state**:
  - **Migrate the existing hand-installed agent:** detect + `launchctl bootout gui/$UID/com.ollama-marshal` before bootstrapping the new one (avoid two same-labeled jobs racing `:11435`). Document the agent→hardened-agent migration in README.
  - Build the plist with **PlistBuddy/`plutil`**, never shell string-interpolation into XML; `plutil -lint` before install; allowlist-validate templated inputs (username, prefix); `install -m 0644` from a root-only `mktemp -d`; `umask 022`.
  - Verify post-install via `curl -fsS http://127.0.0.1:11435/healthz` (WS3).
- **Log rotation:** add a `newsyslog.d` entry (or marshal-side size cap) for `marshal.out.log`/`marshal.err.log` — it is 270 MB today; an unattended agent will keep growing it.

### WS2 — Graceful drain hardening (in-flight-aware, coordinated, tested)
- Extend the lifespan drain (`server.py:196-226`): after the queue empties, also await `_scheduler` in-flight `== 0` (reuse `_wait_for_drain`/`in_flight_count()`, `scheduler.py:669,692` — verified not coupled to admin-pause), all under one `drain_timeout` deadline, keeping `scheduler.stop()` after the combined wait.
- Set `drain_timeout` default **45s** and **pin `uvicorn.run(..., timeout_graceful_shutdown=<≥ drain_timeout + headroom>)`** so uvicorn doesn't cancel the lifespan mid-drain. Invariant to document + test in both repos: `uvicorn graceful ≥ drain_timeout (45s) ≤ restart drain budget (60s)`.
- **Cross-repo (spec #2):** `weekly-restart.sh` must `launchctl bootout`/disable the agent **before** `pkill` (else KeepAlive respawns it during drain); and its skip check should consider in-flight (or accept the 45s cut explicitly) and **exclude LOW/dry-run pending** so a starved dry-run can't block the reboot forever (M5).
- Add a real-subprocess **SIGTERM drain integration test** (the existing test at `tests/test_server.py:1677` is mocked; only this is missing): enqueue → SIGTERM the real process → assert in-flight drained, models unloaded, clean exit, within budget, and the drain wasn't truncated by uvicorn.

### WS3 — Readiness endpoint + re-submit contract
- Add `GET /healthz`: a **new `_ready` flag set True immediately before the lifespan `yield`, reset in `finally`** (NOT `_started_at`, which is set at entry). `200 {"status":"ok","version":...}` when ready, `503` during startup/drain. **Liveness/version only** — no model names, queue depth, or system snapshot (keep `/status`'s disclosure out of the unauthenticated cheap endpoint).
- **Marshal-vs-Ollama distinction (M4):** `/healthz` = marshal process ready. It must NOT imply Ollama is reachable (`registry.initialize` (`registry.py:343`) reaches the `httpx.HTTPError` swallow at `registry.py:669` and continues startup). For "can actually serve", expose a separate dependency check — recommend a two-field readiness (`/readyz` → `{marshal: ok, ollama: ok|down}` with a cheap `/api/version` probe) so downstream "is marshal back AND able to serve?" is answerable. Drop the bare `/readyz` alias-of-`/healthz` idea; if `/readyz` exists it carries dependency state.
- **Additive `/status` priority-pending field (for the restart-skip):** today `/status` exposes only `queue.total_pending` + `queue.by_model` (`server.py:454`) — it cannot distinguish prod-pending from dry-run-pending, so spec #2's "exclude LOW/dry-run from the skip" cannot be computed. Add a **strictly additive** `queue.by_priority` (pending counts per priority, incl. unloaded queued models) — recommend a `blocking_pending` = NORMAL+CRITICAL convenience count. **Do not mutate** existing fields (the `loaded_models[].programs` shape is `list[str]`, `server.py:422`); all dry-run/priority surfacing is additive. Depends on `Priority.LOW` existing (WS4).
- Document the **re-submit contract** (README + `docs/`): marshal makes no cross-crash durability promise; on a dropped connection or readiness flap, clients re-submit idempotently. State the **exact failure signal** a client sees when drain elapses (connection close vs. error body) so clients can detect the drop. Durable queue rejected (see §2).

### WS4 — Dry-run program-id convention (reserved prefix + real priority threading)
Convention: **dry-run inference sends `X-Program-ID: dryrun-<base-id>`; prod sends bare `<base-id>`.** `dryrun-` is a **fleet-reserved prefix** (documented; a real program may not legitimately start with it). Prefix chosen over a separate header/suffix: hyphen-safe + ≤64 chars, left-anchored, and already isolates metrics/audit.

- **Observability (task: status + audit only):** parse a **anchored** leading `dryrun-` (`raw.startswith("dryrun-")`) into `is_dryrun` + `base_program_id`, **re-sanitizing** `base_program_id` through `_PROGRAM_ID_ALLOWED` before it lands anywhere. Surface `dryrun:true` in `/status` `programs` and a `dryrun` field in audit (both already per-program; additive). **No per-program metrics today** (`SchedulerMetrics` is global) — per-program metric buckets are a **separate task** with an explicit new structure + `_METRICS_SCHEMA_VERSION` bump (one-time lifetime-counter reset accepted), NOT folded into the observability task.
- **Priority-demotion (the real surface — task must enumerate ALL sites):**
  - `Priority.LOW` new member (`config.py:20-24`). Keep `Priority` a `StrEnum` compared by `==` only; do **not** introduce ordered `<` comparisons (StrEnum sorts alphabetically — wrong). Introduce an explicit **rank map** as the single ordering source.
  - `get_program_config` (`config.py:789-793`): lookup order = **exact `programs[id]` config first** (so an explicit `dryrun-foo: {priority: normal}` override wins), **then** the `dryrun-` prefix rule → `LOW`, **then** `default`→`NORMAL`. (v1 had prefix-before-exact, which broke the "unless explicitly configured" override — corrected.) Localize the prefix rule to `get_program_config` only (burst-hints/context-profiles keep the raw id, staying exact — but **document the consequence**: a dry-run program does NOT inherit the prod program's burst-hint namespace or context profile floor/ceiling).
  - `_priority_map_from_pending` (`scheduler.py:1227`): add a `"low"` branch (today emits only `"critical"`/`"normal"`).
  - `memory.py:679` eviction scorer: extend `priority_order` to `{"critical":2,"normal":1,"low":0}` (today binary — without this, LOW silently maps to normal = no demotion).
  - **Dispatch ordering (the actual "yield" — does not exist today):** `_forward_loaded_model_requests` (`scheduler.py:1143`) dispatches every loaded model's queue per tick in dict order; `_bin_pack_models` (`scheduler.py:1383`) selects by arrival. Add priority-aware dispatch so a LOW request defers when a NORMAL/CRITICAL is pending. **Define "yield" explicitly:** LOW defers while any NORMAL/CRITICAL is pending **across models** (true GPU yield), not just same-model — confirm during implementation.
  - **Bypass-token × LOW:** a valid `X-Marshal-Test-Bypass` (`server.py:736`) forces dispatch regardless of LOW — document + test.
- Stale anchors corrected: `_priority_map_from_pending` def is `scheduler.py:1227` (not 1242); `_handle_critical_preemption` def `1338` (not 1342).

---

## 5. Security preconditions (installer enforces, fail-closed)

Even as a LaunchAgent (runs as robadmin, no root), the config + entrypoint must be tamper-resistant since they're auto-executed unattended:
- `~/.ollama-marshal/marshal.yaml` mode **600** (it controls `proxy.host` bind, the Ollama forward target, admin/bypass tokens, audit/metrics paths). Installer refuses to start if it is group/world-writable. Validate `proxy.host` stays loopback unless an explicit opt-in.
- `ProgramArguments[0]` target binary + parent dirs **not** group/world-writable; installer refuses otherwise.
- `chmod 700 ~/.ollama-marshal`; ensure log paths' parent is owned by robadmin and not a symlink (pre-placed-symlink hijack).
- Installer: PlistBuddy/`plutil` (no XML string-interpolation), allowlist-validated inputs, atomic `install -m 0644`, root-only staging dir, `umask 022`. `shellcheck` in CI + a test that a hostile venv path (`a"/></string>`) is rejected.
- Shipped plist contains **no** tokens/secrets and no `EnvironmentVariables` secrets (visible via `launchctl print`/`ps -E`).
- Keep `/healthz` liveness-only; do not widen the unauthenticated `/status` with new dry-run breakdowns without noting any local user reads it.
- `dryrun-` prefix is a **best-effort observability + soft-priority** convention on an **unauthenticated** header — document that it is NOT an enforceable trust boundary (a client can omit the prefix to ride NORMAL); enforceable low-priority requires a configured `programs:` entry. Confirm `burst_hint_max_live`/aggregate caps still bound a client flooding `dryrun-<random>` ids.

---

## 6. Testing & verification

Canonical suite (HARD-GATE 6): `make test` (unit, 95% gate) + `make test-integration` (live Ollama). New coverage:
- WS1: shellcheck + idempotency test (re-run no-ops; runs cleanly with the OLD agent present → migrates it); hostile-path rejection test; `plutil -lint` in CI.
- WS2: in-flight-aware drain unit test; real-subprocess SIGTERM drain integration test (asserts in-flight waited, models unloaded, clean exit, not truncated by uvicorn); test asserting `drain_timeout < restart budget`.
- WS3: `/healthz` `200`-ready / `503`-startup; `/readyz` reflects Ollama-down.
- WS4: anchored prefix parse + `base_program_id` re-sanitization; exact-config-override-beats-prefix; `_priority_map_from_pending` `"low"`; `memory.py` eviction ranks LOW below NORMAL; **dispatch: a LOW request yields to a pending NORMAL** (the load-bearing test); bypass-token forces LOW dispatch.
- Add a `## Canonical test suite (HARD-GATE 6 reference)` subsection to `CLAUDE.md`.

---

## 7. Cross-repo sequencing & risks

- **personal-machines (spec #2)** owns: deploying the hardened agent via chezmoi; updating `weekly-restart.sh` to `bootout`/disable marshal **before** `pkill` (KeepAlive-dict still respawns on crash, so signal-only is unsafe), to probe `/healthz` instead of heavy `/status`, and to exclude LOW/dry-run pending from the skip check (consuming the additive `queue.by_priority`/`blocking_pending` field WS3 adds to `/status` — without it the skip can't tell prod-pending from dry-run-pending); finalizing the 45s/60s drain pairing.
- **All 5 consumer specs** depend on WS3 (`/healthz`/`/readyz` to wait for marshal post-reboot) + WS4 (each emits `dryrun-<id>` in dry-run mode — a task in each consumer spec, linking to the convention doc here).
- **Risk (accepted, noted not built):** auto-login is load-bearing for the LaunchAgent choice. A major-OS upgrade that lands at Setup Assistant, or auto-login being disabled, would leave marshal **and** Ollama down post-reboot (both are agents). If that scenario becomes real, daemonizing both marshal + Ollama is a separate cross-repo effort.

---

## 8. Task breakdown (revised, reordered → Asana parent + subtasks)

Parent: **"ollama-marshal: boot resilience, graceful drain, and dry-run program-id"**

1. **Add `GET /healthz` (+ `/readyz` with Ollama dependency state)** — new `_ready` flag set before `yield`/reset in `finally`; `/healthz` liveness+version only; `/readyz` does a cheap Ollama `/api/version` probe. Unit tests. (First — installer + consumers depend on it.)
2. **Ship hardened LaunchAgent `examples/com.ollama-marshal.plist`** — `KeepAlive` dict `{SuccessfulExit=false}`, `ThrottleInterval`, stable non-dev `ProgramArguments` path, minimal env (PATH/HOME), log paths. `plutil -lint`.
3. **Idempotent `install/uninstall-launchd.sh` that MIGRATES the existing hand-installed agent** — `bootout` the live `gui/$UID/com.ollama-marshal` first; PlistBuddy/`plutil` build; allowlist-validate inputs; atomic install; verify via `/healthz`; shellcheck + hostile-path test.
4. **Security preconditions (fail-closed)** — enforce `marshal.yaml` 600, non-world-writable target binary, `chmod 700 ~/.ollama-marshal`, `umask 022`, no secrets in plist; installer refuses on violation. (Folded into task 3's script; separate acceptance for the checks.)
5. **Log rotation for marshal logs** — `newsyslog.d` entry (or size cap); `marshal.out.log` is 270 MB today.
6. **Make lifespan drain in-flight-aware** — await `in_flight_count()==0` after queue drain, under `drain_timeout`; unit test.
7. **Set `drain_timeout=45s` + pin `uvicorn timeout_graceful_shutdown`** — document the `uvicorn ≥ 45s ≤ 60s` invariant.
8. **Real-subprocess SIGTERM drain integration test** — assert in-flight drained, models unloaded, clean exit, not truncated by uvicorn, within budget.
9. **Document re-submit / no-cross-crash-durability contract** — incl. the exact client-visible failure signal; record durable-queue rejection.
10. **Define + document the reserved `dryrun-<id>` convention** — README/`docs/`; the fleet reference the 5 consumer specs link to.
11. **Dry-run observability (status + audit only)** — anchored prefix parse → `is_dryrun` + re-sanitized `base_program_id`; surface in `/status` + audit; document dry-run does NOT inherit prod burst/context profiles. Unit tests.
12. **`Priority.LOW` + lookup ordering** — add enum member + explicit rank map (no StrEnum ordered compare); `get_program_config` order = exact-config → `dryrun-` prefix→LOW → default→NORMAL; unit test that a configured `dryrun-foo: {priority: normal}` override wins.
13. **Thread LOW through eviction + priority map** — `_priority_map_from_pending` `"low"` branch (`scheduler.py:1227`); `memory.py:679` `priority_order` `{"critical":2,"normal":1,"low":0}`; unit test LOW ranks below NORMAL for eviction.
14. **Priority-aware dispatch so dry-run yields to prod (the real "yield")** — make `_forward_loaded_model_requests`/bin-pack defer LOW when a NORMAL/CRITICAL is pending across models; define + test "LOW yields to a pending NORMAL"; bypass-token forces dispatch.
15. **(Conditional) Per-program metrics buckets** — only if we want dry-run-vs-prod *metrics* (not just status/audit): new per-program structure + `_METRICS_SCHEMA_VERSION` bump (one-time counter reset accepted). Defer unless wanted.
16. **Coverage sweep + `CLAUDE.md` HARD-GATE 6 anchor** — cover the previously-untested lifespan-drain path, `/healthz`, dispatch-yield; hold 95%; `make test` + `make test-integration` green; add the canonical-suite subsection.

Order: `/healthz` (1) → agent + installer + security + logrotate (2–5) → drain hardening (6–8) → dry-run convention/observability/priority/dispatch (10–14) → metrics (15, conditional) → coverage/anchor (16). Cross-repo restart-script coordination lives in spec #2.

---

## 9. Decisions (resolved in review 2026-06-25)

- **D1-redux — autostart artifact → hardened, version-controlled LaunchAgent** (reverses v1's LaunchDaemon). Rationale: auto-login ON + FileVault OFF ⇒ agents start post-reboot; Ollama is itself agents (match it, avoid the boot-race); avoids root-daemon privesc; an agent already works and just needs hardening + version control. **KeepAlive must be dict `{SuccessfulExit=false}`** regardless of artifact.
- **D2 — dry-run → Full** (observability + real Priority.LOW yield-to-prod), now correctly threaded through dispatch/eviction, not just an enum.
- **D3 — `drain_timeout` → 45s**, pinned against uvicorn's graceful timeout and the 60s restart budget.
- **D4 — specs → per-repo `docs/specs/`**.

---

## 10. Review provenance (2026-06-25)

- `/codex review` (ChatGPT auth): verified file:line claims; flagged Priority.LOW insufficiency (FIFO dispatch), `/healthz`-vs-Ollama, prefix-before-exact ordering, `dryrun-` collision, raw-vs-base data model, task-order contradiction, stale "no test" claim.
- OMC `code-reviewer` (REQUEST CHANGES): WS1/2/3 sound; WS4 dispatch machinery doesn't exist; `memory.py:679` missing from task surface; StrEnum alphabetical-order trap; dry-run loses prod burst/ctx profiles; `_ready` flag needed.
- OMC `security-reviewer` (HIGH, 4×P1/6×P2): config/binary write-perms, installer XML/command-injection, `dryrun-` as unauthenticated trust boundary + bypass-token interaction, `/healthz` disclosure, KeepAlive crash-loop, log symlink, umask/secrets.
- OMC `critic` (REVISE, adversarial): **verified live machine state** — existing hand-installed LaunchAgent (contradicted v1 premise), `KeepAlive` boolean footgun vs colima precedent, Ollama-not-at-boot, 270 MB log, metrics schema-version, restart-skip × never-draining-LOW.

All folded in. Open machine-state claims were re-verified directly (auto-login, FileVault, existing plist, colima comment, log size, restart-script mechanism) before revision.
