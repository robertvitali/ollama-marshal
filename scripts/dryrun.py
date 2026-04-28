"""Dry-run load harness for ollama-marshal.

Fires realistic concurrent load against a running marshal at :11435 and
asserts expected scheduling behavior. Pair with the 3-pane dashboard
(see scripts/README.md) to watch the queue and memory in real time.

Run from the marshal repo root:

    .venv/bin/python scripts/dryrun.py --help
    .venv/bin/python scripts/dryrun.py parallel-all
    .venv/bin/python scripts/dryrun.py thrash-test --rounds 10

The harness simulates 4 example client programs hitting marshal with
different X-Program-ID headers, endpoints, and models. The program
identifiers and model names below are placeholders — adapt them to
match your own deployment.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --- Constants --------------------------------------------------------------

MARSHAL_URL = "http://localhost:11435"

# Four example client programs, identified by X-Program-ID header.
# Replace these placeholders with your own program IDs to match your
# deployment's marshal.yaml.
PROGRAMS = {
    "chat-a": "example-chat-a",
    "chat-b": "example-chat-b",
    "generate-a": "example-generate-a",
    "generate-b": "example-generate-b",
}

# Endpoint each program tends to hit.
ENDPOINTS = {
    "chat-a": "/api/chat",
    "chat-b": "/api/chat",
    "generate-a": "/api/generate",
    "generate-b": "/api/generate",
}

# Small default model — adjust to a model present on your Ollama install.
SMALL_DEFAULT = "qwen3.5:4b-bf16"

# Realistic-size models per program. Used when --big is passed.
BIG_PER_PROGRAM = {
    "chat-a": "qwen3.5:9b-bf16",
    "chat-b": "qwen3.5:9b-q8_0",
    "generate-a": "qwen3.5:9b-q4_K_M",
    "generate-b": "qwen3.5:4b-q4_K_M",
}

# Distinct small models for parallel-all so each program loads a different
# one (the whole point of parallel-all is to test inter-program scheduling).
PARALLEL_MODELS = {
    "chat-a": "qwen3.5:4b-bf16",
    "chat-b": "qwen3.5:2b-q4_K_M",
    "generate-a": "qwen3.5:0.8b-bf16",
    "generate-b": "qwen3.5:0.8b-q8_0",
}

# Models for thrash-test default (3 distinct small models to alternate between).
THRASH_DEFAULT = ["qwen3.5:4b-bf16", "qwen3.5:2b-q4_K_M", "qwen3.5:0.8b-bf16"]

console = Console()


# --- App layout -------------------------------------------------------------

app = typer.Typer(
    name="dryrun",
    help="Load harness for ollama-marshal. Pair with the 3-pane dashboard.",
    no_args_is_help=True,
    add_completion=False,
)
single_app = typer.Typer(
    help="Single-shot scenarios (one request per program).",
    no_args_is_help=True,
)
app.add_typer(single_app, name="single")

passthrough_app = typer.Typer(
    help="Boundary scenarios — verify proxy guarantees.",
    no_args_is_help=True,
)
app.add_typer(passthrough_app, name="passthrough")


# --- Helpers ----------------------------------------------------------------


@dataclass
class StatusSnapshot:
    """Subset of /api/marshal/status we care about for assertions."""

    requests_served: int
    model_swaps: int
    evictions: int
    avg_wait_ms: float
    loaded_models: list[str] = field(default_factory=list)
    pending_total: int = 0


def get_status() -> StatusSnapshot:
    """Fetch /api/marshal/status and reduce to assertion-relevant fields.

    Uses .get() with defaults so a partial response (e.g. during marshal
    startup before metrics are initialized) doesn't crash the harness with
    KeyError.
    """
    resp = httpx.get(f"{MARSHAL_URL}/api/marshal/status", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    metrics = data.get("metrics", {})
    queue = data.get("queue", {})
    return StatusSnapshot(
        requests_served=metrics.get("requests_served", 0),
        model_swaps=metrics.get("model_swaps", 0),
        evictions=metrics.get("evictions", 0),
        avg_wait_ms=metrics.get("average_wait_ms", 0.0),
        loaded_models=[m["name"] for m in data.get("loaded_models", [])],
        pending_total=queue.get("total_pending", 0),
    )


def banner(name: str, what_it_tests: str) -> None:
    """Print scenario banner."""
    console.print(
        Panel.fit(
            f"[bold cyan]{name}[/bold cyan]\n[dim]{what_it_tests}[/dim]",
            border_style="cyan",
        )
    )


def verdict(passed: bool, label: str, detail: str = "") -> None:
    """Print colored ✓ / ✗ assertion result."""
    mark = "[green]✓[/green]" if passed else "[red]✗[/red]"
    suffix = f" — {detail}" if detail else ""
    console.print(f"  {mark} {label}{suffix}")


def summary_table(title: str, before: StatusSnapshot, after: StatusSnapshot) -> None:
    """Print before/after delta of marshal status."""
    t = Table(title=title, show_header=True)
    t.add_column("Metric")
    t.add_column("Before", justify="right")
    t.add_column("After", justify="right")
    t.add_column("Δ", justify="right")

    def row(label: str, b: int | float, a: int | float) -> None:
        delta = a - b
        delta_s = (
            f"[green]+{delta}[/green]"
            if delta > 0
            else f"[red]{delta}[/red]"
            if delta < 0
            else "[dim]0[/dim]"
        )
        t.add_row(label, str(b), str(a), delta_s)

    row("requests_served", before.requests_served, after.requests_served)
    row("model_swaps", before.model_swaps, after.model_swaps)
    row("evictions", before.evictions, after.evictions)
    console.print(t)


# --- HTTP firing primitives -------------------------------------------------


def _build_chat_payload(model: str, prompt: str, num_predict: int = 20) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }


def _build_generate_payload(model: str, prompt: str, num_predict: int = 20) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }


def fire_sync(
    program: str,
    model: str,
    prompt: str = "Reply with one word: OK",
    num_predict: int = 20,
    timeout: float = 60.0,
    extra_headers: dict | None = None,
) -> tuple[int, float, str]:
    """Fire a single sync request matching the program's typical endpoint.

    Returns (status_code, duration_ms, content_preview).
    """
    program_id = PROGRAMS[program]
    endpoint = ENDPOINTS[program]
    payload = (
        _build_chat_payload(model, prompt, num_predict)
        if endpoint == "/api/chat"
        else _build_generate_payload(model, prompt, num_predict)
    )
    headers = {"X-Program-ID": program_id}
    if extra_headers:
        headers.update(extra_headers)

    t0 = time.monotonic()
    try:
        resp = httpx.post(
            f"{MARSHAL_URL}{endpoint}",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            content = (
                data.get("message", {}).get("content")
                if endpoint == "/api/chat"
                else data.get("response", "")
            )
            return resp.status_code, duration_ms, (content or "")[:60]
        return resp.status_code, duration_ms, resp.text[:120]
    except httpx.HTTPError as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        return -1, duration_ms, f"HTTPError: {exc}"


async def fire_async(
    program: str, model: str, prompt: str, num_predict: int = 20, timeout: float = 60.0
) -> tuple[int, float, str]:
    """Async variant — used for parallel/burst scenarios."""
    program_id = PROGRAMS[program]
    endpoint = ENDPOINTS[program]
    payload = (
        _build_chat_payload(model, prompt, num_predict)
        if endpoint == "/api/chat"
        else _build_generate_payload(model, prompt, num_predict)
    )

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                f"{MARSHAL_URL}{endpoint}",
                json=payload,
                headers={"X-Program-ID": program_id},
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("message", {}).get("content")
                    if endpoint == "/api/chat"
                    else data.get("response", "")
                )
                return resp.status_code, duration_ms, (content or "")[:60]
            return resp.status_code, duration_ms, resp.text[:120]
        except httpx.HTTPError as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return -1, duration_ms, f"HTTPError: {exc}"


# --- Single-shot scenarios --------------------------------------------------


def _run_single(program: str, model: str) -> None:
    """Shared single-shot runner."""
    program_id = PROGRAMS[program]
    endpoint = ENDPOINTS[program]
    banner(
        f"single {program}",
        f"One {endpoint} request tagged X-Program-ID: {program_id}",
    )
    before = get_status()
    code, ms, content = fire_sync(program, model)
    after = get_status()

    console.print(f"  HTTP [bold]{code}[/bold]  in {ms}ms  → {content!r}")
    verdict(code == 200, "request returned 200")
    verdict(
        after.requests_served >= before.requests_served + 1,
        "requests_served incremented in marshal",
    )


@single_app.command("chat-a")
def single_chat_a(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to use")
    ] = SMALL_DEFAULT,
) -> None:
    """One chat request tagged example-chat-a."""
    _run_single("chat-a", model)


@single_app.command("chat-b")
def single_chat_b(
    model: Annotated[str, typer.Option("--model", "-m")] = SMALL_DEFAULT,
) -> None:
    """One chat request tagged example-chat-b."""
    _run_single("chat-b", model)


@single_app.command("generate-a")
def single_generate_a(
    model: Annotated[str, typer.Option("--model", "-m")] = SMALL_DEFAULT,
) -> None:
    """One generate request tagged example-generate-a."""
    _run_single("generate-a", model)


@single_app.command("generate-b")
def single_generate_b(
    model: Annotated[str, typer.Option("--model", "-m")] = SMALL_DEFAULT,
) -> None:
    """One generate request tagged example-generate-b."""
    _run_single("generate-b", model)


# --- Pattern scenarios ------------------------------------------------------


@app.command("same-model-burst")
def same_model_burst(
    k: Annotated[int, typer.Option("--k", help="Concurrent count")] = 5,
    model: Annotated[str, typer.Option("--model", "-m")] = SMALL_DEFAULT,
) -> None:
    """K concurrent same-model requests (e.g. self-consistency vote).

    Asserts model_swaps delta is 0 — marshal should group all K to the same
    already-loaded model with zero swaps.
    """
    banner(
        f"same-model-burst (k={k})",
        f"K={k} concurrent chat-a requests on {model}. Expect 0 model_swaps.",
    )
    before = get_status()

    async def run() -> list[tuple[int, float, str]]:
        return await asyncio.gather(
            *(
                fire_async("chat-a", model, f"Reply with the digit {i}")
                for i in range(k)
            )
        )

    t0 = time.monotonic()
    results = asyncio.run(run())
    elapsed = int((time.monotonic() - t0) * 1000)
    after = get_status()

    ok = sum(1 for code, _, _ in results if code == 200)
    console.print(f"  All {k} done in [bold]{elapsed}ms[/bold] — {ok}/{k} returned 200")
    for i, (code, ms, content) in enumerate(results):
        console.print(f"    [{i}] {code}  {ms}ms  → {content!r}")

    summary_table("same-model-burst delta", before, after)
    verdict(ok == k, f"all {k} requests succeeded")
    verdict(
        after.model_swaps == before.model_swaps,
        "no model swaps (group-by-model worked)",
        f"swaps Δ={after.model_swaps - before.model_swaps}",
    )


@app.command("same-model-loop")
def same_model_loop(
    rounds: Annotated[int, typer.Option("--rounds", "-n")] = 10,
    model: Annotated[str, typer.Option("--model", "-m")] = SMALL_DEFAULT,
) -> None:
    """Sequential same-model loop (e.g. tool-calling rounds).

    Asserts 0 swaps and 0 evictions — same model throughout.
    """
    banner(
        f"same-model-loop (rounds={rounds})",
        f"Sequential chat-b requests on {model}. Expect 0 swaps, 0 evictions.",
    )
    before = get_status()
    durations: list[int] = []
    failures = 0
    for i in range(rounds):
        code, ms, _ = fire_sync(
            "chat-b", model, f"Round {i}: reply with: round-{i}", num_predict=10
        )
        if code != 200:
            failures += 1
        durations.append(ms)
    after = get_status()

    avg = sum(durations) // len(durations) if durations else 0
    console.print(
        f"  {rounds} rounds, avg {avg}ms, {rounds - failures}/{rounds} succeeded"
    )
    summary_table("same-model-loop delta", before, after)
    verdict(failures == 0, f"all {rounds} rounds succeeded")
    verdict(after.model_swaps == before.model_swaps, "0 model swaps")
    verdict(after.evictions == before.evictions, "0 evictions")


@app.command("parallel-all")
def parallel_all(
    big: Annotated[
        bool, typer.Option("--big", help="Use realistic-size production models")
    ] = False,
) -> None:
    """One request from each of the 4 programs in parallel — THE headline test.

    With small distinct models, all 4 should load if VRAM allows (bin-pack).
    With --big, watch for evictions and drain-before-evict in the dashboard.
    """
    models = BIG_PER_PROGRAM if big else PARALLEL_MODELS

    banner(
        "parallel-all",
        "1 request from each of the 4 example programs, in parallel.\n"
        + "\n".join(f"  {p}: {m}" for p, m in models.items()),
    )
    before = get_status()

    async def run() -> list[tuple[str, tuple[int, float, str]]]:
        async def one(prog: str) -> tuple[str, tuple[int, float, str]]:
            r = await fire_async(
                prog, models[prog], f"Reply: {prog} OK", num_predict=10
            )
            return prog, r

        return await asyncio.gather(*(one(p) for p in PROGRAMS))

    t0 = time.monotonic()
    results = asyncio.run(run())
    elapsed = int((time.monotonic() - t0) * 1000)
    after = get_status()

    console.print(f"  All 4 done in [bold]{elapsed}ms[/bold]")
    for prog, (code, ms, content) in results:
        marker = "[green]OK[/green]" if code == 200 else f"[red]FAIL ({code})[/red]"
        console.print(f"    {prog:14s}  {marker}  {ms}ms  → {content!r}")

    summary_table("parallel-all delta", before, after)
    all_ok = all(code == 200 for _, (code, _, _) in results)
    verdict(all_ok, "all 4 programs succeeded")
    swap_delta = after.model_swaps - before.model_swaps
    verdict(
        swap_delta <= 4,
        "model_swaps bounded (<=4 — one load per distinct model)",
        f"observed Δ={swap_delta}",
    )


@app.command("thrash-test")
def thrash_test(
    rounds: Annotated[int, typer.Option("--rounds", "-n")] = 10,
    models_csv: Annotated[
        str,
        typer.Option(
            "--models",
            help="Comma-separated models to alternate between",
        ),
    ] = ",".join(THRASH_DEFAULT),
) -> None:
    """Alternates between N models to provoke bin-packing and evictions.

    Watch the dashboard for `scheduler.bin_pack_load` and (if memory pressure)
    `scheduler.evicting` events.
    """
    models = [m.strip() for m in models_csv.split(",")]
    banner(
        f"thrash-test (rounds={rounds}, {len(models)} models)",
        f"Alternating: {', '.join(models)}",
    )
    before = get_status()
    failures = 0
    for i in range(rounds):
        m = models[i % len(models)]
        # Cycle through programs too so X-Program-ID varies.
        prog = list(PROGRAMS.keys())[i % len(PROGRAMS)]
        code, ms, _ = fire_sync(prog, m, f"Round {i}", num_predict=8)
        if code != 200:
            failures += 1
        console.print(f"  [{i:2}] {prog:14s}  {m:30s}  {code}  {ms}ms")
    after = get_status()

    summary_table("thrash-test delta", before, after)
    verdict(failures == 0, f"all {rounds} rounds succeeded")
    console.print(
        "  [dim]Look at the log pane for "
        "scheduler.bin_pack_load / evicting events.[/dim]"
    )


@app.command("priority-test")
def priority_test() -> None:
    """Fire normal-priority load, then a critical-priority request.

    Note: requires marshal.yaml to have at least one program with
    `priority: critical`. If all programs are `normal`, this scenario only
    exercises the queue, not preemption.
    """
    banner(
        "priority-test",
        "Normal-priority background load, then a critical-priority request.\n"
        "Requires `priority: critical` for some program in marshal.yaml.",
    )
    before = get_status()

    # Background load: 3 sequential calls on small model from a normal-priority program.
    async def background() -> None:
        for i in range(3):
            await fire_async("chat-b", SMALL_DEFAULT, f"bg {i}", num_predict=20)

    async def run() -> tuple[tuple[int, float, str], None]:
        bg = asyncio.create_task(background())
        # Tiny delay so background gets ahead.
        await asyncio.sleep(0.1)
        # Critical preemption candidate (requires marshal.yaml setting):
        crit = await fire_async(
            "chat-a",
            "qwen3.5:2b-q4_K_M",
            "critical request",
            num_predict=10,
        )
        await bg
        return crit, None

    crit_result, _ = asyncio.run(run())
    after = get_status()

    code, ms, content = crit_result
    console.print(f"  Critical request: HTTP {code}  in {ms}ms  → {content!r}")
    summary_table("priority-test delta", before, after)
    verdict(code == 200, "critical request returned 200")
    console.print(
        "  [dim]Look at log pane for scheduler.critical_preemption "
        "(only fires if a program has priority: critical in marshal.yaml).[/dim]"
    )


# --- Boundary scenarios -----------------------------------------------------


@passthrough_app.command("allowed")
def passthrough_allowed() -> None:
    """GET /api/tags should pass through to Ollama and 200."""
    banner(
        "passthrough allowed",
        "GET /api/tags — must 200 (allowlisted pass-through).",
    )
    resp = httpx.get(f"{MARSHAL_URL}/api/tags", timeout=10)
    console.print(f"  HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        n = len(data.get("models", []))
        console.print(f"  Returned {n} models from Ollama")
    verdict(resp.status_code == 200, "allowlisted pass-through works")


@passthrough_app.command("blocked")
def passthrough_blocked() -> None:
    """POST /api/pull should be blocked with 403."""
    banner(
        "passthrough blocked",
        "POST /api/pull — must 403 (destructive endpoint not proxied).",
    )
    resp = httpx.post(f"{MARSHAL_URL}/api/pull", json={"name": "fake-model"}, timeout=5)
    console.print(f"  HTTP {resp.status_code} → {resp.text[:120]}")
    verdict(
        resp.status_code == 403,
        "destructive endpoint correctly blocked",
        f"got {resp.status_code}",
    )
    body = resp.text.lower()
    verdict(
        "not proxied" in body or "model management" in body,
        "error message names the policy",
    )


@app.command("bad-model")
def bad_model() -> None:
    """Request a model that doesn't exist in Ollama. Must error gracefully, not hang."""
    banner(
        "bad-model",
        "Request model 'this-model-does-not-exist:0' — should error fast, not hang.",
    )
    t0 = time.monotonic()
    code, ms, content = fire_sync(
        "chat-a", "this-model-does-not-exist:0", "test", num_predict=5, timeout=20
    )
    elapsed = int((time.monotonic() - t0) * 1000)
    console.print(f"  HTTP {code} in {elapsed}ms (request itself: {ms}ms)")
    console.print(f"  Content/error: {content!r}")
    # Either 4xx (Ollama rejects) or 502/504 (marshal proxy error) is fine.
    # The point is no hang, no silent wedge.
    verdict(elapsed < 15000, "did not hang (responded under 15s)")
    verdict(code != 200, "did not falsely succeed", f"got {code}")


# --- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    app()
