"""Diagnostic helpers for the `marshal doctor` CLI subcommand.

`doctor` answers the question: *"Why is Ollama swapping models when
both fit in RAM?"* It reads `/api/tags`, `/api/show`, and `/api/ps` to
compute worst-case KV cache demand under the user's likely co-resident
model pairs, compares against system RAM, and recommends specific
environment variables for the Ollama startup config.

Pure data analysis — no scheduler state, no marshal config dependency.
Lets the CLI render the report and lets tests assert against
deterministic recommendation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import psutil
import structlog

from ollama_marshal.registry import MalformedTagsResponseError, ModelRegistry

logger = structlog.get_logger()

# KV cache type savings vs the default fp16:
# - q8_0 halves the KV cache size at negligible quality loss
# - q4_0 quarters it but visible quality loss on long contexts
# We only recommend q8_0; users wanting q4_0 set it manually.
_KV_CACHE_TYPE_RECOMMEND = "q8_0"
_KV_CACHE_FACTOR_Q8_0 = 0.5

# Flash attention dramatically reduces attention memory and is widely
# supported on M-series Macs and recent NVIDIA GPUs.
_FLASH_ATTENTION_RECOMMEND = True


@dataclass
class ModelDoctorEntry:
    """One model's analysis row.

    Attributes:
        name: Model name.
        loaded: Whether the model is currently loaded in /api/ps.
        size_vram_bytes: Currently-loaded VRAM size, or 0 if not loaded.
        kv_per_slot_at_max_ctx: KV cache cost per slot at the model's
            architectural max context, in bytes (fp16). The single
            biggest knob in determining whether two models can coexist.
        max_context_length: The model's architectural max num_ctx.
        probe_ok: Whether /api/show returned usable metadata.
    """

    name: str
    loaded: bool
    size_vram_bytes: int
    kv_per_slot_at_max_ctx: int
    max_context_length: int
    probe_ok: bool


@dataclass
class DoctorReport:
    """Full diagnostic report.

    Attributes:
        total_ram_bytes: psutil.virtual_memory().total
        loaded_models: Models currently in /api/ps.
        all_models: All models in /api/tags.
        recommended_env: dict of OLLAMA_* env var → recommended value.
        notes: Free-form lines explaining the recommendations.
        unexpected_unloads: Lifetime counter from marshal's
            SchedulerMetrics (or None if marshal isn't reachable).
        live_pressure_refusals: Lifetime count of loads marshal refused
            because LIVE OS memory couldn't fit them (M2 admission), or
            None if marshal isn't reachable / predates v0.6.7.
        live_memory: Marshal's `memory.live` status block (enabled /
            available / headroom / last_refusal), or None if marshal
            isn't reachable / predates v0.6.7.
    """

    total_ram_bytes: int
    loaded_models: list[ModelDoctorEntry]
    all_models: list[ModelDoctorEntry]
    recommended_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unexpected_unloads: int | None = None
    live_pressure_refusals: int | None = None
    live_memory: dict[str, Any] | None = None


async def gather_report(
    ollama_host: str,
    marshal_status_url: str | None = None,
) -> DoctorReport:
    """Build a DoctorReport by probing Ollama and (optionally) marshal.

    Args:
        ollama_host: e.g. "http://localhost:11434".
        marshal_status_url: e.g. "http://localhost:11435/api/marshal/status".
            When set, the report includes marshal's `unexpected_unloads`
            counter — non-zero values strongly suggest Ollama-side
            tuning is needed.

    Returns:
        Populated DoctorReport.
    """
    total_ram = psutil.virtual_memory().total

    registry = ModelRegistry(ollama_host=ollama_host)
    try:
        all_names = await registry.fetch_model_list()
    except (httpx.HTTPError, MalformedTagsResponseError):
        logger.warning("doctor.tags_unreachable")
        return DoctorReport(total_ram_bytes=total_ram, loaded_models=[], all_models=[])

    loaded_names_with_size = await _fetch_loaded(ollama_host)

    all_entries: list[ModelDoctorEntry] = []
    for name in all_names:
        meta = await registry.probe_metadata(name)
        size_vram = loaded_names_with_size.get(name, 0)
        if meta is None:
            all_entries.append(
                ModelDoctorEntry(
                    name=name,
                    loaded=name in loaded_names_with_size,
                    size_vram_bytes=size_vram,
                    kv_per_slot_at_max_ctx=0,
                    max_context_length=0,
                    probe_ok=False,
                )
            )
            continue
        all_entries.append(
            ModelDoctorEntry(
                name=name,
                loaded=name in loaded_names_with_size,
                size_vram_bytes=size_vram,
                kv_per_slot_at_max_ctx=meta.kv_per_slot_at_max_ctx,
                max_context_length=meta.max_context_length,
                probe_ok=True,
            )
        )

    loaded_entries = [e for e in all_entries if e.loaded]

    report = DoctorReport(
        total_ram_bytes=total_ram,
        loaded_models=loaded_entries,
        all_models=all_entries,
    )
    _populate_recommendations(report)

    if marshal_status_url is not None:
        status = await _fetch_marshal_status(marshal_status_url)
        if status is not None:
            metrics = status.get("metrics") or {}
            # bool is an int subclass — reject it so a malformed payload
            # (e.g. "unexpected_unloads": true) can't render as a count.
            val = metrics.get("unexpected_unloads")
            if isinstance(val, int) and not isinstance(val, bool):
                report.unexpected_unloads = val
            refusals = metrics.get("live_pressure_refusals")
            if isinstance(refusals, int) and not isinstance(refusals, bool):
                report.live_pressure_refusals = refusals
            live = (status.get("memory") or {}).get("live")
            if isinstance(live, dict):
                report.live_memory = live
            _populate_live_memory_notes(report)

    return report


async def _fetch_loaded(ollama_host: str) -> dict[str, int]:
    """Return {model_name: size_vram_bytes} from /api/ps."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ollama_host}/api/ps", timeout=5)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
    except httpx.HTTPError:
        return {}
    out: dict[str, int] = {}
    for m in data.get("models", []):
        name = m.get("name", "")
        if name:
            out[name] = int(m.get("size_vram", 0))
    return out


async def _fetch_marshal_status(marshal_status_url: str) -> dict[str, Any] | None:
    """Fetch marshal's full /api/marshal/status payload (or None if down)."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(marshal_status_url, timeout=5)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
    except httpx.HTTPError:
        return None
    return data


def _populate_live_memory_notes(report: DoctorReport) -> None:
    """Add notes for the M2 live-aware admission signal (Memory M3).

    Deterministic and testable, mirroring _populate_recommendations:

    1. Refusals observed → the box's REAL free RAM blocked loads (the
       static budget may or may not have had room); point at likely
       external consumers without asserting sole causality.
    2. Live admission enabled but never sampled → the poll loop hasn't
       seeded the EWMA; admission is running static-only.
    3. Live admission disabled → say so (one-line visibility).
    """
    live = report.live_memory
    if report.live_pressure_refusals is not None and report.live_pressure_refusals > 0:
        # Causality is deliberately hedged (codex P3): a refusal proves
        # live headroom < model size at that moment — the static budget
        # may ALSO have been insufficient. Common culprits are external
        # consumers, but this note must not assert the budget had room.
        report.notes.append(
            f"marshal refused {report.live_pressure_refusals} model load(s) "
            "because the box's LIVE free RAM couldn't fit them at the "
            "time. Frequent causes: an external consumer (another "
            "inference engine, a build, another app) holding RAM the "
            "static budget doesn't see. Check `top` sorted by memory, "
            "and compare marshal's memory budget against reality."
        )
    if live is None:
        return
    if live.get("enabled") and live.get("available") is None:
        report.notes.append(
            "Live-aware admission is enabled but no live-memory sample has "
            "landed yet (cold start) — admission is running static-only "
            "until the first poll."
        )
    elif not live.get("enabled"):
        report.notes.append(
            "Live-aware admission (memory.live_memory_enabled) is DISABLED — "
            "marshal admits loads from its static budget alone and cannot "
            "see RAM held by other processes."
        )


def _populate_recommendations(report: DoctorReport) -> None:
    """Compute the recommended env vars based on report contents.

    Recommendation logic is deterministic and testable:

    1. Always recommend `OLLAMA_KV_CACHE_TYPE=q8_0` — halves KV cache
       size with negligible quality loss.
    2. Always recommend `OLLAMA_FLASH_ATTENTION=1` — modern, supported
       on M-series + recent NVIDIA, reduces attention memory.
    3. Compute `OLLAMA_NUM_PARALLEL`: the largest power-of-2 number
       such that the LARGEST model's KV cache (* num_parallel) +
       weight allocation fits comfortably in 25% of system RAM. Caps
       at 4 (Ollama's prior default) for stability.
    4. Compute `OLLAMA_MAX_LOADED_MODELS`: the count of models we
       could simultaneously load with `OLLAMA_NUM_PARALLEL` slots
       each, treating each as worst-case at its max_context.
    """
    report.recommended_env["OLLAMA_KV_CACHE_TYPE"] = _KV_CACHE_TYPE_RECOMMEND
    if _FLASH_ATTENTION_RECOMMEND:
        report.recommended_env["OLLAMA_FLASH_ATTENTION"] = "1"

    ram = report.total_ram_bytes
    if ram <= 0:
        report.notes.append(
            "Cannot determine system RAM; skipping NUM_PARALLEL recommendation."
        )
        return

    largest = _largest_kv_cost_model(report.all_models)
    if largest is None:
        report.notes.append(
            "No probed model metadata available; skipping NUM_PARALLEL recommendation."
        )
        return

    # Budget per-model KV cache to 25% of RAM (leaves headroom for
    # weights + OS + safety). Apply q8_0 savings since we recommend it.
    kv_budget = int(ram * 0.25)
    kv_per_slot_q8 = int(largest.kv_per_slot_at_max_ctx * _KV_CACHE_FACTOR_Q8_0)
    if kv_per_slot_q8 == 0:
        report.notes.append(
            "Couldn't compute KV-per-slot for the largest-context model; "
            "leaving NUM_PARALLEL at Ollama default."
        )
        return

    parallel = max(1, min(4, kv_budget // kv_per_slot_q8))
    report.recommended_env["OLLAMA_NUM_PARALLEL"] = str(parallel)
    report.notes.append(
        f"Largest-context model is {largest.name} "
        f"({largest.kv_per_slot_at_max_ctx / (1024**3):.1f} GB KV/slot at "
        f"max ctx, fp16 → ~{kv_per_slot_q8 / (1024**3):.1f} GB at q8_0)."
    )

    # MAX_LOADED_MODELS: how many distinct models could co-reside at
    # NUM_PARALLEL slots each. Use mean KV cost as a rough proxy.
    if report.all_models:
        mean_kv_q8 = (
            sum(m.kv_per_slot_at_max_ctx for m in report.all_models)
            / len(report.all_models)
            * _KV_CACHE_FACTOR_Q8_0
        )
        if mean_kv_q8 > 0:
            kv_per_model = mean_kv_q8 * parallel
            max_loaded = max(1, int(ram * 0.6 / kv_per_model))
            report.recommended_env["OLLAMA_MAX_LOADED_MODELS"] = str(max_loaded)


def _largest_kv_cost_model(
    entries: list[ModelDoctorEntry],
) -> ModelDoctorEntry | None:
    """Return the entry with the largest kv_per_slot_at_max_ctx, or None.

    Skips entries where the probe failed (no metadata to score on).
    """
    candidates = [e for e in entries if e.probe_ok and e.kv_per_slot_at_max_ctx > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.kv_per_slot_at_max_ctx)


def render_report(report: DoctorReport) -> str:
    """Render a DoctorReport as a human-readable string.

    Format is plain text, no colors — intended to be safely piped
    into a launchd plist editor. The CLI writes this directly to
    stdout via typer.echo.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  ollama-marshal doctor")
    lines.append("=" * 60)

    ram_gb = report.total_ram_bytes / (1024**3)
    lines.append(f"  System RAM:        {ram_gb:.1f} GB")
    lines.append(f"  Models installed:  {len(report.all_models)}")
    lines.append(f"  Models loaded:     {len(report.loaded_models)}")
    if report.unexpected_unloads is not None:
        lines.append(f"  Unexpected unloads (marshal): {report.unexpected_unloads}")
        if report.unexpected_unloads > 0:
            lines.append(
                "    ⚠ Ollama is dropping models on its own — apply the "
                "recommendations below."
            )
    if report.live_pressure_refusals is not None:
        lines.append(
            f"  Live-pressure refusals (marshal): {report.live_pressure_refusals}"
        )
        if report.live_pressure_refusals > 0:
            lines.append("    ⚠ Live OS memory blocked model loads — see Notes.")
    if report.live_memory is not None:
        live = report.live_memory
        if live.get("enabled"):
            avail = live.get("available")
            headroom = live.get("headroom")
            avail_s = (
                f"{avail / (1024**3):.1f} GB" if isinstance(avail, int) else "unsampled"
            )
            head_s = (
                f"{headroom / (1024**3):.1f} GB" if isinstance(headroom, int) else "n/a"
            )
            lines.append(
                f"  Live memory (marshal): available={avail_s}   headroom={head_s}"
            )
        else:
            lines.append("  Live memory (marshal): DISABLED")
    lines.append("")

    if report.loaded_models:
        lines.append("  Loaded models:")
        for m in report.loaded_models:
            kv_gb = m.kv_per_slot_at_max_ctx / (1024**3)
            lines.append(
                f"    {m.name:<35} weights={m.size_vram_bytes / (1024**3):>5.1f} GB"
                f"   KV/slot@max_ctx={kv_gb:>5.1f} GB"
            )
        lines.append("")

    if report.recommended_env:
        lines.append("  Recommended Ollama startup env (set in launchd plist):")
        for k, v in report.recommended_env.items():
            lines.append(f"    {k}={v}")
        lines.append("")

    if report.notes:
        lines.append("  Notes:")
        for note in report.notes:
            lines.append(f"    - {note}")
        lines.append("")

    lines.append(
        "  Apply these by editing your Ollama launchd plist (macOS) or "
        "systemd unit (Linux),"
    )
    lines.append(
        "  then restart Ollama. Verify with: `marshal doctor` again — "
        "Unexpected unloads"
    )
    lines.append("  should drop to 0.")
    lines.append("=" * 60)
    return "\n".join(lines)
