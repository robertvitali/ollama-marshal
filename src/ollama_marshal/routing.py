"""Multi-instance routing — pick which Ollama instance serves a request.

Pure decision logic, no I/O. Callers provide a snapshot of current
state via ``RoutingState``; the function returns the chosen instance
plus a structured ``RoutingDecision`` describing why.

Design intent (from the v0.5.0 Track 2 design discussion):

- f16 (primary)        : default target for every request
- q8_0 (fallback)      : used when loading on f16 would require evicting
                         a NON-IDLE model (B-rule, "avoid evicting work")
- q4_0 (last resort)   : used only when the request strictly cannot fit
                         on q8_0 (A-rule, "no other option")

Already-loaded model handling:

- If model is loaded on f16: stay on f16
- If model is loaded on q8: stay on q8 (already-loaded wins; q8 quality
  is good enough that we don't pay the load-time tax to promote it)
- If model is loaded on q4 ONLY: try to escape — apply B-rule for f16,
  then A-rule for q8, only fall back to q4 if neither tier can take it
- If model is loaded on multiple tiers: highest tier wins (caller is
  responsible for unloading the lower-tier copies if it wants to)

Single-instance setups (``len(instances) == 1``) short-circuit to that
single instance — no routing logic runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ollama_marshal.config import KVCacheType, OllamaInstance


class RoutingReason(StrEnum):
    """Structured cause for a routing decision (audit log + status)."""

    SINGLE_INSTANCE = "single_instance"
    """Only one instance configured — no routing choice to make."""

    ALREADY_LOADED = "already_loaded"
    """Model is already loaded on this instance; using it directly."""

    PRIMARY_FITS = "primary_fits"
    """Cold-start: f16 has room for the model + KV cache."""

    PRIMARY_WOULD_EVICT = "primary_would_evict"
    """f16 can't fit without evicting a non-idle model — falling back."""

    FALLBACK_FITS = "fallback_fits"
    """Cold-start: q8_0 fits and was chosen as fallback for f16."""

    FALLBACK_NO_FIT = "fallback_no_fit"
    """q8_0 strictly cannot fit either; using q4_0 as last resort."""

    PROMOTING_FROM_LAST_RESORT = "promoting_from_last_resort"
    """Model was on q4_0; f16 or q8_0 has room now → promote up."""


@dataclass(frozen=True)
class FitProbe:
    """Result of asking memory whether a model would fit on an instance.

    Three possible outcomes:
    - ``fits=True`` — instance has room, no eviction required
    - ``fits=False, would_evict_non_idle=True`` — instance can fit only
      by evicting a model that has pending requests or recent activity
    - ``fits=False, would_evict_non_idle=False`` — instance cannot fit
      even after evicting every idle model on it

    The B-rule (avoid evicting work) cares about the second case; the
    A-rule (strict no-fit) only treats the third case as ineligible.
    """

    fits: bool
    """True if the instance can serve the request without ANY eviction."""

    would_evict_non_idle: bool
    """If True, fitting requires evicting a model that's actively in use."""


@dataclass(frozen=True)
class RoutingState:
    """Snapshot of state the routing decision needs.

    Caller (typically Scheduler._ensure_model_loaded) builds this from
    MemoryManager + the request being scheduled. Routing logic is pure
    given this snapshot — no I/O, no global lookups, easy to test.
    """

    model_name: str
    """The model the request needs."""

    requested_num_ctx: int
    """num_ctx the request is asking for (already clamped to model max)."""

    instances: list[OllamaInstance]
    """All configured instances, sorted by precision (f16 → q4)."""

    loaded_on: dict[str, set[str]]
    """``instance_url → set of model names loaded on that instance``.

    Authoritative source: MemoryManager. Caller passes a snapshot at
    decision time; routing doesn't observe state changing during its
    execution.
    """


@dataclass(frozen=True)
class RoutingDecision:
    """Chosen instance + the reason for picking it.

    The reason is recorded in the audit log so operators can correlate
    "why did this request run on q8 instead of f16?" with the memory
    state at that moment.
    """

    instance: OllamaInstance
    """The instance the request will be served by."""

    reason: RoutingReason
    """Structured cause; goes into audit log and status output."""

    unload_from: list[OllamaInstance]
    """Other instances currently holding stale copies of this model.

    Populated when promoting a model OFF q4_0 — caller should unload
    from these instances after preloading on the chosen instance.
    Empty list means no cleanup needed.
    """


def pick_instance(
    state: RoutingState,
    fit_probe: dict[str, FitProbe],
) -> RoutingDecision:
    """Choose which instance serves the request.

    Args:
        state: Snapshot of routing-relevant state.
        fit_probe: Per-instance answer to "would this model + num_ctx
            fit on this instance, and at what cost?". Caller computes
            this by asking MemoryManager for each instance. Keyed by
            ``instance.url``.

    Returns:
        RoutingDecision with the chosen instance + structured reason.

    Raises:
        ValueError: If ``state.instances`` is empty (config validator
            should prevent this; treated as a programming error).

    The decision tree intentionally walks state in a fixed order so
    behavior is deterministic and easy to reason about during audit:

    1. Trivial: single instance → take it
    2. Already-loaded: stay where the model is, preferring higher tier
       (f16 > q8 > q4). The q4-only case is special — we try to escape.
    3. Cold-start: walk tiers top-down, applying the trigger rule for
       each step (B-rule from f16 to q8, A-rule from q8 to q4)
    """
    if not state.instances:
        msg = "routing.pick_instance called with no instances configured"
        raise ValueError(msg)

    # Trivial case: single-instance setup. No routing decision.
    if len(state.instances) == 1:
        return RoutingDecision(
            instance=state.instances[0],
            reason=RoutingReason.SINGLE_INSTANCE,
            unload_from=[],
        )

    # Index instances by precision tier for clearer logic below.
    tier_to_instance: dict[KVCacheType, OllamaInstance] = {
        inst.kv_cache_type: inst for inst in state.instances
    }
    f16 = tier_to_instance.get(KVCacheType.F16)
    q8 = tier_to_instance.get(KVCacheType.Q8_0)
    q4 = tier_to_instance.get(KVCacheType.Q4_0)

    # Already-loaded check: pick the highest tier that has it.
    loaded_on_f16 = f16 is not None and state.model_name in state.loaded_on.get(
        f16.url, set()
    )
    loaded_on_q8 = q8 is not None and state.model_name in state.loaded_on.get(
        q8.url, set()
    )
    loaded_on_q4 = q4 is not None and state.model_name in state.loaded_on.get(
        q4.url, set()
    )

    if loaded_on_f16:
        # Already on the highest tier — done. (Unload q8/q4 stragglers
        # if they exist, though that should be rare.)
        assert f16 is not None
        unload = []
        if loaded_on_q8 and q8 is not None:
            unload.append(q8)
        if loaded_on_q4 and q4 is not None:
            unload.append(q4)
        return RoutingDecision(
            instance=f16,
            reason=RoutingReason.ALREADY_LOADED,
            unload_from=unload,
        )

    if loaded_on_q8:
        # q8 is "good enough — don't pay the load tax to promote".
        assert q8 is not None
        unload = []
        if loaded_on_q4 and q4 is not None:
            unload.append(q4)
        return RoutingDecision(
            instance=q8,
            reason=RoutingReason.ALREADY_LOADED,
            unload_from=unload,
        )

    if loaded_on_q4:
        # q4 only — try to escape. Apply B-rule for f16, A-rule for q8.
        if f16 is not None:
            f16_probe = fit_probe.get(f16.url)
            if f16_probe is not None and not f16_probe.would_evict_non_idle:
                # f16 fits without evicting work → promote.
                assert q4 is not None
                return RoutingDecision(
                    instance=f16,
                    reason=RoutingReason.PROMOTING_FROM_LAST_RESORT,
                    unload_from=[q4],
                )
        if q8 is not None:
            q8_probe = fit_probe.get(q8.url)
            if q8_probe is not None and q8_probe.fits:
                # q8 strictly fits (A-rule) → promote.
                assert q4 is not None
                return RoutingDecision(
                    instance=q8,
                    reason=RoutingReason.PROMOTING_FROM_LAST_RESORT,
                    unload_from=[q4],
                )
        # Neither tier can take it → stay on q4.
        assert q4 is not None
        return RoutingDecision(
            instance=q4,
            reason=RoutingReason.ALREADY_LOADED,
            unload_from=[],
        )

    # Cold-start path: nothing loaded anywhere. Walk tiers top-down.
    if f16 is not None:
        f16_probe = fit_probe.get(f16.url)
        if f16_probe is not None and not f16_probe.would_evict_non_idle:
            return RoutingDecision(
                instance=f16,
                reason=RoutingReason.PRIMARY_FITS,
                unload_from=[],
            )

    if q8 is not None:
        q8_probe = fit_probe.get(q8.url)
        if q8_probe is not None and q8_probe.fits:
            # f16 either doesn't exist or would evict non-idle.
            return RoutingDecision(
                instance=q8,
                reason=(
                    RoutingReason.PRIMARY_WOULD_EVICT
                    if f16 is not None
                    else RoutingReason.FALLBACK_FITS
                ),
                unload_from=[],
            )

    if q4 is not None:
        # Last resort. q4 always "fits" because we don't have anywhere
        # else to go; if even q4 can't accommodate, that's a downstream
        # error not routing's job to surface.
        return RoutingDecision(
            instance=q4,
            reason=RoutingReason.FALLBACK_NO_FIT,
            unload_from=[],
        )

    # Reaching here means f16/q8/q4 are all None but len(instances) >=
    # 2 — possible only with non-canonical kv_cache_types we don't
    # know how to rank. Fall back to first configured instance.
    return RoutingDecision(
        instance=state.instances[0],
        reason=RoutingReason.SINGLE_INSTANCE,
        unload_from=[],
    )
