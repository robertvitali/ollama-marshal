"""Unit tests for the multi-instance routing decision tree.

The whole point of routing.py is that it's pure logic — no I/O, no
async, no MemoryManager. These tests build synthetic ``RoutingState``
+ ``FitProbe`` snapshots and assert the decision matches the
documented behavior in the module docstring.
"""

from __future__ import annotations

import pytest

from ollama_marshal.config import KVCacheType, OllamaInstance
from ollama_marshal.routing import (
    FitProbe,
    RoutingDecision,
    RoutingReason,
    RoutingState,
    pick_instance,
)

# Convenience builders so each test reads as a state declaration, not setup.
F16 = OllamaInstance(
    url="http://localhost:11434", kv_cache_type=KVCacheType.F16, tier_label="primary"
)
Q8 = OllamaInstance(
    url="http://localhost:11444", kv_cache_type=KVCacheType.Q8_0, tier_label="fallback"
)
Q4 = OllamaInstance(
    url="http://localhost:11454",
    kv_cache_type=KVCacheType.Q4_0,
    tier_label="last_resort",
)

FITS = FitProbe(fits=True, would_evict_non_idle=False)
WOULD_EVICT_NON_IDLE = FitProbe(fits=False, would_evict_non_idle=True)
WOULD_EVICT_ONLY_IDLE = FitProbe(fits=False, would_evict_non_idle=False)


def _state(
    *instances: OllamaInstance,
    loaded_on: dict[OllamaInstance, set[str]] | None = None,
    model: str = "qwen3.5:4b-bf16",
    num_ctx: int = 8192,
) -> RoutingState:
    return RoutingState(
        model_name=model,
        requested_num_ctx=num_ctx,
        instances=list(instances),
        loaded_on={inst.url: names for inst, names in (loaded_on or {}).items()},
    )


# ---------------------------------------------------------------------------
# Trivial: single-instance setups short-circuit
# ---------------------------------------------------------------------------


def test_single_instance_no_routing_decision():
    """One instance configured → take it, regardless of fit info."""
    decision = pick_instance(_state(F16), fit_probe={})
    assert decision == RoutingDecision(
        instance=F16, reason=RoutingReason.SINGLE_INSTANCE, unload_from=[]
    )


def test_empty_instances_raises():
    """Empty instance list is a programming error (validator should catch)."""
    state = RoutingState(
        model_name="x", requested_num_ctx=1, instances=[], loaded_on={}
    )
    with pytest.raises(ValueError, match="no instances configured"):
        pick_instance(state, fit_probe={})


# ---------------------------------------------------------------------------
# Already-loaded: pick highest tier the model is on
# ---------------------------------------------------------------------------


def test_already_loaded_on_f16_stays():
    """Model on f16 → use f16, no probes consulted."""
    decision = pick_instance(
        _state(F16, Q8, Q4, loaded_on={F16: {"qwen3.5:4b-bf16"}}),
        fit_probe={},
    )
    assert decision.instance == F16
    assert decision.reason == RoutingReason.ALREADY_LOADED
    assert decision.unload_from == []


def test_already_loaded_on_f16_unloads_q8_and_q4_stragglers():
    """Same model loaded on multiple tiers → use f16, unload q8 + q4."""
    decision = pick_instance(
        _state(
            F16,
            Q8,
            Q4,
            loaded_on={
                F16: {"qwen3.5:4b-bf16"},
                Q8: {"qwen3.5:4b-bf16"},
                Q4: {"qwen3.5:4b-bf16"},
            },
        ),
        fit_probe={},
    )
    assert decision.instance == F16
    assert decision.reason == RoutingReason.ALREADY_LOADED
    assert set(decision.unload_from) == {Q8, Q4}


def test_already_loaded_on_q8_stays_no_promotion():
    """q8 is 'good enough' — don't pay load tax to promote even if f16 fits."""
    decision = pick_instance(
        _state(F16, Q8, loaded_on={Q8: {"qwen3.5:4b-bf16"}}),
        fit_probe={F16.url: FITS},  # f16 has plenty of room — irrelevant
    )
    assert decision.instance == Q8
    assert decision.reason == RoutingReason.ALREADY_LOADED
    assert decision.unload_from == []


def test_already_loaded_on_q8_unloads_q4_straggler():
    """q8 wins; if q4 also has it, unload q4."""
    decision = pick_instance(
        _state(
            F16,
            Q8,
            Q4,
            loaded_on={Q8: {"qwen3.5:4b-bf16"}, Q4: {"qwen3.5:4b-bf16"}},
        ),
        fit_probe={},
    )
    assert decision.instance == Q8
    assert decision.unload_from == [Q4]


def test_already_loaded_on_q4_promotes_to_f16_when_room():
    """q4-only and f16 fits without eviction → promote up, unload q4."""
    decision = pick_instance(
        _state(F16, Q8, Q4, loaded_on={Q4: {"qwen3.5:4b-bf16"}}),
        fit_probe={F16.url: FITS, Q8.url: FITS},
    )
    assert decision.instance == F16
    assert decision.reason == RoutingReason.PROMOTING_FROM_LAST_RESORT
    assert decision.unload_from == [Q4]


def test_already_loaded_on_q4_promotes_to_q8_when_f16_would_evict_work():
    """q4-only, f16 would evict non-idle, q8 strictly fits → promote to q8."""
    decision = pick_instance(
        _state(F16, Q8, Q4, loaded_on={Q4: {"qwen3.5:4b-bf16"}}),
        fit_probe={F16.url: WOULD_EVICT_NON_IDLE, Q8.url: FITS},
    )
    assert decision.instance == Q8
    assert decision.reason == RoutingReason.PROMOTING_FROM_LAST_RESORT
    assert decision.unload_from == [Q4]


def test_already_loaded_on_q4_stays_when_neither_tier_can_take_it():
    """q4-only, f16 would evict, q8 can't strictly fit → stay on q4."""
    decision = pick_instance(
        _state(F16, Q8, Q4, loaded_on={Q4: {"qwen3.5:4b-bf16"}}),
        fit_probe={
            F16.url: WOULD_EVICT_NON_IDLE,
            Q8.url: WOULD_EVICT_ONLY_IDLE,  # not strictly fits
        },
    )
    assert decision.instance == Q4
    assert decision.reason == RoutingReason.ALREADY_LOADED
    assert decision.unload_from == []


def test_already_loaded_on_q4_promotes_to_f16_even_if_q8_fits():
    """q4-only, f16 fits cleanly → prefer f16 over q8 even if q8 also fits."""
    decision = pick_instance(
        _state(F16, Q8, Q4, loaded_on={Q4: {"qwen3.5:4b-bf16"}}),
        fit_probe={F16.url: FITS, Q8.url: FITS},
    )
    assert decision.instance == F16
    assert decision.reason == RoutingReason.PROMOTING_FROM_LAST_RESORT


# ---------------------------------------------------------------------------
# Cold-start: nothing loaded anywhere
# ---------------------------------------------------------------------------


def test_cold_start_f16_fits():
    decision = pick_instance(
        _state(F16, Q8, Q4),
        fit_probe={F16.url: FITS, Q8.url: FITS, Q4.url: FITS},
    )
    assert decision.instance == F16
    assert decision.reason == RoutingReason.PRIMARY_FITS


def test_cold_start_f16_would_evict_falls_to_q8():
    decision = pick_instance(
        _state(F16, Q8, Q4),
        fit_probe={F16.url: WOULD_EVICT_NON_IDLE, Q8.url: FITS, Q4.url: FITS},
    )
    assert decision.instance == Q8
    assert decision.reason == RoutingReason.PRIMARY_WOULD_EVICT


def test_cold_start_q8_strict_no_fit_falls_to_q4():
    """q8's A-rule: must STRICTLY fit. Eviction-required → not eligible."""
    decision = pick_instance(
        _state(F16, Q8, Q4),
        fit_probe={
            F16.url: WOULD_EVICT_NON_IDLE,
            Q8.url: WOULD_EVICT_ONLY_IDLE,  # would evict — not strict fit
            Q4.url: FITS,
        },
    )
    assert decision.instance == Q4
    assert decision.reason == RoutingReason.FALLBACK_NO_FIT


def test_two_instance_setup_f16_q8_only():
    """f16 + q8 only (no q4). f16 would-evict → falls to q8."""
    decision = pick_instance(
        _state(F16, Q8),
        fit_probe={F16.url: WOULD_EVICT_NON_IDLE, Q8.url: FITS},
    )
    assert decision.instance == Q8
    assert decision.reason == RoutingReason.PRIMARY_WOULD_EVICT


def test_two_instance_setup_q8_q4_only_no_f16_in_play():
    """If no f16 instance, q8 fits → use q8 with FALLBACK_FITS reason."""
    decision = pick_instance(
        _state(Q8, Q4),
        fit_probe={Q8.url: FITS, Q4.url: FITS},
    )
    assert decision.instance == Q8
    assert decision.reason == RoutingReason.FALLBACK_FITS


# ---------------------------------------------------------------------------
# Reason precision: PRIMARY_FITS vs PRIMARY_EVICTING_IDLE
# ---------------------------------------------------------------------------


def test_cold_start_primary_fits_when_fit_probe_fits_true():
    """Cleanly-fits f16 → PRIMARY_FITS."""
    decision = pick_instance(
        _state(F16, Q8, Q4),
        fit_probe={F16.url: FITS},
    )
    assert decision.reason == RoutingReason.PRIMARY_FITS


def test_cold_start_primary_evicting_idle_distinguished_from_fits():
    """f16 fits-only-by-evicting-idle → PRIMARY_EVICTING_IDLE (not PRIMARY_FITS).

    Audit log needs to distinguish "no eviction needed" from "evicting
    idle work but B-rule still allows it" — both route to f16 but
    operators reading the log shouldn't conflate the memory states.
    """
    decision = pick_instance(
        _state(F16, Q8, Q4),
        fit_probe={F16.url: WOULD_EVICT_ONLY_IDLE},
    )
    assert decision.instance == F16
    assert decision.reason == RoutingReason.PRIMARY_EVICTING_IDLE


# ---------------------------------------------------------------------------
# Two-instance topologies missed by the original test pass
# ---------------------------------------------------------------------------


def test_already_loaded_on_q4_no_f16_promotes_to_q8():
    """No f16 configured, q4 has the model, q8 fits → promote to q8."""
    decision = pick_instance(
        _state(Q8, Q4, loaded_on={Q4: {"qwen3.5:4b-bf16"}}),
        fit_probe={Q8.url: FITS},
    )
    assert decision.instance == Q8
    assert decision.reason == RoutingReason.PROMOTING_FROM_LAST_RESORT
    assert decision.unload_from == [Q4]


def test_already_loaded_on_q8_no_f16_stays():
    """No f16 configured, q8 has the model → stay on q8."""
    decision = pick_instance(
        _state(Q8, Q4, loaded_on={Q8: {"qwen3.5:4b-bf16"}}),
        fit_probe={},
    )
    assert decision.instance == Q8
    assert decision.reason == RoutingReason.ALREADY_LOADED


def test_already_loaded_on_f16_no_q8_unloads_only_q4_straggler():
    """f16 + q4 only (no q8), model on both → use f16, unload q4."""
    decision = pick_instance(
        _state(
            F16,
            Q4,
            loaded_on={F16: {"qwen3.5:4b-bf16"}, Q4: {"qwen3.5:4b-bf16"}},
        ),
        fit_probe={},
    )
    assert decision.instance == F16
    assert decision.unload_from == [Q4]


# ---------------------------------------------------------------------------
# Missing-probe contract: if caller forgets to populate, cascade gracefully
# ---------------------------------------------------------------------------


def test_missing_f16_probe_falls_to_q8():
    """Caller didn't populate f16 probe → treat as ineligible, try q8.

    Pins the cascade contract for Stage 2: a forgotten probe entry
    silently routes to lower tier rather than panicking. A future
    Stage 2 caller that misses populating one instance gets the
    behavior captured here, not a surprise.
    """
    decision = pick_instance(
        _state(F16, Q8, Q4),
        fit_probe={Q8.url: FITS, Q4.url: FITS},  # f16 entry missing
    )
    assert decision.instance == Q8


def test_completely_empty_fit_probe_falls_to_q4():
    """Empty probe dict → cascade all the way to q4 (last resort)."""
    decision = pick_instance(
        _state(F16, Q8, Q4),
        fit_probe={},
    )
    assert decision.instance == Q4
    assert decision.reason == RoutingReason.FALLBACK_NO_FIT
