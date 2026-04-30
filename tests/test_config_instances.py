"""Unit tests for the multi-instance config schema + validators.

Covers the new types added in v0.5.0 Track 2 Stage 1:
- ``OllamaInstance`` — frozen + URL-normalizing field validator
- ``MarshalConfig._normalize_instances`` model validator — backward
  compat with legacy ``ollama.host``, duplicate-URL rejection, sort
  by descending precision

Lives in its own file rather than tests/test_config.py to keep the
multi-instance feature work clearly attributable in the diff history.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ollama_marshal.config import (
    TIER_PRIMARY,
    KVCacheType,
    MarshalConfig,
    OllamaInstance,
)

# ---------------------------------------------------------------------------
# OllamaInstance: frozen + url normalization
# ---------------------------------------------------------------------------


def test_ollama_instance_is_frozen():
    """``frozen=True`` so routing can use instances as dict keys / set members."""
    inst = OllamaInstance(url="http://localhost:11434", kv_cache_type=KVCacheType.F16)
    with pytest.raises(ValidationError):
        inst.url = "http://other:1234"  # type: ignore[misc]


def test_ollama_instance_is_hashable():
    """Frozen pydantic model → hashable, usable in sets and as dict keys."""
    inst1 = OllamaInstance(url="http://localhost:11434", kv_cache_type=KVCacheType.F16)
    inst2 = OllamaInstance(url="http://localhost:11434", kv_cache_type=KVCacheType.F16)
    inst3 = OllamaInstance(url="http://other:1234", kv_cache_type=KVCacheType.F16)
    # Same fields → same hash + equal.
    assert hash(inst1) == hash(inst2)
    assert inst1 == inst2
    # Different URL → different hash (well, very likely).
    assert inst1 != inst3
    # Set deduplication works.
    assert len({inst1, inst2, inst3}) == 2


def test_ollama_instance_url_strips_trailing_slash():
    """Cosmetic-only URL differences shouldn't bypass duplicate detection."""
    inst = OllamaInstance(url="http://localhost:11434/", kv_cache_type=KVCacheType.F16)
    assert inst.url == "http://localhost:11434"


def test_ollama_instance_url_lowercases_scheme_and_host():
    """``http://Localhost:11434`` and ``http://localhost:11434`` collapse."""
    inst = OllamaInstance(url="HTTP://Localhost:11434", kv_cache_type=KVCacheType.F16)
    assert inst.url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# MarshalConfig._normalize_instances: backward compat with legacy form
# ---------------------------------------------------------------------------


def test_instances_backfilled_from_singular_ollama_host():
    """Legacy ``ollama.host`` config form auto-promotes to single-instance list."""
    cfg = MarshalConfig.model_validate({"ollama": {"host": "http://localhost:11434"}})
    assert len(cfg.instances) == 1
    assert cfg.instances[0].url == "http://localhost:11434"
    assert cfg.instances[0].kv_cache_type == KVCacheType.F16
    assert cfg.instances[0].tier_label == TIER_PRIMARY


def test_instances_backfill_uses_default_host_when_omitted():
    """No ``ollama`` block → instances backfilled from default host."""
    cfg = MarshalConfig.model_validate({})
    assert len(cfg.instances) == 1
    assert cfg.instances[0].url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Explicit ``instances`` list form
# ---------------------------------------------------------------------------


def test_instances_explicit_list_preserved():
    """Explicit list form takes precedence; entries are validated."""
    cfg = MarshalConfig.model_validate(
        {
            "instances": [
                {
                    "url": "http://localhost:11434",
                    "kv_cache_type": "f16",
                    "tier_label": "primary",
                },
                {
                    "url": "http://localhost:11444",
                    "kv_cache_type": "q8_0",
                    "tier_label": "fallback",
                },
            ],
        }
    )
    assert len(cfg.instances) == 2


def test_instances_sorted_by_precision_descending():
    """Routing assumes ``instances[0]`` is highest precision; validator sorts."""
    cfg = MarshalConfig.model_validate(
        {
            "instances": [
                # Declared in REVERSE precision order to verify sort happens.
                {"url": "http://localhost:11454", "kv_cache_type": "q4_0"},
                {"url": "http://localhost:11434", "kv_cache_type": "f16"},
                {"url": "http://localhost:11444", "kv_cache_type": "q8_0"},
            ],
        }
    )
    assert [i.kv_cache_type for i in cfg.instances] == [
        KVCacheType.F16,
        KVCacheType.Q8_0,
        KVCacheType.Q4_0,
    ]


def test_instances_duplicate_url_rejected():
    """Two entries with the same URL → ValidationError."""
    with pytest.raises(ValidationError, match="duplicate Ollama instance URL"):
        MarshalConfig.model_validate(
            {
                "instances": [
                    {"url": "http://localhost:11434", "kv_cache_type": "f16"},
                    {"url": "http://localhost:11434", "kv_cache_type": "q8_0"},
                ],
            }
        )


def test_instances_duplicate_url_after_normalization_rejected():
    """``http://X:Y/`` and ``http://X:Y`` collapse to same URL → rejected."""
    with pytest.raises(ValidationError, match="duplicate Ollama instance URL"):
        MarshalConfig.model_validate(
            {
                "instances": [
                    {"url": "http://localhost:11434", "kv_cache_type": "f16"},
                    {"url": "http://localhost:11434/", "kv_cache_type": "q8_0"},
                ],
            }
        )


def test_instances_invalid_kv_cache_type_rejected():
    """Unknown ``kv_cache_type`` is rejected at field-validation time."""
    with pytest.raises(ValidationError):
        MarshalConfig.model_validate(
            {
                "instances": [
                    {"url": "http://localhost:11434", "kv_cache_type": "q5_0"},
                ],
            }
        )
