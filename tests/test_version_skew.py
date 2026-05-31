"""Unit tests for the integration prod-pause version-skew helper.

The helper lives in ``tests/integration/_version_skew.py`` but is a pure
function with no fixture/Ollama dependency, so it is unit-tested here in
the default ``make test`` suite (``tests/integration`` is collection-
ignored there, but the module is still importable as a package).
"""

from __future__ import annotations

from tests.integration._version_skew import version_skew_reason


def test_version_skew_reason_none_when_versions_match():
    assert version_skew_reason("0.6.7", "0.6.7") is None


def test_version_skew_reason_flags_mismatch():
    reason = version_skew_reason("0.6.6", "0.6.7")
    assert reason is not None
    assert "0.6.6" in reason
    assert "0.6.7" in reason


def test_version_skew_reason_flags_missing_prod_version():
    reason = version_skew_reason(None, "0.6.7")
    assert reason is not None
    assert "0.6.7" in reason
    # Communicates that prod didn't report a version (stale marshal).
    assert "did not report" in reason
