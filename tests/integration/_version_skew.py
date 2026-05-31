"""Pure version-skew helper for the integration prod-pause fixture.

Lives outside ``conftest.py`` so the comparison logic can be unit-tested
directly (``tests/test_version_skew.py``) without importing the heavy
fixture module. Mirrors the ``_admin_token`` / ``_fault_proxy`` pattern
of keeping non-fixture helpers in importable ``_*.py`` modules.
"""

from __future__ import annotations


def version_skew_reason(prod_version: str | None, local_version: str) -> str | None:
    """Return a reason string if prod marshal's version differs from the test's.

    Args:
        prod_version: The ``version`` field from the prod marshal's
            ``/api/marshal/status`` payload, or ``None`` when the running
            prod marshal is too old to report it (pre-0.6.7) or the field
            could not be read.
        local_version: ``ollama_marshal.__version__`` of the test marshal.

    Returns:
        ``None`` when the versions match (no skew). Otherwise a short
        human-readable reason describing the mismatch, suitable for a
        structured-log ``reason=`` value.
    """
    if prod_version is None:
        return (
            "prod marshal did not report a version "
            f"(pre-0.6.7, or status unreadable); test marshal is {local_version}"
        )
    if prod_version != local_version:
        return (
            f"prod marshal version {prod_version} != "
            f"test marshal version {local_version}"
        )
    return None
