"""Admin / bypass token discovery for prod-marshal pause/resume.

Two sources, in order:

1. Environment variables — ``MARSHAL_TEST_ADMIN_TOKEN`` and
   ``MARSHAL_TEST_BYPASS_TOKEN``. Set when the operator has sourced
   ``~/.ollama-marshal/admin-tokens.env`` into their shell.
2. ``~/.ollama-marshal/admin-tokens.env`` parsed directly. Lets
   contributors run ``make test-integration`` without first running
   ``source ...`` — important for the v0.6.3 autouse pause fixture
   that needs the token without operator pre-config.

Safety checks (the file is treated as untrusted and ignored when):

- Permissions allow group/other access (any bit in ``0o077`` set).
  Owner-only modes like ``0o400``, ``0o600``, ``0o700`` all pass.
- File is not owned by the current user (defends against a symlink
  pointing at a root-owned or other-user-owned mode-600 file).

Returns ``None`` rather than raising when no token is available so
the autouse fixture can degrade to no-op cleanly when the operator
has no prod marshal configured.
"""

from __future__ import annotations

import os
import re
import stat
from functools import cache
from pathlib import Path

ADMIN_TOKEN_ENV = "MARSHAL_TEST_ADMIN_TOKEN"  # noqa: S105 — env var name, not a secret
BYPASS_TOKEN_ENV = "MARSHAL_TEST_BYPASS_TOKEN"  # noqa: S105 — env var name, not a secret
TOKENS_FILE = Path.home() / ".ollama-marshal" / "admin-tokens.env"

# Matches:  export NAME=value   or   NAME=value   (optional surrounding quotes)
_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)\s*$")


@cache
def _parse_tokens_file(path: Path) -> dict[str, str]:
    """Parse admin-tokens.env into a dict; returns {} if unreadable or unsafe.

    Memoized — token file rarely changes within a test session; the
    cache avoids re-stat/re-parse on every ``discover_*`` call.
    Call ``_parse_tokens_file.cache_clear()`` to invalidate.
    """
    if not path.is_file():
        return {}
    st = path.stat()
    if stat.S_IMODE(st.st_mode) & 0o077:
        return {}
    if st.st_uid != os.getuid():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(raw)
        if m:
            out[m.group(1)] = m.group(2).strip().strip("\"'")
    return out


def _discover(env_var: str) -> str | None:
    """Discover token: env var first, then admin-tokens.env file."""
    if env := os.environ.get(env_var):
        return env
    return _parse_tokens_file(TOKENS_FILE).get(env_var)


def discover_admin_token() -> str | None:
    """Return prod admin token from env or admin-tokens.env, else None."""
    return _discover(ADMIN_TOKEN_ENV)


def discover_bypass_token() -> str | None:
    """Return prod bypass token from env or admin-tokens.env, else None."""
    return _discover(BYPASS_TOKEN_ENV)
