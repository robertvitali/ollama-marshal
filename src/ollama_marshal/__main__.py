"""Run the marshal CLI as ``python -m ollama_marshal``.

Exists so integration tests can invoke the branch-under-test's CLI via
``sys.executable -m ollama_marshal`` rather than whatever
``ollama-marshal`` happens to be on PATH (which could be a stale
system install). Production users still invoke ``ollama-marshal`` via
the entry point in pyproject.toml.
"""

from __future__ import annotations

from ollama_marshal.cli import app

if __name__ == "__main__":
    # Pin the program name so help/error text reads "ollama-marshal"
    # rather than "__main__.py" when invoked via -m. Cosmetic but it
    # keeps doctor output consistent across invocation paths.
    app(prog_name="ollama-marshal")
