# Contributing to ollama-marshal

Thank you for your interest in contributing! This guide will help you get
started.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com/) (for integration tests)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/robertvitali/ollama-marshal.git
cd ollama-marshal

# Install in development mode with all dev dependencies
make install-dev

# Verify everything works
make check
```

This installs the package in editable mode, installs dev dependencies, and
sets up pre-commit hooks.

## Development Workflow

### Common Commands

| Command                | What it does                                    |
|------------------------|-------------------------------------------------|
| `make test`            | Run unit tests with coverage                    |
| `make test-integration`| Run integration tests (requires Ollama running) |
| `make lint`            | Run ruff linter                                 |
| `make format`          | Auto-format code with ruff                      |
| `make typecheck`       | Run mypy strict type checking                   |
| `make check`           | Run lint + typecheck + test (full CI locally)    |
| `make clean`           | Remove build artifacts and caches               |

### Code Style

- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/) — handles both
  formatting and linting in one tool
- **Type checking**: [mypy](https://mypy.readthedocs.io/) in strict mode
- **Line length**: 88 characters
- **Docstrings**: Google style on all public functions
- **Type hints**: Required on all function signatures

Pre-commit hooks enforce these automatically. If a hook fails, fix the issue
and re-commit.

### Testing

- **Unit tests**: 95% coverage minimum. Use mocked Ollama responses.
- **Integration tests**: Marked with `@pytest.mark.integration`. Require a
  running Ollama instance. Not required for PRs but encouraged locally.
- **Test naming**: `test_<feature>_<condition>_<expected_result>`

```bash
# Run unit tests only
make test

# Run integration tests (start Ollama first)
make test-integration
```

## Making Changes

### Branch Naming

- `feat/<description>` — new features
- `fix/<description>` — bug fixes
- `docs/<description>` — documentation only
- `refactor/<description>` — code restructuring

### Commit Messages

Write clear, concise commit messages:

- Use imperative mood: "Add feature" not "Added feature"
- First line under 72 characters
- Reference issues where applicable: "Fix queue starvation (#42)"

### Pull Request Process

1. Create a branch from `main`
2. Make your changes with tests
3. Ensure `make check` passes locally
4. Push and open a PR against `main`
5. Fill in the PR template with a summary and test plan
6. PRs are reviewed by the maintainer and by automated Claude Code review

### What Makes a Good PR

- **Focused**: one logical change per PR
- **Tested**: new code has unit tests, coverage stays above 95%
- **Typed**: all new functions have type annotations
- **Documented**: public functions have Google-style docstrings
- **No hardcoded values**: everything configurable via YAML/env/CLI

## Architecture Overview

```
program A ──┐    ┌──────────────────┐
program B ──┼───>│  ollama-marshal   │───> Ollama :11434
program C ──┘    │  (:11435)         │
                 └──────────────────┘
```

Key modules:

| Module            | Responsibility                              |
|-------------------|---------------------------------------------|
| `config.py`       | YAML + env + CLI config with Pydantic       |
| `queue.py`        | Per-model request queues + skip tracking    |
| `registry.py`     | Model size benchmarking and caching         |
| `memory.py`       | RAM detection, budget, /api/ps polling      |
| `lifecycle.py`    | Model preload/unload via Ollama API         |
| `scheduler.py`    | FIFO + bin-packing + fairness scheduler     |
| `stream.py`       | Streaming response proxy                    |
| `openai_compat.py`| OpenAI format translation                   |
| `server.py`       | FastAPI app wiring                          |
| `cli.py`          | Typer CLI entry point                       |

## Questions?

- Open a [Discussion](https://github.com/robertvitali/ollama-marshal/discussions)
  for questions
- Open an [Issue](https://github.com/robertvitali/ollama-marshal/issues) for
  bugs or feature requests
- See [SECURITY.md](SECURITY.md) for vulnerability reports
