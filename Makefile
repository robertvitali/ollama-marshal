.PHONY: install-dev test test-integration lint format typecheck check pre-pr clean all dryrun-dashboard

install-dev:
	uv pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type pre-push

test:
	uv run --extra dev pytest tests/ --ignore=tests/integration

test-integration:
	uv run --extra dev pytest tests/integration/ -m integration -v

lint:
	uv run --extra dev ruff check src/ tests/

format:
	uv run --extra dev ruff format src/ tests/

typecheck:
	uv run --extra dev mypy src/

check: lint typecheck test

# Run before opening a PR. Mirrors the pre-push hook (which runs the
# integration suite); use this target to verify locally before push so
# the hook doesn't surprise you. Requires Ollama on localhost:11434.
pre-pr: check test-integration

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

all: format check

dryrun-dashboard:
	@echo "Open 3 iTerm panes (or 3 terminals). Paste one of these into each:"
	@echo ""
	@echo "  Pane 1 (top — refreshing status snapshot):"
	@echo "    watch -n 1 ollama-marshal status"
	@echo ""
	@echo "  Pane 2 (middle — live event stream):"
	@echo "    tail -f ~/.ollama-marshal/marshal.out.log \\"
	@echo "      | grep --line-buffered -E \"scheduler\\.|request_(enqueued|served|timeout|error)\""
	@echo ""
	@echo "  Pane 3 (bottom — fire scenarios):"
	@echo "    .venv/bin/python scripts/dryrun.py --help"
	@echo "    .venv/bin/python scripts/dryrun.py parallel-all"
	@echo ""
	@echo "See scripts/README.md for the full scenario list and expected behaviors."
