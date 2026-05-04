.PHONY: install-dev test test-integration load-test lint format typecheck check pre-pr clean all dryrun-dashboard

install-dev:
	uv pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type pre-push

test:
	uv run --extra dev pytest tests/ --ignore=tests/integration

test-integration:
	uv run --extra dev pytest tests/integration/ -m integration -v

# Long-running load tests — opt-in only. Excluded from `pre-pr` and
# default `make test-integration` because each scenario runs for
# minutes against real Ollama. Run before each release as a sanity
# check on the v0.6.4 Hop 1 unbounded design (no client→marshal wait
# cap → uvicorn worker pool must not be exhausted by patient
# clients).
load-test:
	uv run --extra dev pytest tests/integration/ -m load -v

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
