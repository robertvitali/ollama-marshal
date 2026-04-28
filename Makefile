.PHONY: install-dev test test-integration lint format typecheck check clean all dryrun-dashboard

install-dev:
	uv pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ --ignore=tests/test_integration.py

test-integration:
	pytest tests/test_integration.py -m integration -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

check: lint typecheck test

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
