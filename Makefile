.PHONY: install-dev test test-integration lint format typecheck check clean all

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
