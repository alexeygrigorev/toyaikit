.PHONY: test setup shell coverage format

test:
	uv run pytest

coverage:
	uv run pytest --cov=toyaikit --cov-report=term-missing --cov-report=html

setup:
	uv sync --dev

shell:
	uv shell

format:
	uv run ruff format .
	uv run ruff check --fix .