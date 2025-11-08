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

publish-build:
	uv run hatch build

publish-test:
	uv run hatch publish --repo test

publish:
	uv run hatch publish

publish-clean:
	rm -r dist/