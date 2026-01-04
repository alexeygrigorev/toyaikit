.PHONY: test test-integration setup shell coverage format

test:
	uv run pytest

# Run only integration tests in tests_integration/ in parallel
test-integration:
	uv run pytest -n auto tests_integration

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

# Run the full publish script (tests, version bump, build, upload)
publish-script:
	uv run python scripts/publish.py
