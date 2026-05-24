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

publish-clean:
	rm -r dist/

# Release: tag the current version and push to trigger CI publish.
# CI workflow: .github/workflows/publish.yml (on tag push v*)
release:
	@VERSION=$$(grep -E "^__version__" toyaikit/__version__.py | sed -E "s/.*['\"]([^'\"]+)['\"].*/\1/"); \
	echo "Releasing v$$VERSION"; \
	git tag "v$$VERSION"; \
	git push origin "v$$VERSION"
