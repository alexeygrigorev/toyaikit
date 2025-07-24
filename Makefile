.PHONY: test setup shell

test:
	uv run pytest

setup:
	uv sync --dev

shell:
	uv shell
