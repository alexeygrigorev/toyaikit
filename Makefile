.PHONY: test setup shell

test:
	uv run pytest

setup:
	uv sync --extra dev

shell:
	uv shell
