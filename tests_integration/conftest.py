import os
from pathlib import Path

from dotenv import load_dotenv


def pytest_configure(config):  # noqa: D401
    """Pytest hook to load .env for integration tests if present."""
    load_dotenv()

