"""Helpers for data loading and path handling."""

from pathlib import Path

from src.common.config import DATA_DIR


def dataset_path(*parts: str) -> Path:
    """Return a path inside the data directory."""
    return DATA_DIR.joinpath(*parts)
