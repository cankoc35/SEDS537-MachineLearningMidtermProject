"""Plot output helpers."""

from pathlib import Path

from src.common.config import RESULTS_DIR


def figure_dir(question: str) -> Path:
    """Return the figure directory for a question and ensure it exists."""
    path = RESULTS_DIR / "figures" / question
    path.mkdir(parents=True, exist_ok=True)
    return path
