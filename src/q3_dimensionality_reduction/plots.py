"""Plotting utilities for Question 3."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.common.plotting import figure_dir


def save_embedding_scatter(embedding: np.ndarray, labels: np.ndarray, filename: str, title: str) -> None:
    """Save a 2D embedding scatter plot coloured by digit label."""
    output_path = figure_dir("q3") / filename

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.colorbar(scatter, ticks=range(10), label="Digit label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
