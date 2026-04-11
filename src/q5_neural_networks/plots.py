"""Plotting helpers for Question 5."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.common.plotting import figure_dir
from src.q5_neural_networks.data import CLASS_NAMES


def save_fashion_mnist_sample_grid(train_dataset, num_examples: int = 10) -> None:
    """Save a simple sample grid for the Fashion-MNIST training set."""
    output_path = figure_dir("q5") / "fashion_mnist_sample_grid.png"

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for index in range(num_examples):
        image, label = train_dataset[index]
        axes[index].imshow(image.squeeze(0), cmap="gray")
        axes[index].set_title(CLASS_NAMES[label], fontsize=9)
        axes[index].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
