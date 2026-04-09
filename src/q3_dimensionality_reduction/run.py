"""Entry point for Question 3."""

import numpy as np
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

from src.common.config import DATA_DIR
from src.common.metrics import print_section
from src.common.plotting import figure_dir


MNIST_DIR = DATA_DIR / "raw" / "mnist"


def load_mnist_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the full MNIST dataset and flatten each image to 784 features."""
    train_dataset = MNIST(root=MNIST_DIR, train=True, download=True)
    test_dataset = MNIST(root=MNIST_DIR, train=False, download=True)

    x = np.concatenate(
        [
            train_dataset.data.numpy().reshape(len(train_dataset), -1),
            test_dataset.data.numpy().reshape(len(test_dataset), -1),
        ],
        axis=0,
    ).astype(np.float32)
    y = np.concatenate(
        [
            train_dataset.targets.numpy(),
            test_dataset.targets.numpy(),
        ],
        axis=0,
    ).astype(np.int64)

    return x, y


def report_dataset_overview(x: np.ndarray, y: np.ndarray) -> None:
    """Print basic structure information for the MNIST dataset."""
    print(f"Feature matrix shape: {x.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"Pixel value range: min={x.min():.0f}, max={x.max():.0f}")


def save_sample_digits_figure(x: np.ndarray, y: np.ndarray) -> Path:
    """Save a quick sample grid so the dataset is easy to inspect visually."""
    output_path = figure_dir("q3") / "mnist_sample_digits.png"
    sample_indices = [np.where(y == digit)[0][0] for digit in range(10)]

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for ax, index in zip(axes.ravel(), sample_indices):
        ax.imshow(x[index].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {y[index]}")
        ax.axis("off")

    fig.suptitle("MNIST Sample Digits", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    print_section("Question 3")

    x, y = load_mnist_data()
    report_dataset_overview(x, y)

    sample_path = save_sample_digits_figure(x, y)
    print(f"Saved sample digit figure to: {sample_path}")


if __name__ == "__main__":
    main()
