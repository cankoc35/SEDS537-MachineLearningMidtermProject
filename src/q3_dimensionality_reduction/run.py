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
from src.q3_dimensionality_reduction.modeling import run_knn_accuracy_comparison
from src.q3_dimensionality_reduction.pca_analysis import run_pca_analysis
from src.q3_dimensionality_reduction.tsne_analysis import run_tsne_embedding, select_tsne_subset_indices


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

    x_pca_2d, x_pca_50d, explained_variance_df = run_pca_analysis(x, y)
    print(f"PCA 2D shape: {x_pca_2d.shape}")
    print(f"PCA 50D shape: {x_pca_50d.shape}")
    print("\nExplained variance ratio for first 10 PCA components:")
    print(explained_variance_df.round(4).to_string(index=False))

    print("\nRunning t-SNE on the PCA 50D representation...")
    x_tsne_30, y_tsne_30 = run_tsne_embedding(x_pca_50d, y, perplexity=30)
    print(f"t-SNE 2D shape (perplexity=30): {x_tsne_30.shape}")
    print(f"t-SNE label shape (perplexity=30): {y_tsne_30.shape}")

    x_tsne_50, y_tsne_50 = run_tsne_embedding(x_pca_50d, y, perplexity=50)
    print(f"t-SNE 2D shape (perplexity=50): {x_tsne_50.shape}")
    print(f"t-SNE label shape (perplexity=50): {y_tsne_50.shape}")

    subset_indices = select_tsne_subset_indices(y)
    x_subset = x[subset_indices]
    y_subset = y[subset_indices]
    x_pca_50d_subset = x_pca_50d[subset_indices]

    knn_results_df = run_knn_accuracy_comparison(
        {
            "Original 784D": (x_subset, y_subset),
            "PCA 50D": (x_pca_50d_subset, y_subset),
            "t-SNE 2D (perplexity 30)": (x_tsne_30, y_tsne_30),
            "t-SNE 2D (perplexity 50)": (x_tsne_50, y_tsne_50),
        }
    )
    print("\nk-NN (k=5) 5-fold cross-validation accuracy:")
    print(knn_results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
