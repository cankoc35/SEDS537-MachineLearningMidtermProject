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


def save_training_history_plot(history_df, model_name: str) -> None:
    """Save train/validation loss and accuracy curves for one model."""
    output_path = figure_dir("q5") / f"{model_name.lower()}_training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    axes[0].plot(history_df["epoch"], history_df["validation_loss"], label="Validation Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy")
    axes[1].plot(history_df["epoch"], history_df["validation_accuracy"], label="Validation Accuracy")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
