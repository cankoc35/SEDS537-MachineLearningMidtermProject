"""Plotting utilities for Question 2."""

import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve

from src.common.plotting import figure_dir


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_roc_curve(y_true, y_score, model_name: str, filename: str) -> None:
    """Save an ROC curve plot for a binary classifier."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    output_path = figure_dir("q2") / filename

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
