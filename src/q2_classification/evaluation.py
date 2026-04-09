"""Evaluation helpers for Question 2."""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.config import RESULTS_DIR


def evaluate_binary_classifier(model_name: str, y_true, y_pred, y_score) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the required Question 2 classification metrics."""
    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_score),
            }
        ]
    )

    confusion_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )

    return metrics_df, confusion_df


def save_results(
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    experiment_name: str = "baseline",
) -> None:
    """Save classifier outputs for a named Question 2 experiment."""
    table_dir = RESULTS_DIR / "tables" / "q2"
    table_dir.mkdir(parents=True, exist_ok=True)
    file_stem = metrics_df.loc[0, "model"].lower().replace(" ", "_")
    metrics_df.to_csv(table_dir / f"{file_stem}_{experiment_name}_metrics.csv", index=False)
    confusion_df.to_csv(table_dir / f"{file_stem}_{experiment_name}_confusion_matrix.csv")


def save_baseline_results(metrics_df: pd.DataFrame, confusion_df: pd.DataFrame) -> None:
    """Save baseline classifier outputs for Question 2."""
    save_results(metrics_df, confusion_df, experiment_name="baseline")
