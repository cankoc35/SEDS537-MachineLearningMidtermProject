"""Evaluation helpers for Question 5."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true, y_pred) -> pd.DataFrame:
    """Return accuracy and macro-F1 in a small tabular format."""
    return pd.DataFrame(
        [
            {
                "Accuracy": accuracy_score(y_true, y_pred),
                "MacroF1": f1_score(y_true, y_pred, average="macro"),
            }
        ]
    ).round(4)


def confusion_matrix_frame(y_true, y_pred, class_names: list[str]) -> pd.DataFrame:
    """Return the confusion matrix as a labeled DataFrame."""
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(matrix, index=class_names, columns=class_names)
