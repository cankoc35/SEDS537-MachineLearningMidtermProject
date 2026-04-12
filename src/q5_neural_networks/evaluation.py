"""Evaluation helpers for Question 5."""

import pandas as pd
import torch
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


@torch.no_grad()
def predict_labels(model, data_loader, device: torch.device) -> tuple[list[int], list[int]]:
    """Collect ground-truth and predicted labels from a dataloader."""
    model.eval()
    y_true = []
    y_pred = []

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        predictions = logits.argmax(dim=1).cpu().tolist()

        y_true.extend(targets.tolist())
        y_pred.extend(predictions)

    return y_true, y_pred


@torch.no_grad()
def collect_misclassified_examples(
    model,
    data_loader,
    device: torch.device,
    max_examples: int = 5,
) -> list[dict]:
    """Collect a small set of misclassified examples from a dataloader."""
    model.eval()
    examples = []

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        predictions = logits.argmax(dim=1).cpu()

        for image, true_label, predicted_label in zip(inputs.cpu(), targets.cpu(), predictions):
            if true_label.item() != predicted_label.item():
                examples.append(
                    {
                        "image": image,
                        "true_label": true_label.item(),
                        "predicted_label": predicted_label.item(),
                    }
                )
                if len(examples) >= max_examples:
                    return examples

    return examples
