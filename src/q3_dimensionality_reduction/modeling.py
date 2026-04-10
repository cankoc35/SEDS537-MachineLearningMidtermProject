"""k-NN modeling helpers for Question 3."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from src.common.config import RESULTS_DIR, RANDOM_STATE


def run_knn_accuracy_comparison(representations: dict[str, tuple]) -> pd.DataFrame:
    """Compare k-NN accuracy across multiple feature representations."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    results = []
    for representation_name, (features, representation_labels) in representations.items():
        scores = cross_val_score(
            model,
            features,
            representation_labels,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )
        results.append(
            {
                "representation": representation_name,
                "num_samples": len(representation_labels),
                "num_features": features.shape[1],
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="mean_accuracy", ascending=False)
    table_dir = RESULTS_DIR / "tables" / "q3"
    table_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_dir / "knn_accuracy_comparison.csv", index=False)
    return results_df
