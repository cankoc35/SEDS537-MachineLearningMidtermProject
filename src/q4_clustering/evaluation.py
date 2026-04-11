"""Cluster evaluation helpers for Question 4."""

import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.common.config import RESULTS_DIR


def evaluate_clustering(
    x: pd.DataFrame,
    cluster_labels: pd.Series,
    method_name: str,
) -> pd.DataFrame:
    """Compute and save core internal clustering metrics."""
    metrics_df = pd.DataFrame(
        [
            {
                "Method": method_name,
                "SilhouetteScore": silhouette_score(x, cluster_labels),
                "DaviesBouldinIndex": davies_bouldin_score(x, cluster_labels),
            }
        ]
    ).round(4)

    table_dir = RESULTS_DIR / "tables" / "q4"
    table_dir.mkdir(parents=True, exist_ok=True)
    safe_name = method_name.lower().replace(" ", "_")
    metrics_df.to_csv(table_dir / f"{safe_name}_metrics.csv", index=False)

    return metrics_df
