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
    unique_labels = sorted(pd.Series(cluster_labels).unique())
    non_noise_labels = [label for label in unique_labels if label != -1]
    noise_points = int((pd.Series(cluster_labels) == -1).sum())

    silhouette_value = None
    davies_bouldin_value = None

    if len(non_noise_labels) >= 2:
        silhouette_value = silhouette_score(x, cluster_labels)
        davies_bouldin_value = davies_bouldin_score(x, cluster_labels)

    metrics_df = pd.DataFrame(
        [
            {
                "Method": method_name,
                "ClusterCountExcludingNoise": len(non_noise_labels),
                "NoisePoints": noise_points,
                "SilhouetteScore": silhouette_value,
                "DaviesBouldinIndex": davies_bouldin_value,
            }
        ]
    ).round(4)

    table_dir = RESULTS_DIR / "tables" / "q4"
    table_dir.mkdir(parents=True, exist_ok=True)
    safe_name = method_name.lower().replace(" ", "_")
    metrics_df.to_csv(table_dir / f"{safe_name}_metrics.csv", index=False)

    return metrics_df
