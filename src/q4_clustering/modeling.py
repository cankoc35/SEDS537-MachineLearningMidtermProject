"""Clustering model helpers for Question 4."""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.common.config import RANDOM_STATE, RESULTS_DIR


def run_kmeans_model_selection(x, k_values=range(2, 11)) -> pd.DataFrame:
    """Evaluate K-Means across candidate k values using inertia and silhouette score."""
    results = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        cluster_labels = model.fit_predict(x)

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(x, cluster_labels),
            }
        )

    results_df = pd.DataFrame(results)

    table_dir = RESULTS_DIR / "tables" / "q4"
    table_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_dir / "kmeans_model_selection.csv", index=False)

    return results_df


def run_final_kmeans(
    original_df: pd.DataFrame,
    x: pd.DataFrame,
    n_clusters: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the final K-Means model and save labeled rows and cluster summaries."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    cluster_labels = model.fit_predict(x)

    labeled_df, cluster_summary_df = build_cluster_outputs(original_df, cluster_labels)

    table_dir = RESULTS_DIR / "tables" / "q4"
    table_dir.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(table_dir / "kmeans_cluster_assignments_k5.csv", index=False)
    cluster_summary_df.to_csv(table_dir / "kmeans_cluster_summary_k5.csv", index=False)

    return labeled_df, cluster_summary_df


def run_final_agglomerative(
    original_df: pd.DataFrame,
    x: pd.DataFrame,
    n_clusters: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit final Agglomerative Clustering with Ward linkage and save outputs."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )
    cluster_labels = model.fit_predict(x)

    labeled_df, cluster_summary_df = build_cluster_outputs(original_df, cluster_labels)

    table_dir = RESULTS_DIR / "tables" / "q4"
    table_dir.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(table_dir / "agglomerative_cluster_assignments_k5.csv", index=False)
    cluster_summary_df.to_csv(table_dir / "agglomerative_cluster_summary_k5.csv", index=False)

    return labeled_df, cluster_summary_df


def run_final_dbscan(
    original_df: pd.DataFrame,
    x: pd.DataFrame,
    eps: float = 1.0,
    min_samples: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit final DBSCAN and save labeled rows and cluster summaries."""
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    cluster_labels = model.fit_predict(x)

    labeled_df, cluster_summary_df = build_cluster_outputs(original_df, cluster_labels)

    table_dir = RESULTS_DIR / "tables" / "q4"
    table_dir.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(table_dir / "dbscan_cluster_assignments_eps_1_0.csv", index=False)
    cluster_summary_df.to_csv(table_dir / "dbscan_cluster_summary_eps_1_0.csv", index=False)

    return labeled_df, cluster_summary_df


def build_cluster_outputs(
    original_df: pd.DataFrame,
    cluster_labels,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach cluster labels and compute cluster-level summary statistics."""
    labeled_df = original_df.copy()
    labeled_df["Cluster"] = cluster_labels

    cluster_summary_df = (
        labeled_df.assign(IsFemale=(labeled_df["Gender"] == "Female").astype(int))
        .groupby("Cluster")
        .agg(
            CustomerCount=("CustomerID", "count"),
            FemaleShare=("IsFemale", "mean"),
            AvgAge=("Age", "mean"),
            AvgAnnualIncomeK=("Annual Income (k$)", "mean"),
            AvgSpendingScore=("Spending Score (1-100)", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("Cluster")
    )

    return labeled_df, cluster_summary_df


def compute_dbscan_k_distance_curve(
    x: pd.DataFrame,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Compute sorted k-distances for DBSCAN eps selection."""
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(x)
    distances, _ = neighbors.kneighbors(x)

    k_distances = sorted(distances[:, -1])
    k_distance_df = pd.DataFrame(
        {
            "PointIndex": range(1, len(k_distances) + 1),
            "KDistance": k_distances,
        }
    )

    table_dir = RESULTS_DIR / "tables" / "q4"
    table_dir.mkdir(parents=True, exist_ok=True)
    k_distance_df.to_csv(
        table_dir / f"dbscan_k_distance_min_samples_{min_samples}.csv",
        index=False,
    )

    return k_distance_df
