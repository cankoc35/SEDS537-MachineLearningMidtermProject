"""Entry point for Question 4."""

import pandas as pd

from src.common.config import DATA_DIR
from src.common.metrics import print_section
from src.q4_clustering.evaluation import evaluate_clustering
from src.q4_clustering.modeling import (
    run_final_agglomerative,
    run_final_kmeans,
    run_kmeans_model_selection,
)
from src.q4_clustering.plots import (
    save_agglomerative_pca_cluster_plot,
    save_kmeans_pca_cluster_plot,
    save_kmeans_selection_plot,
)
from src.q4_clustering.preprocessing import prepare_clustering_features


DATASET_FILENAME = "Mall_Customers.csv"
FINAL_KMEANS_K = 5
FINAL_AGGLOMERATIVE_K = 5


def load_mall_customer_data() -> pd.DataFrame:
    """Load the Mall Customer Segmentation dataset."""
    data_path = DATA_DIR / "raw" / DATASET_FILENAME
    return pd.read_csv(data_path)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print the first-pass dataset inspection for Question 4."""
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSummary statistics:")
    print(df.describe(include="all").transpose())


def main() -> None:
    print_section("Question 4")

    df = load_mall_customer_data()
    inspect_dataset(df)

    feature_df, scaled_df = prepare_clustering_features(df)
    print("\nClustering features after dropping CustomerID and encoding Gender:")
    print(feature_df.head())

    print("\nScaled clustering feature matrix:")
    print(scaled_df.head())
    print(f"\nProcessed feature matrix shape: {scaled_df.shape}")

    kmeans_results_df = run_kmeans_model_selection(scaled_df)
    print("\nK-Means model selection results:")
    print(kmeans_results_df.round(4).to_string(index=False))
    save_kmeans_selection_plot(kmeans_results_df)
    print("\nSaved K-Means elbow and silhouette plot to:")
    print("results/figures/q4/kmeans_model_selection.png")

    labeled_df, cluster_summary_df = run_final_kmeans(
        original_df=df,
        x=scaled_df,
        n_clusters=FINAL_KMEANS_K,
    )
    print(f"\nFinal K-Means cluster counts for k={FINAL_KMEANS_K}:")
    print(labeled_df["Cluster"].value_counts().sort_index().to_string())

    print(f"\nFinal K-Means cluster summary for k={FINAL_KMEANS_K}:")
    print(cluster_summary_df.to_string(index=False))

    print("\nSaved final K-Means outputs to:")
    print("results/tables/q4/kmeans_cluster_assignments_k5.csv")
    print("results/tables/q4/kmeans_cluster_summary_k5.csv")

    save_kmeans_pca_cluster_plot(scaled_df, labeled_df["Cluster"])
    print("\nSaved final K-Means PCA cluster plot to:")
    print("results/figures/q4/kmeans_clusters_pca_k5.png")

    agglomerative_labeled_df, agglomerative_summary_df = run_final_agglomerative(
        original_df=df,
        x=scaled_df,
        n_clusters=FINAL_AGGLOMERATIVE_K,
    )
    print(f"\nAgglomerative cluster counts for k={FINAL_AGGLOMERATIVE_K}:")
    print(agglomerative_labeled_df["Cluster"].value_counts().sort_index().to_string())

    print(f"\nAgglomerative cluster summary for k={FINAL_AGGLOMERATIVE_K}:")
    print(agglomerative_summary_df.to_string(index=False))

    agglomerative_metrics_df = evaluate_clustering(
        x=scaled_df,
        cluster_labels=agglomerative_labeled_df["Cluster"],
        method_name="Agglomerative",
    )
    print("\nAgglomerative internal evaluation metrics:")
    print(agglomerative_metrics_df.to_string(index=False))

    print("\nSaved Agglomerative outputs to:")
    print("results/tables/q4/agglomerative_cluster_assignments_k5.csv")
    print("results/tables/q4/agglomerative_cluster_summary_k5.csv")
    print("results/tables/q4/agglomerative_metrics.csv")

    save_agglomerative_pca_cluster_plot(scaled_df, agglomerative_labeled_df["Cluster"])
    print("\nSaved Agglomerative PCA cluster plot to:")
    print("results/figures/q4/agglomerative_clusters_pca_k5.png")


if __name__ == "__main__":
    main()
