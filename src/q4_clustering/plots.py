"""Plotting helpers for Question 4."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from src.common.plotting import figure_dir


def save_kmeans_selection_plot(results_df: pd.DataFrame) -> None:
    """Save elbow and silhouette plots for K-Means model selection."""
    output_path = figure_dir("q4") / "kmeans_model_selection.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(results_df["k"], results_df["inertia"], marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_df["k"], results_df["silhouette_score"], marker="o", color="tab:orange")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_kmeans_pca_cluster_plot(x: pd.DataFrame, cluster_labels: pd.Series) -> None:
    """Project the final K-Means clusters to 2D with PCA and save the scatter plot."""
    save_cluster_pca_plot(
        x=x,
        cluster_labels=cluster_labels,
        output_filename="kmeans_clusters_pca_k5.png",
        title="K-Means Clusters (k=5) in PCA 2D Space",
    )


def save_agglomerative_pca_cluster_plot(x: pd.DataFrame, cluster_labels: pd.Series) -> None:
    """Project final Agglomerative clusters to 2D with PCA and save the scatter plot."""
    save_cluster_pca_plot(
        x=x,
        cluster_labels=cluster_labels,
        output_filename="agglomerative_clusters_pca_k5.png",
        title="Agglomerative Clusters (k=5) in PCA 2D Space",
    )


def save_dbscan_pca_cluster_plot(x: pd.DataFrame, cluster_labels: pd.Series) -> None:
    """Project final DBSCAN clusters to 2D with PCA and save the scatter plot."""
    save_cluster_pca_plot(
        x=x,
        cluster_labels=cluster_labels,
        output_filename="dbscan_clusters_pca_eps_1_0.png",
        title="DBSCAN Clusters (eps=1.0, min_samples=5) in PCA 2D Space",
    )


def save_cluster_pca_plot(
    x: pd.DataFrame,
    cluster_labels: pd.Series,
    output_filename: str,
    title: str,
) -> None:
    """Project cluster labels to a 2D PCA embedding and save the scatter plot."""
    output_path = figure_dir("q4") / output_filename

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(x)

    plot_df = pd.DataFrame({"PC1": embedding[:, 0], "PC2": embedding[:, 1], "Cluster": cluster_labels})

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster_id in sorted(plot_df["Cluster"].unique()):
        cluster_points = plot_df[plot_df["Cluster"] == cluster_id]
        ax.scatter(
            cluster_points["PC1"],
            cluster_points["PC2"],
            s=40,
            alpha=0.75,
            label=f"Cluster {cluster_id}",
        )

    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_dbscan_k_distance_plot(k_distance_df: pd.DataFrame, min_samples: int) -> None:
    """Save the sorted k-distance graph for DBSCAN eps tuning."""
    output_path = figure_dir("q4") / f"dbscan_k_distance_min_samples_{min_samples}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_distance_df["PointIndex"], k_distance_df["KDistance"], linewidth=1.8)
    ax.set_title(f"DBSCAN k-Distance Graph (min_samples={min_samples})")
    ax.set_xlabel("Sorted Points")
    ax.set_ylabel(f"Distance to {min_samples}th Nearest Neighbor")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
