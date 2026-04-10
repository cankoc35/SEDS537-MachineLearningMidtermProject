"""PCA helpers for Question 3."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.common.config import RESULTS_DIR, RANDOM_STATE
from src.q3_dimensionality_reduction.plots import save_embedding_scatter


def run_pca_analysis(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run PCA for 2D and 50D representations and save the required outputs."""
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    x_pca_2d = pca_2d.fit_transform(x)

    pca_50d = PCA(n_components=50, random_state=RANDOM_STATE)
    x_pca_50d = pca_50d.fit_transform(x)

    explained_variance_df = pd.DataFrame(
        {
            "component": np.arange(1, 11),
            "explained_variance_ratio": pca_50d.explained_variance_ratio_[:10],
            "cumulative_explained_variance_ratio": np.cumsum(pca_50d.explained_variance_ratio_[:10]),
        }
    )

    table_dir = RESULTS_DIR / "tables" / "q3"
    table_dir.mkdir(parents=True, exist_ok=True)
    explained_variance_df.to_csv(table_dir / "pca_explained_variance_ratio.csv", index=False)

    save_embedding_scatter(
        embedding=x_pca_2d,
        labels=y,
        filename="pca_2d_embedding.png",
        title="MNIST PCA 2D Embedding",
    )

    return x_pca_2d, x_pca_50d, explained_variance_df


def explained_variance_table_path() -> Path:
    """Return the saved PCA explained variance table path."""
    return RESULTS_DIR / "tables" / "q3" / "pca_explained_variance_ratio.csv"
