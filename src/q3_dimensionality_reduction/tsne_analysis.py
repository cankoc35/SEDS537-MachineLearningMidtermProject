"""t-SNE helpers for Question 3."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from src.common.config import RANDOM_STATE
from src.q3_dimensionality_reduction.plots import save_embedding_scatter

TSNE_SAMPLE_SIZE = 20000


def select_tsne_subset_indices(y: np.ndarray) -> np.ndarray:
    """Return fixed stratified indices for tractable Q3 t-SNE experiments."""
    all_indices = np.arange(len(y))
    if len(all_indices) <= TSNE_SAMPLE_SIZE:
        return all_indices

    selected_indices, _ = train_test_split(
        all_indices,
        train_size=TSNE_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return selected_indices


def select_tsne_subset(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take a stratified subset for tractable t-SNE visualisation on MNIST."""
    selected_indices = select_tsne_subset_indices(y)
    return x[selected_indices], y[selected_indices]


def run_tsne_embedding(x: np.ndarray, y: np.ndarray, perplexity: int) -> tuple[np.ndarray, np.ndarray]:
    """Run a 2D t-SNE embedding for one perplexity value and save the plot."""
    x_subset, y_subset = select_tsne_subset(x, y)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        method="barnes_hut",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    embedding = tsne.fit_transform(x_subset)

    save_embedding_scatter(
        embedding=embedding,
        labels=y_subset,
        filename=f"tsne_2d_perplexity_{perplexity}.png",
        title=(
            f"MNIST t-SNE 2D Embedding (Perplexity = {perplexity}, "
            f"{len(x_subset):,} stratified samples)"
        ),
    )

    return embedding, y_subset
