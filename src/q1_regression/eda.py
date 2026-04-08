"""Exploratory data analysis for Question 1."""

import os

from src.common.config import RESULTS_DIR

MATPLOTLIB_CACHE_DIR = RESULTS_DIR / ".matplotlib_cache"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

FONT_CACHE_DIR = RESULTS_DIR / ".cache"
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.common.plotting import figure_dir


TARGET_COLUMN = "MedHouseVal"


def run_eda(df) -> None:
    """Run required EDA checks and save plots/tables for Question 1."""
    print_dataset_summary(df)
    save_summary_tables(df)
    save_histograms(df)
    save_correlation_matrix(df)
    save_target_scatter_plots(df)


def print_dataset_summary(df) -> None:
    """Print basic dataset information to the console."""
    print("\nDataset preview:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSummary statistics:")
    print(df.describe().round(3))


def save_summary_tables(df) -> None:
    """Save summary statistics and missing value counts as CSV files."""
    table_dir = RESULTS_DIR / "tables" / "q1"
    table_dir.mkdir(parents=True, exist_ok=True)

    df.describe().round(4).to_csv(table_dir / "summary_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv(table_dir / "missing_values.csv")


def save_histograms(df) -> None:
    """Save histograms for all dataset columns."""
    output_dir = figure_dir("q1")
    axes = df.hist(bins=30, figsize=(14, 10), edgecolor="black")

    for ax in axes.ravel():
        ax.set_xlabel(ax.get_title())
        ax.set_ylabel("Count")

    plt.suptitle("California Housing Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "histograms.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_correlation_matrix(df) -> None:
    """Save a correlation matrix heatmap."""
    output_dir = figure_dir("q1")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("California Housing Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_target_scatter_plots(df) -> None:
    """Save scatter plots showing each feature against the target."""
    output_dir = figure_dir("q1")
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, feature in zip(axes.ravel(), feature_columns):
        sns.scatterplot(
            data=df,
            x=feature,
            y=TARGET_COLUMN,
            ax=ax,
            s=10,
            alpha=0.35,
            edgecolor=None,
        )
        ax.set_title(f"{feature} vs. {TARGET_COLUMN}")

    plt.suptitle("Features vs. Median House Value", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_scatter_plots.png", dpi=300, bbox_inches="tight")
    plt.close()
