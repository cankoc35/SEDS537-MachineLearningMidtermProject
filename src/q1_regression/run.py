"""Entry point for Question 1."""

from sklearn.datasets import fetch_california_housing

from src.common.config import DATA_DIR
from src.common.metrics import print_section
from src.q1_regression.eda import run_eda
from src.q1_regression.modeling import (
    run_baseline_models,
    run_polynomial_models,
    run_residual_analysis,
    run_scaled_models,
)


def load_california_housing():
    """Load the California Housing dataset as a pandas DataFrame."""
    dataset = fetch_california_housing(as_frame=True)
    return dataset.frame


def main() -> None:
    print_section("Question 1")

    df = load_california_housing()
    output_path = DATA_DIR / "raw" / "california_housing.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Loaded California Housing dataset with shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Saved raw dataset copy to: {output_path}")

    run_eda(df)
    print("Saved EDA outputs to results/figures/q1 and results/tables/q1.")

    run_baseline_models(df)
    print("Saved baseline metrics to results/tables/q1/regression_baseline_metrics.csv.")

    run_scaled_models(df)
    print("Saved scaled metrics to results/tables/q1/regression_scaled_metrics.csv.")

    run_polynomial_models(df)
    print("Saved polynomial metrics to results/tables/q1/regression_polynomial_metrics.csv.")

    run_residual_analysis(df)
    print("Saved residual plots to results/figures/q1/residual_plots.png.")


if __name__ == "__main__":
    main()
