"""Entry point for Question 1."""

from sklearn.datasets import fetch_california_housing

from src.common.config import DATA_DIR
from src.common.metrics import print_section
from src.q1_regression.eda import run_eda


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


if __name__ == "__main__":
    main()
