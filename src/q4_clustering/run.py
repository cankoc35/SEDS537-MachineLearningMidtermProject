"""Entry point for Question 4."""

import pandas as pd

from src.common.config import DATA_DIR
from src.common.metrics import print_section
from src.q4_clustering.preprocessing import prepare_clustering_features


DATASET_FILENAME = "Mall_Customers.csv"


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


if __name__ == "__main__":
    main()
