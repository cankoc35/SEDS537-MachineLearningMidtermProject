"""Entry point for Question 2."""

import pandas as pd

from src.common.config import DATA_DIR
from src.common.metrics import print_section


TARGET_COLUMN = "Class"


def load_credit_card_data() -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    data_path = DATA_DIR / "raw" / "creditcard.csv"
    return pd.read_csv(data_path)


def report_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return and print class counts and percentages."""
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    class_percentages = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

    distribution_df = pd.DataFrame(
        {
            "class": class_counts.index,
            "count": class_counts.values,
            "percentage": class_percentages.values,
        }
    )

    print("\nClass distribution:")
    for _, row in distribution_df.iterrows():
        print(
            f"Class {int(row['class'])}: "
            f"{int(row['count'])} samples "
            f"({row['percentage']:.4f}%)"
        )

    return distribution_df


def main() -> None:
    print_section("Question 2")

    df = load_credit_card_data()
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"First 5 rows:\n{df.head()}")
    print(f"Info:\n{df.info()}")
    report_class_distribution(df)


if __name__ == "__main__":
    main()
