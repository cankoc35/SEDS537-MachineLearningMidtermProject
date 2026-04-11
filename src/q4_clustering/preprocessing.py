"""Preprocessing helpers for Question 4."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


IDENTIFIER_COLUMN = "CustomerID"
GENDER_COLUMN = "Gender"


def prepare_clustering_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop identifiers, encode Gender, and scale features for clustering."""
    feature_df = df.drop(columns=[IDENTIFIER_COLUMN]).copy()

    # Encode Gender as a binary feature so it can be used in distance-based models.
    feature_df[GENDER_COLUMN] = feature_df[GENDER_COLUMN].map({"Male": 1, "Female": 0})

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_df.columns, index=df.index)

    return feature_df, scaled_df
