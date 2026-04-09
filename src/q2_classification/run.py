"""Entry point for Question 2."""

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.common.config import DATA_DIR, RANDOM_STATE
from src.common.metrics import print_section
from src.q2_classification.evaluation import (
    evaluate_binary_classifier,
    save_baseline_results,
    save_results,
)
from src.q2_classification.modeling import (
    build_logistic_regression_model,
    build_random_forest_model,
    build_xgboost_model,
)
from src.q2_classification.plots import save_roc_curve


TARGET_COLUMN = "Class"
TEST_SIZE = 0.20


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


def class_distribution_table(labels: pd.Series) -> pd.DataFrame:
    """Return class counts and percentages for a label series."""
    class_counts = labels.value_counts().sort_index()
    class_percentages = labels.value_counts(normalize=True).sort_index() * 100

    return pd.DataFrame(
        {
            "class": class_counts.index,
            "count": class_counts.values,
            "percentage": class_percentages.values,
        }
    )


def split_features_and_target(df: pd.DataFrame):
    """Create a stratified train/test split for Question 2."""
    x = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def report_split_distribution(y_train: pd.Series, y_test: pd.Series) -> None:
    """Print train/test class distribution after stratified splitting."""
    train_distribution = class_distribution_table(y_train)
    test_distribution = class_distribution_table(y_test)

    print("\nTrain set distribution:")
    for _, row in train_distribution.iterrows():
        print(
            f"Class {int(row['class'])}: "
            f"{int(row['count'])} samples "
            f"({row['percentage']:.4f}%)"
        )

    print("\nTest set distribution:")
    for _, row in test_distribution.iterrows():
        print(
            f"Class {int(row['class'])}: "
            f"{int(row['count'])} samples "
            f"({row['percentage']:.4f}%)"
        )


def build_baseline_models() -> dict:
    """Create the baseline classifiers used in Question 2."""
    models = {
        "Logistic Regression": build_logistic_regression_model(),
        "Random Forest": build_random_forest_model(),
    }

    try:
        models["XGBoost"] = build_xgboost_model()
    except Exception as exc:
        print("\nXGBoost baseline skipped.")
        print(f"Reason: {exc}")
        print("On macOS, XGBoost may require `brew install libomp` before it can run.")

    return models


def apply_smote(x_train: pd.DataFrame, y_train: pd.Series):
    """Resample the training split with SMOTE only."""
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(x_train, y_train)


def apply_random_undersampling(x_train: pd.DataFrame, y_train: pd.Series):
    """Reduce the majority class on the training split only."""
    undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
    return undersampler.fit_resample(x_train, y_train)


def run_experiment(models: dict, x_train, y_train, x_test, y_test, experiment_name: str) -> None:
    """Train and evaluate a set of classifiers for one experiment setting."""
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]

        metrics_df, confusion_df = evaluate_binary_classifier(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
        )

        if experiment_name == "baseline":
            save_baseline_results(metrics_df, confusion_df)
        else:
            save_results(metrics_df, confusion_df, experiment_name=experiment_name)

        print(f"\n{model_name} {experiment_name} metrics:")
        print(metrics_df.round(4).to_string(index=False))
        print("\nConfusion matrix:")
        print(confusion_df)

        if experiment_name == "baseline" and model_name == "Random Forest":
            save_roc_curve(
                y_true=y_test,
                y_score=y_score,
                model_name="Random Forest",
                filename="random_forest_roc_curve.png",
            )


def main() -> None:
    print_section("Question 2")

    df = load_credit_card_data()
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nSummary statistics:")
    print(df.describe().round(4))

    print("\nMissing values:")
    print(df.isna().sum())

    report_class_distribution(df)

    x_train, x_test, y_train, y_test = split_features_and_target(df)
    print(f"\nTraining set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    report_split_distribution(y_train, y_test)

    baseline_models = build_baseline_models()
    run_experiment(baseline_models, x_train, y_train, x_test, y_test, experiment_name="baseline")

    x_train_smote, y_train_smote = apply_smote(x_train, y_train)
    print(f"\nSMOTE training set shape: {x_train_smote.shape}")

    print("\nSMOTE training set distribution:")
    smote_distribution = class_distribution_table(y_train_smote)
    for _, row in smote_distribution.iterrows():
        print(
            f"Class {int(row['class'])}: "
            f"{int(row['count'])} samples "
            f"({row['percentage']:.4f}%)"
        )

    smote_models = build_baseline_models()
    run_experiment(smote_models, x_train_smote, y_train_smote, x_test, y_test, experiment_name="smote")

    x_train_under, y_train_under = apply_random_undersampling(x_train, y_train)
    print(f"\nRandom undersampling training set shape: {x_train_under.shape}")

    print("\nRandom undersampling training set distribution:")
    under_distribution = class_distribution_table(y_train_under)
    for _, row in under_distribution.iterrows():
        print(
            f"Class {int(row['class'])}: "
            f"{int(row['count'])} samples "
            f"({row['percentage']:.4f}%)"
        )

    undersampling_models = build_baseline_models()
    run_experiment(
        undersampling_models,
        x_train_under,
        y_train_under,
        x_test,
        y_test,
        experiment_name="undersampling",
    )


if __name__ == "__main__":
    main()
