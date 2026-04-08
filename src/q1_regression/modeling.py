"""Regression modeling for Question 1."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.common.config import RANDOM_STATE, RESULTS_DIR
from src.common.plotting import figure_dir
from src.q1_regression.eda import TARGET_COLUMN

import matplotlib.pyplot as plt
import seaborn as sns


TEST_SIZE = 0.20
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
POLYNOMIAL_DEGREE = 2


def run_baseline_models(df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate the baseline regression models."""
    x_train, x_test, y_train, y_test = split_features_and_target(df)
    metrics_df = fit_and_evaluate_models(
        models=build_baseline_models(),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing="none",
    )
    save_metrics(metrics_df, "regression_baseline_metrics.csv")
    print("\nBaseline regression metrics:")
    print(metrics_df.round(4).to_string(index=False))
    return metrics_df


def run_scaled_models(df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate the required models with standard-scaled features."""
    x_train, x_test, y_train, y_test = split_features_and_target(df)
    metrics_df = fit_and_evaluate_models(
        models=build_scaled_models(),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing="standard_scaling",
    )
    save_metrics(metrics_df, "regression_scaled_metrics.csv")
    print("\nStandard-scaled regression metrics:")
    print(metrics_df.round(4).to_string(index=False))
    return metrics_df


def run_polynomial_models(df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate models with degree-2 polynomial features."""
    x_train, x_test, y_train, y_test = split_features_and_target(df)
    metrics_df = fit_and_evaluate_models(
        models=build_polynomial_models(),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing=f"polynomial_degree_{POLYNOMIAL_DEGREE}",
    )
    save_metrics(metrics_df, "regression_polynomial_metrics.csv")
    print("\nPolynomial feature regression metrics:")
    print(metrics_df.round(4).to_string(index=False))
    return metrics_df


def run_residual_analysis(df: pd.DataFrame) -> None:
    """Create fitted-vs-residual plots for representative Q1 models."""
    x_train, x_test, y_train, y_test = split_features_and_target(df)
    models = build_residual_plot_models()

    residual_data = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        residuals = y_test - y_pred
        residual_data.append((model_name, y_pred, residuals))

    all_predictions = np.concatenate([data[1] for data in residual_data])
    all_residuals = np.concatenate([data[2].to_numpy() for data in residual_data])
    x_padding = (all_predictions.max() - all_predictions.min()) * 0.05
    y_limit = np.abs(all_residuals).max() * 1.05

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (model_name, y_pred, residuals) in zip(axes.ravel(), residual_data):
        sns.scatterplot(x=y_pred, y=residuals, ax=ax, s=12, alpha=0.4, edgecolor=None)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_xlim(all_predictions.min() - x_padding, all_predictions.max() + x_padding)
        ax.set_ylim(-y_limit, y_limit)

    plt.suptitle("Fitted Values vs. Residuals", y=1.02)
    plt.tight_layout()
    plt.savefig(figure_dir("q1") / "residual_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def split_features_and_target(df: pd.DataFrame):
    """Create one consistent held-out test split for all Q1 models."""
    x = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def build_baseline_models() -> dict:
    """Create the required baseline models."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": RidgeCV(alphas=RIDGE_ALPHAS, cv=5),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_scaled_models() -> dict:
    """Create the required models with training-only standard scaling."""
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=RIDGE_ALPHAS, cv=5),
        ),
        "Random Forest Regressor": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def build_polynomial_models() -> dict:
    """Create the required models with degree-2 polynomial features."""
    return {
        "Polynomial Linear Regression": make_pipeline(
            PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        ),
        "Polynomial Ridge Regression": make_pipeline(
            PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=RIDGE_ALPHAS, cv=5),
        ),
        "Polynomial Random Forest Regressor": make_pipeline(
            PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def build_residual_plot_models() -> dict:
    """Create representative models for residual diagnostics."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": RidgeCV(alphas=RIDGE_ALPHAS, cv=5),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Polynomial Linear Regression": make_pipeline(
            PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        ),
    }


def fit_and_evaluate_models(
    models: dict,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing: str,
) -> pd.DataFrame:
    """Fit each model and return a sorted metrics table."""
    rows = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        row = evaluate_predictions(model_name, y_test, y_pred)
        row["preprocessing"] = preprocessing
        row["selected_alpha"] = get_selected_alpha(model)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("rmse")


def evaluate_predictions(model_name: str, y_true, y_pred) -> dict:
    """Calculate regression metrics required by the assignment."""
    return {
        "model": model_name,
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "preprocessing": None,
        "selected_alpha": np.nan,
    }


def get_selected_alpha(model) -> float:
    """Return the selected Ridge alpha when the model exposes one."""
    if hasattr(model, "alpha_"):
        return model.alpha_

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "alpha_"):
                return step.alpha_

    return np.nan


def save_metrics(metrics_df: pd.DataFrame, filename: str) -> None:
    """Save model metrics for the report."""
    table_dir = RESULTS_DIR / "tables" / "q1"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(table_dir / filename, index=False)
