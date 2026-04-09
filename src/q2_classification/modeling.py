"""Model definitions for Question 2."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import RANDOM_STATE


def build_logistic_regression_model():
    """Create the baseline Logistic Regression classifier."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            # Increase iterations so optimization converges reliably.
            max_iter=1000,
            # Fix randomness for reproducible results.
            random_state=RANDOM_STATE,
        ),
    )


def build_random_forest_model():
    """Create the baseline Random Forest classifier."""
    return RandomForestClassifier(
        # Use 200 trees for a stable baseline.
        n_estimators=200,
        # Fix randomness for reproducible results.
        random_state=RANDOM_STATE,
        # Use all available CPU cores during training.
        n_jobs=-1,
    )


def build_xgboost_model():
    """Create the baseline XGBoost classifier."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        # Use a moderate number of trees for the baseline.
        n_estimators=200,
        # Keep the baseline tree depth modest.
        max_depth=6,
        # Standard shrinkage rate for boosting.
        learning_rate=0.1,
        # Fix randomness for reproducible results.
        random_state=RANDOM_STATE,
        # Binary classification with probability output.
        objective="binary:logistic",
        # Evaluate with log loss during training.
        eval_metric="logloss",
        # Use all available CPU cores during training.
        n_jobs=-1,
    )
