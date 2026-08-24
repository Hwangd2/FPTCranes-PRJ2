from __future__ import annotations

import logging

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from src.constants import SEED
from src.models import ModelDefinition

LOGGER = logging.getLogger("ml_pipeline.training")


def model_catalog() -> dict[str, ModelDefinition]:
    """Return constructor-based definitions so every fit receives a fresh estimator."""
    definitions = (
        ModelDefinition(
            "Dummy (Median)",
            lambda: DummyRegressor(strategy="median"),
            False,
        ),
        ModelDefinition("Linear Regression", LinearRegression, True),
        ModelDefinition("Ridge Regression", lambda: Ridge(alpha=10.0), True),
        ModelDefinition(
            "Random Forest",
            lambda: RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=SEED,
                n_jobs=-1,
            ),
            False,
        ),
        ModelDefinition(
            "Gradient Boosting",
            lambda: GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                loss="huber",
                random_state=SEED,
            ),
            False,
        ),
        ModelDefinition(
            "SVR (RBF)",
            lambda: SVR(C=100.0, epsilon=0.1, gamma="scale"),
            True,
        ),
    )
    catalog = {definition.name: definition for definition in definitions}
    LOGGER.info("Model catalog prepared: %d candidates (%s)", len(catalog), ", ".join(catalog))
    return catalog


__all__ = ["model_catalog"]
