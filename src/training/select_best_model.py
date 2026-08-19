from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.constants import SEED
from src.models import TrainingSelection
from src.training.model_catalog import model_catalog
from src.training.tune_gradient_boosting import tune_gradient_boosting
from src.training.tune_random_forest import tune_random_forest

LOGGER = logging.getLogger("ml_pipeline.training")


def select_best_model(
    train_df: pd.DataFrame,
    folds: list[dict[str, Any]],
    comparison: pd.DataFrame,
) -> TrainingSelection:
    selected_name = str(comparison.iloc[0]["model"])
    LOGGER.info("Selected model family from temporal CV: %s", selected_name)
    definition = model_catalog()[selected_name]
    if selected_name == "Random Forest":
        parameters, tuning = tune_random_forest(train_df, folds)
        estimator = RandomForestRegressor(random_state=SEED, n_jobs=-1, **parameters)
    elif selected_name == "Gradient Boosting":
        parameters, tuning = tune_gradient_boosting(train_df, folds)
        estimator = GradientBoostingRegressor(random_state=SEED, **parameters)
    else:
        estimator = definition.build_estimator()
        parameters = estimator.get_params()
        tuning = pd.DataFrame(
            [
                {
                    "candidate_id": 1,
                    "CV_MAE_mean": float(comparison.iloc[0]["CV_MAE_mean"]),
                    "note": "No extra tuning grid for this model family.",
                }
            ]
        )
    selection = TrainingSelection(
        model_name=selected_name,
        estimator=estimator,
        scale_numeric=definition.scale_numeric,
        hyperparameters=parameters,
        tuning_results=tuning,
    )
    LOGGER.debug(
        "Training selection estimator=%s scale_numeric=%s hyperparameters=%s",
        type(selection.estimator).__name__,
        selection.scale_numeric,
        selection.hyperparameters,
    )
    return selection


__all__ = ["select_best_model"]
