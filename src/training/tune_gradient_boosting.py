from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.constants import SEED
from src.training.evaluate_candidates import evaluate_candidates

LOGGER = logging.getLogger("ml_pipeline.training")


def tune_gradient_boosting(
    train_df: pd.DataFrame, folds: list[dict[str, Any]]
) -> tuple[dict[str, Any], pd.DataFrame]:
    LOGGER.info("Tuning Gradient Boosting")
    candidates = [
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2, "loss": "huber"},
        {"n_estimators": 250, "learning_rate": 0.04, "max_depth": 2, "loss": "huber"},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "loss": "huber"},
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 2,
            "loss": "squared_error",
        },
    ]
    results = evaluate_candidates(
        train_df,
        folds,
        candidates,
        lambda params: GradientBoostingRegressor(random_state=SEED, **params),
    )
    best = results.iloc[0]
    parameters = {
        "n_estimators": int(best["n_estimators"]),
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "loss": str(best["loss"]),
    }
    LOGGER.info(
        "Gradient Boosting tuning complete; best CV_MAE=%.2f",
        results.iloc[0]["CV_MAE_mean"],
    )
    LOGGER.debug("Gradient Boosting selected parameters=%s", parameters)
    return parameters, results


__all__ = ["tune_gradient_boosting"]
