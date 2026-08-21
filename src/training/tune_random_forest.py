from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.constants import SEED
from src.training.evaluate_candidates import evaluate_candidates

LOGGER = logging.getLogger("ml_pipeline.training")


def tune_random_forest(
    train_df: pd.DataFrame, folds: list[dict[str, Any]]
) -> tuple[dict[str, Any], pd.DataFrame]:
    LOGGER.info("Tuning Random Forest")
    candidates = [
        {
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "max_features": 0.8,
            "max_depth": None,
        },
        {
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "max_depth": None,
        },
        {
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "max_features": 0.6,
            "max_depth": None,
        },
        {
            "n_estimators": 400,
            "min_samples_leaf": 1,
            "max_features": 0.8,
            "max_depth": 16,
        },
    ]
    results = evaluate_candidates(
        train_df,
        folds,
        candidates,
        lambda params: RandomForestRegressor(random_state=SEED, n_jobs=1, **params),
    )
    best = results.iloc[0]
    parameters = {
        "n_estimators": int(best["n_estimators"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": best["max_features"],
        "max_depth": None if pd.isna(best["max_depth"]) else int(best["max_depth"]),
    }
    LOGGER.info(
        "Random Forest tuning complete; best CV_MAE=%.2f",
        results.iloc[0]["CV_MAE_mean"],
    )
    LOGGER.debug("Random Forest selected parameters=%s", parameters)
    return parameters, results


__all__ = ["tune_random_forest"]
