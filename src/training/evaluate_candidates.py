from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error

from src.constants import MODEL_FEATURES, TARGET
from src.training.build_pipeline import build_pipeline

LOGGER = logging.getLogger("ml_pipeline.training")


def evaluate_candidates(
    train_df: pd.DataFrame,
    folds: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    estimator_factory: Callable[[dict[str, Any]], RegressorMixin],
) -> pd.DataFrame:
    """Evaluate one model family's parameter candidates with temporal folds."""
    rows: list[dict[str, Any]] = []
    for candidate_id, parameters in enumerate(candidates, 1):
        LOGGER.info(
            "Tuning candidate %d/%d parameters=%s",
            candidate_id,
            len(candidates),
            parameters,
        )
        fold_mae: list[float] = []
        for fold in folds:
            training = train_df.loc[fold["train_idx"]]
            validation = train_df.loc[fold["val_idx"]]
            pipeline = build_pipeline(estimator_factory(parameters), False)
            pipeline.fit(training[list(MODEL_FEATURES)], training[TARGET])
            prediction = pipeline.predict(validation[list(MODEL_FEATURES)])
            mae = mean_absolute_error(validation[TARGET], prediction)
            fold_mae.append(mae)
            LOGGER.debug(
                "Candidate %d fold=%d validation_month=%s MAE=%.2f",
                candidate_id,
                fold["fold"],
                fold["val_month"],
                mae,
            )
        rows.append(
            {
                "candidate_id": candidate_id,
                **parameters,
                "CV_MAE_mean": float(np.mean(fold_mae)),
                "CV_MAE_std": float(np.std(fold_mae)),
            }
        )
    return pd.DataFrame(rows).sort_values("CV_MAE_mean").reset_index(drop=True)


__all__ = ["evaluate_candidates"]
