from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.constants import MODEL_FEATURES, TARGET
from src.models import TrainingSelection
from src.training.build_pipeline import build_pipeline

LOGGER = logging.getLogger("ml_pipeline.training")


def out_of_fold_absolute_errors(
    train_df: pd.DataFrame,
    folds: list[dict[str, Any]],
    selection: TrainingSelection,
) -> list[float]:
    LOGGER.info("Building out-of-fold residual distribution")
    errors: list[float] = []
    for fold in folds:
        training = train_df.loc[fold["train_idx"]]
        validation = train_df.loc[fold["val_idx"]]
        pipeline = build_pipeline(clone(selection.estimator), selection.scale_numeric)
        pipeline.fit(training[list(MODEL_FEATURES)], training[TARGET])
        prediction = pipeline.predict(validation[list(MODEL_FEATURES)])
        fold_errors = np.abs(validation[TARGET].to_numpy() - prediction).tolist()
        errors.extend(fold_errors)
        LOGGER.debug(
            "Residual fold=%d validation_month=%s rows=%d mean_abs_error=%.2f",
            fold["fold"],
            fold["val_month"],
            len(fold_errors),
            float(np.mean(fold_errors)),
        )
    LOGGER.info("Out-of-fold residual distribution contains %d errors", len(errors))
    return errors


__all__ = ["out_of_fold_absolute_errors"]
