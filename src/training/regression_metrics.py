from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

LOGGER = logging.getLogger("ml_pipeline.training")


def regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
    }
    LOGGER.debug(
        "Regression metrics rows=%d MAE=%.2f RMSE=%.2f R2=%.4f MedAE=%.2f",
        len(y_true),
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
        metrics["MedAE"],
    )
    return metrics


__all__ = ["regression_metrics"]
