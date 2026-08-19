from __future__ import annotations

import logging

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from src.training.build_preprocessor import build_preprocessor

LOGGER = logging.getLogger("ml_pipeline.training")


def build_pipeline(estimator: RegressorMixin, scale_numeric: bool) -> Pipeline:
    """Build the preprocessing-and-estimator pipeline for a training run."""
    LOGGER.debug(
        "Building training pipeline estimator=%s scale_numeric=%s",
        type(estimator).__name__,
        scale_numeric,
    )
    return Pipeline(
        [("preprocessor", build_preprocessor(scale_numeric)), ("model", estimator)]
    )


__all__ = ["build_pipeline"]
