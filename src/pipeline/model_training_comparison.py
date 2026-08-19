"""Stage 9: Model Training & Comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models import PipelinePaths
from src.training import compare_models, monthly_temporal_folds
from src.utils.pipeline_plots import plot_model_comparison


def model_training_comparison(
    dev: pd.DataFrame, paths: PipelinePaths
) -> tuple[pd.DataFrame, list[Any], pd.DataFrame]:
    dev = dev.sort_values(["posting_year", "posting_month"]).reset_index(drop=True)
    folds = monthly_temporal_folds(dev, n_folds=5)
    fold_metrics, comparison = compare_models(dev, folds)
    fold_metrics.to_csv(
        paths.comparison / "09_model_comparison_fold_metrics.csv", index=False
    )
    comparison.to_csv(
        paths.comparison / "09_model_comparison_temporal_cv.csv", index=False
    )
    plot_model_comparison(comparison, paths.comparison)
    return dev, folds, comparison
