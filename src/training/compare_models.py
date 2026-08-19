from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.constants import MODEL_FEATURES, TARGET
from src.training.build_model_pipeline import build_model_pipeline
from src.training.model_catalog import model_catalog
from src.training.regression_metrics import regression_metrics

LOGGER = logging.getLogger("ml_pipeline.training")


def compare_models(
    train_df: pd.DataFrame, folds: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    catalog = model_catalog()
    LOGGER.info("Comparing %d model families with temporal validation", len(catalog))
    for model_number, definition in enumerate(catalog.values(), 1):
        LOGGER.info(
            "Training model %d/%d: %s",
            model_number,
            len(catalog),
            definition.name,
        )
        model_fold_metrics: list[dict[str, Any]] = []
        for fold in folds:
            training = train_df.loc[fold["train_idx"]]
            validation = train_df.loc[fold["val_idx"]]
            LOGGER.debug(
                "%s fold=%d validation_month=%s train_rows=%d validation_rows=%d",
                definition.name,
                fold["fold"],
                fold["val_month"],
                len(training),
                len(validation),
            )
            pipeline = build_model_pipeline(definition)
            pipeline.fit(training[list(MODEL_FEATURES)], training[TARGET])
            prediction = pipeline.predict(validation[list(MODEL_FEATURES)])
            metrics: dict[str, Any] = regression_metrics(validation[TARGET], prediction)
            metrics.update(
                {
                    "model": definition.name,
                    "fold": fold["fold"],
                    "validation_month": str(fold["val_month"]),
                }
            )
            LOGGER.debug(
                "%s fold=%d metrics MAE=%.2f RMSE=%.2f R2=%.4f",
                definition.name,
                fold["fold"],
                metrics["MAE"],
                metrics["RMSE"],
                metrics["R2"],
            )
            fold_rows.append(metrics)
            model_fold_metrics.append(metrics)

        frame = pd.DataFrame(model_fold_metrics)
        comparison_rows.append(
            {
                "model": definition.name,
                "CV_MAE_mean": frame["MAE"].mean(),
                "CV_MAE_std": frame["MAE"].std(ddof=0),
                "CV_RMSE_mean": frame["RMSE"].mean(),
                "CV_RMSE_std": frame["RMSE"].std(ddof=0),
                "CV_R2_mean": frame["R2"].mean(),
                "CV_R2_std": frame["R2"].std(ddof=0),
                "CV_MedAE_mean": frame["MedAE"].mean(),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["CV_MAE_mean", "CV_RMSE_mean"]
    )
    comparison = comparison.reset_index(drop=True)
    LOGGER.info(
        "Model comparison complete; current leader=%s CV_MAE=%.2f",
        comparison.iloc[0]["model"],
        comparison.iloc[0]["CV_MAE_mean"],
    )
    return pd.DataFrame(fold_rows), comparison


__all__ = ["compare_models"]
