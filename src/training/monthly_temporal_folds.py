from __future__ import annotations

import logging
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("ml_pipeline.training")


def monthly_temporal_folds(
    train_df: pd.DataFrame, n_folds: int = 5
) -> list[dict[str, Any]]:
    month_key = train_df["posting_year"].astype(int) * 100 + train_df[
        "posting_month"
    ].astype(int)
    months = sorted(month_key.unique().tolist())
    if len(months) < n_folds + 2:
        raise RuntimeError("Not enough distinct development months for temporal CV.")

    folds: list[dict[str, Any]] = []
    for number, validation_month in enumerate(months[-n_folds:], 1):
        folds.append(
            {
                "fold": number,
                "val_month": int(validation_month),
                "train_idx": train_df.index[month_key < validation_month].to_numpy(),
                "val_idx": train_df.index[month_key == validation_month].to_numpy(),
            }
        )
    LOGGER.info("Prepared %d expanding-window temporal folds", len(folds))
    LOGGER.debug("Validation months: %s", [fold["val_month"] for fold in folds])
    return folds


__all__ = ["monthly_temporal_folds"]
