"""Stage 3: Data Quality Check."""

from __future__ import annotations

import logging

import pandas as pd

from src.models import PipelinePaths

LOGGER = logging.getLogger("ml_pipeline")


def data_quality_check(raw: pd.DataFrame, paths: PipelinePaths) -> None:
    hidden_missing_tokens = {"", "na", "n/a", "none", "null", "nan", "unknown", "-"}
    quality_rows = []
    for column in raw.columns:
        series = raw[column]
        hidden = 0
        if series.dtype == "object":
            normalized = series.dropna().astype(str).str.strip().str.lower()
            hidden = int(normalized.isin(hidden_missing_tokens).sum())
        quality_rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "hidden_missing_tokens": hidden,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(paths.basic / "03_data_quality_by_column.csv", index=False)
    quality_summary = pd.DataFrame(
        [
            {
                "rows": len(raw),
                "columns": raw.shape[1],
                "missing_cells": int(raw.isna().sum().sum()),
                "hidden_missing_tokens": int(quality["hidden_missing_tokens"].sum()),
                "duplicate_rows": int(raw.duplicated().sum()),
                "unique_job_id": int(raw["job_id"].nunique()),
            }
        ]
    )
    quality_summary.to_csv(paths.basic / "03_data_quality_summary.csv", index=False)
    LOGGER.debug("Data quality summary: %s", quality_summary.iloc[0].to_dict())
