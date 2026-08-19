"""Stage 4: Corrupted Row Removal."""

from __future__ import annotations

import logging

import pandas as pd

from src.constants import LOCKED_MONTH, LOCKED_YEAR
from src.models import PipelinePaths
from src.utils.pipeline_features import normalize_skill_string

LOGGER = logging.getLogger("ml_pipeline")


def corrupted_row_removal(
    raw: pd.DataFrame, paths: PipelinePaths
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    corruption_mask = raw["AI Engineering"].astype(str).str.strip().eq("job_category")
    raw.loc[corruption_mask].copy().to_csv(
        paths.basic / "04_corrupted_rows_removed.csv", index=False
    )
    clean = (
        raw.loc[~corruption_mask]
        .copy()
        .rename(columns={"AI Engineering": "job_category"})
    )
    for column in clean.select_dtypes(include="object").columns:
        clean[column] = clean[column].astype(str).str.strip()
    clean["required_skills"] = clean["required_skills"].map(normalize_skill_string)
    clean.reset_index(drop=True, inplace=True)
    clean.to_csv(paths.basic / "data_basic_clean.csv", index=False)
    LOGGER.debug(
        "Cleaned rows raw=%d corrupted=%d clean=%d",
        len(raw),
        int(corruption_mask.sum()),
        len(clean),
    )

    locked_mask = (clean["posting_year"].astype(int) == LOCKED_YEAR) & (
        clean["posting_month"].astype(int) == LOCKED_MONTH
    )
    dev = clean.loc[~locked_mask].copy()
    locked = clean.loc[locked_mask].copy()
    if dev.empty or locked.empty:
        raise RuntimeError(
            "Temporal split would create an empty development or locked-test partition."
        )
    return clean, dev, locked, corruption_mask
