"""Stage 2: Project Scope & Initial Inspection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import TARGET
from src.models import PipelinePaths
from src.utils.pipeline_features import data_dictionary


def project_scope_initial_inspection(raw: pd.DataFrame, paths: PipelinePaths) -> None:
    required = {
        "job_id",
        "job_title",
        "AI Engineering",
        "experience_level",
        "years_of_experience",
        "education_required",
        TARGET,
        "salary_min_usd",
        "salary_max_usd",
        "required_skills",
        "posting_year",
        "posting_month",
        "salary_tier",
    }
    missing_required = sorted(required - set(raw.columns))
    if missing_required:
        raise RuntimeError(f"Required columns missing: {missing_required}")

    data_dictionary(raw).to_csv(paths.basic / "02_data_dictionary_raw.csv", index=False)
    numeric_summary = (
        raw.describe(include=[np.number])
        .T.reset_index()
        .rename(columns={"index": "column"})
    )
    numeric_summary.to_csv(paths.basic / "02_numeric_summary_raw.csv", index=False)
