"""Stage 5: Contradictory-Feature Investigation."""

from __future__ import annotations

import logging

import pandas as pd

from src.constants import TARGET
from src.models import PipelinePaths
from src.utils.pipeline_features import experience_bucket, salary_tier_expected
from src.utils.pipeline_io import save_json
from src.utils.pipeline_plots import plot_basic_outputs

LOGGER = logging.getLogger("ml_pipeline")


def contradictory_feature_investigation(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    corruption_mask: pd.Series,
    paths: PipelinePaths,
) -> None:
    expected_exp = clean["years_of_experience"].map(experience_bucket)
    exp_mismatch = clean["experience_level"].astype(str).ne(expected_exp.astype(str))
    range_bad = ~clean[TARGET].between(
        clean["salary_min_usd"], clean["salary_max_usd"], inclusive="both"
    )
    tier_expected = clean[TARGET].map(salary_tier_expected)
    tier_bad = clean["salary_tier"].astype(str).ne(tier_expected.astype(str))
    job_suffix = pd.to_numeric(
        clean["job_id"].str.extract(r"(\d+)$")[0], errors="coerce"
    )
    job_id_r = float(job_suffix.corr(clean[TARGET]))
    repeated_raw_skills = (
        raw.loc[~corruption_mask, "required_skills"]
        .fillna("")
        .map(
            lambda value: (
                len([item.strip() for item in str(value).split("|") if item.strip()])
                != len(
                    set(item.strip() for item in str(value).split("|") if item.strip())
                )
            )
        )
    )
    issues = pd.DataFrame(
        [
            {
                "issue": "experience_bucket_mismatch",
                "affected_rows": int(exp_mismatch.sum()),
                "total_rows": len(clean),
                "affected_pct": float(exp_mismatch.mean() * 100),
                "action": "Drop experience_level from primary model; retain years_of_experience as an ablation-sensitive numeric feature.",
            },
            {
                "issue": "salary_outside_min_max",
                "affected_rows": int(range_bad.sum()),
                "total_rows": len(clean),
                "affected_pct": float(range_bad.mean() * 100),
                "action": "Block salary_min_usd and salary_max_usd from X.",
            },
            {
                "issue": "salary_tier_mismatch",
                "affected_rows": int(tier_bad.sum()),
                "total_rows": len(clean),
                "affected_pct": float(tier_bad.mean() * 100),
                "action": "Block salary_tier from X.",
            },
            {
                "issue": "skill_rows_with_duplicate_tokens",
                "affected_rows": int(repeated_raw_skills.sum()),
                "total_rows": len(clean),
                "affected_pct": float(repeated_raw_skills.mean() * 100),
                "action": "De-duplicate skill tokens within each row before multi-hot encoding.",
            },
        ]
    )
    issues.to_csv(paths.basic / "05_logic_integrity_findings.csv", index=False)
    LOGGER.debug(
        "Logic-integrity findings: %s",
        issues[["issue", "affected_rows"]].to_dict(orient="records"),
    )
    save_json(
        {
            "job_id_suffix_target_pearson_r": job_id_r,
            "interpretation": "Strong identifier ordering artifact; job_id is strictly blocked.",
        },
        paths.basic / "05_identifier_ordering_artifact.json",
    )
    clean.groupby("experience_level")[TARGET].agg(
        ["count", "mean", "median"]
    ).reset_index().to_csv(
        paths.basic / "05_salary_by_experience_level.csv", index=False
    )
    clean.groupby("years_of_experience")[TARGET].agg(
        ["count", "mean", "median"]
    ).reset_index().to_csv(
        paths.basic / "05_salary_by_years_of_experience.csv", index=False
    )
    plot_basic_outputs(clean, issues, paths.basic)
