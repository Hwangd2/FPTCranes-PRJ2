"""Stage 7: Correlation Encoding & Analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.constants import MODEL_FEATURES, TARGET
from src.models import PipelinePaths
from src.training import build_preprocessor
from src.utils.pipeline_features import skill_count
from src.utils.pipeline_plots import plot_correlation

LOGGER = logging.getLogger("ml_pipeline")


def correlation_encoding_analysis(
    clean: pd.DataFrame,
    dev: pd.DataFrame,
    locked: pd.DataFrame,
    paths: PipelinePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dev = dev.copy()
    locked = locked.copy()
    dev["skill_count"] = dev["required_skills"].map(skill_count)
    locked["skill_count"] = locked["required_skills"].map(skill_count)
    clean_with_skill = clean.copy()
    clean_with_skill["skill_count"] = clean_with_skill["required_skills"].map(
        skill_count
    )
    clean_with_skill.to_csv(
        paths.ml_ready / "data_full_with_normalized_skills_and_skill_count.csv",
        index=False,
    )

    analysis_pre = build_preprocessor(scale_numeric=False)
    x_dev_encoded = analysis_pre.fit_transform(dev[list(MODEL_FEATURES)])
    encoded_names = analysis_pre.get_feature_names_out()
    skill_vocab = list(
        analysis_pre.named_transformers_["skills"].get_feature_names_out()
    )
    pd.DataFrame({"skill_token": skill_vocab}).to_csv(
        paths.ml_ready / "skill_vocabulary_train_only.csv", index=False
    )
    if len(skill_vocab) != 93:
        LOGGER.warning(
            "TRAIN skill vocabulary contains %d tokens; current supplied snapshot is expected to contain 93",
            len(skill_vocab),
        )
    LOGGER.debug(
        "Development encoding shape=%s skill_vocabulary=%d",
        x_dev_encoded.shape,
        len(skill_vocab),
    )

    correlation_rows = []
    y_dev = dev[TARGET].to_numpy(dtype=float)
    for index, name in enumerate(encoded_names):
        values = x_dev_encoded[:, index]
        correlation = (
            0.0 if np.nanstd(values) == 0 else float(np.corrcoef(values, y_dev)[0, 1])
        )
        correlation_rows.append(
            {
                "encoded_feature": name,
                "pearson_r": correlation,
                "abs_r": abs(correlation),
            }
        )
    correlations = (
        pd.DataFrame(correlation_rows)
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )
    correlations.to_csv(
        paths.ml_ready / "07_train_encoded_feature_target_correlation.csv",
        index=False,
    )
    plot_correlation(correlations, paths.ml_ready)

    skill_frequency_rows = []
    for token in skill_vocab:
        mask = dev["required_skills"].str.split("|").map(lambda values: token in values)
        salaries = dev.loc[mask, TARGET]
        skill_frequency_rows.append(
            {
                "skill_token": token,
                "record_count": int(mask.sum()),
                "mean_salary_usd": float(salaries.mean()) if len(salaries) else np.nan,
                "median_salary_usd": float(salaries.median())
                if len(salaries)
                else np.nan,
            }
        )
    pd.DataFrame(skill_frequency_rows).sort_values(
        "record_count", ascending=False
    ).to_csv(paths.ml_ready / "07_skill_frequency_and_salary_train.csv", index=False)
    return dev, locked, skill_vocab
