"""Stage 11: Save Deployable Pipeline + Metadata."""

from __future__ import annotations

from typing import Any
import logging

import joblib
import pandas as pd

from src.constants import (
    BLOCKED_FEATURES,
    EDUCATION_ORDER,
    LOCKED_MONTH,
    LOCKED_YEAR,
    MODEL_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    STRUCTURED_FEATURES,
    TARGET,
)
from src.models import PipelinePaths, TrainingSelection
from src.utils.pipeline_io import save_json

LOGGER = logging.getLogger("ml_pipeline")


def save_deployable_pipeline_metadata(
    dev: pd.DataFrame,
    locked: pd.DataFrame,
    selected_name: str,
    best_pipeline: Any,
    fitted_preprocessor: Any,
    selection: TrainingSelection,
    interval_q90: float,
    final_metrics: dict[str, float],
    paths: PipelinePaths,
) -> None:
    joblib.dump(best_pipeline, paths.artifacts / "model_bundle.joblib")
    LOGGER.debug("Saved model bundle: %s", paths.artifacts / "model_bundle.joblib")
    category_options = {
        column: sorted(dev[column].dropna().astype(str).unique().tolist())
        for column in NOMINAL_FEATURES + ORDINAL_FEATURES
    }
    numeric_ranges = {
        column: {
            "min": float(dev[column].min()),
            "max": float(dev[column].max()),
            "median": float(dev[column].median()),
        }
        for column in NUMERIC_FEATURES
        if column != "skill_count"
    }
    skill_vocabulary = list(
        fitted_preprocessor.named_transformers_["skills"].get_feature_names_out()
    )
    metadata = {
        "project_name": "AI Job Market Salary Prediction",
        "target": TARGET,
        "model_name": selected_name,
        "model_features": MODEL_FEATURES,
        "structured_features": STRUCTURED_FEATURES,
        "blocked_features": BLOCKED_FEATURES,
        "locked_test": {
            "year": LOCKED_YEAR,
            "month": LOCKED_MONTH,
            "rows": len(locked),
        },
        "development_rows": len(dev),
        "category_options": category_options,
        "numeric_ranges": numeric_ranges,
        "education_order": EDUCATION_ORDER,
        "skill_vocabulary": skill_vocabulary,
        "skill_vocabulary_count": len(skill_vocabulary),
        "prediction_interval_abs_error_q90": interval_q90,
        "final_locked_test_metrics": final_metrics,
        "selected_hyperparameters": selection.hyperparameters,
        "limitations": [
            "Dataset contains strong synthetic-looking ordering and contradictory relationships.",
            "Feature importance explains the fitted model, not causal salary economics.",
            "External validation on verified live job postings is required before operational salary decisions.",
        ],
        "demo_login": {
            "username": "admin",
            "password": "AIJob2026!",
            "warning": "Local demo only; override with environment variables or Streamlit secrets before deployment.",
        },
    }
    save_json(metadata, paths.artifacts / "metadata.json")
    LOGGER.debug("Saved model metadata: %s", paths.artifacts / "metadata.json")
