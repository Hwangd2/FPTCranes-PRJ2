"""Stage 6: Feature Selection & Leakage Prevention."""

from __future__ import annotations

import logging

from src.constants import BLOCKED_FEATURES, MODEL_FEATURES
from src.models import PipelinePaths
from src.utils.pipeline_features import feature_policy_table

LOGGER = logging.getLogger("ml_pipeline")


def feature_selection_leakage_prevention(paths: PipelinePaths) -> None:
    feature_policy_table().to_csv(paths.ml_ready / "06_feature_policy.csv", index=False)
    leakage_gate = set(MODEL_FEATURES).intersection(BLOCKED_FEATURES)
    if leakage_gate:
        raise RuntimeError(
            f"Leakage gate failed; blocked features entered X: {sorted(leakage_gate)}"
        )
    LOGGER.debug(
        "Leakage gate passed model_features=%d blocked_features=%d",
        len(MODEL_FEATURES),
        len(BLOCKED_FEATURES),
    )
