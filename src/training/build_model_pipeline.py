from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.models import ModelDefinition
from src.training.build_pipeline import build_pipeline


def build_model_pipeline(definition: ModelDefinition) -> Pipeline:
    """Build a fresh pipeline from a constructor-based model definition."""
    return build_pipeline(definition.build_estimator(), definition.scale_numeric)


__all__ = ["build_model_pipeline"]
