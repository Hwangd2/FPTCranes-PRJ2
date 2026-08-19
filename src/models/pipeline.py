from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    root: Path
    data: Path
    output: Path
    basic: Path
    ml_ready: Path
    comparison: Path
    best: Path
    prediction: Path
    artifacts: Path
    assets: Path


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    name: str
    estimator_factory: Callable[[], RegressorMixin]
    scale_numeric: bool

    def build_estimator(self) -> RegressorMixin:
        """Construct an unfitted estimator for one training run."""
        return self.estimator_factory()


@dataclass(frozen=True, slots=True)
class TrainingSelection:
    model_name: str
    estimator: RegressorMixin
    scale_numeric: bool
    hyperparameters: dict[str, Any]
    tuning_results: pd.DataFrame


__all__ = ["ModelDefinition", "PipelinePaths", "TrainingSelection"]
