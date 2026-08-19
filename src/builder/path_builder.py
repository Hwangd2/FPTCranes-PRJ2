from __future__ import annotations

from pathlib import Path

from src.models import PipelinePaths


def build_paths(project_root: Path, output_root: Path) -> PipelinePaths:
    """Build the pipeline path object and ensure writable output directories exist."""
    paths = PipelinePaths(
        root=project_root,
        data=project_root / "data",
        output=output_root,
        basic=output_root / "01_data_basic_clean",
        ml_ready=output_root / "02_data_ready_for_machine_learning",
        comparison=output_root / "03_model_comparison",
        best=output_root / "04_best_model_and_feature_importance",
        prediction=output_root / "05_salary_prediction",
        artifacts=project_root / "artifacts",
        assets=project_root / "outputs",
    )
    for directory in (
        paths.output,
        paths.basic,
        paths.ml_ready,
        paths.comparison,
        paths.best,
        paths.prediction,
        paths.artifacts,
        paths.assets,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = ["build_paths"]
