"""Stage 12: Streamlit Salary Prediction Dashboard."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import time

import numpy as np
import pandas as pd
from rich.console import Console

from src.constants import TARGET
from src.models import PipelinePaths
from src.utils.pipeline_io import save_json
from src.utils.pipeline_plots import chart_pipeline_12

LOGGER = logging.getLogger("ml_pipeline")


def streamlit_salary_prediction_dashboard(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    dev: pd.DataFrame,
    locked: pd.DataFrame,
    skill_vocabulary: list[str],
    selected_name: str,
    locked_predictions: np.ndarray,
    final_metrics: dict[str, float],
    interval_q90: float,
    paths: PipelinePaths,
    project_root: Path,
    asset_output: Path,
    start: float,
    console: Console,
) -> dict[str, object]:
    predictions = locked[
        ["job_title", "job_category", "city", "country", TARGET]
    ].copy()
    predictions["predicted_salary_usd"] = locked_predictions
    predictions["absolute_error_usd"] = np.abs(
        predictions[TARGET] - predictions["predicted_salary_usd"]
    )
    predictions["prediction_low_90"] = np.maximum(
        0, predictions["predicted_salary_usd"] - interval_q90
    )
    predictions["prediction_high_90"] = (
        predictions["predicted_salary_usd"] + interval_q90
    )
    predictions.to_csv(paths.prediction / "11_locked_test_predictions.csv", index=False)

    pd.DataFrame(
        [
            {
                "model": selected_name,
                "locked_test_rows": len(predictions),
                "prediction_mean_usd": float(
                    predictions["predicted_salary_usd"].mean()
                ),
                "prediction_median_usd": float(
                    predictions["predicted_salary_usd"].median()
                ),
                "interval_half_width_q90_usd": interval_q90,
                **final_metrics,
            }
        ]
    ).to_csv(paths.prediction / "11_prediction_summary.csv", index=False)

    for source_name, destination_name in [
        (
            "actual_vs_predicted_locked_test.png",
            "salary_prediction_actual_vs_predicted.png",
        ),
        ("locked_test_residuals.png", "salary_prediction_residuals.png"),
    ]:
        source = paths.best / source_name
        if source.exists():
            (paths.prediction / destination_name).write_bytes(source.read_bytes())

    pipeline_image = asset_output / "12_stage_pipeline.png"
    chart_pipeline_12(pipeline_image)
    LOGGER.debug("Saved pipeline diagram: %s", pipeline_image)
    report_index = {
        "01_data_basic_clean": str(paths.basic.relative_to(project_root)),
        "02_data_ready_for_machine_learning": str(
            paths.ml_ready.relative_to(project_root)
        ),
        "03_model_comparison": str(paths.comparison.relative_to(project_root)),
        "04_best_model_and_feature_importance": str(
            paths.best.relative_to(project_root)
        ),
        "05_salary_prediction": str(paths.prediction.relative_to(project_root)),
        "model_bundle": str(
            (paths.artifacts / "model_bundle.joblib").relative_to(project_root)
        ),
        "metadata": str((paths.artifacts / "metadata.json").relative_to(project_root)),
        "pipeline_picture": str(pipeline_image.relative_to(project_root)),
    }
    save_json(report_index, paths.output / "report_index.json")

    run_summary = {
        "status": "SUCCESS",
        "elapsed_seconds": round(time.time() - start, 2),
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "train_rows": len(dev),
        "locked_test_rows": len(locked),
        "skill_vocabulary_train": len(skill_vocabulary),
        "selected_model": selected_name,
        "locked_test_metrics": final_metrics,
        "outputs": report_index,
    }
    save_json(run_summary, paths.output / "run_summary.json")
    console.print("\n[bold green]Pipeline completed successfully.[/]")
    console.print_json(json.dumps(run_summary, default=str))
    return run_summary
