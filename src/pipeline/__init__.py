"""AI Job Market Salary Prediction - 12-stage reproducible pipeline.

Stage 1 intentionally remains in this orchestration module because it owns input
resolution and data loading. Stages 2-12 live in focused package modules.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import platform
import time
import warnings
from datetime import datetime, timezone

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler

from src.pipeline.best_model_selection_feature_importance_review import (
    best_model_selection_feature_importance_review,
)
from src.pipeline.contradictory_feature_investigation import (
    contradictory_feature_investigation,
)
from src.pipeline.correlation_encoding_analysis import correlation_encoding_analysis
from src.pipeline.corrupted_row_removal import corrupted_row_removal
from src.pipeline.data_quality_check import data_quality_check
from src.pipeline.feature_selection_leakage_prevention import (
    feature_selection_leakage_prevention,
)
from src.pipeline.model_training_comparison import model_training_comparison
from src.pipeline.project_scope_initial_inspection import (
    project_scope_initial_inspection,
)
from src.pipeline.save_deployable_pipeline_metadata import (
    save_deployable_pipeline_metadata,
)
from src.pipeline.streamlit_salary_prediction_dashboard import (
    streamlit_salary_prediction_dashboard,
)
from src.pipeline.train_test_split import train_test_split
from src.builder import build_paths
from src.constants import STAGES, TARGET
from src.utils.pipeline_io import save_json, sha256_file

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER_NAME = "ml_pipeline"
CONSOLE = Console()
LOGGER = logging.getLogger(LOGGER_NAME)
STAGE_DEBUG_DETAILS = (
    "Resolve the dataset, load the CSV, and record provenance.",
    "Validate required columns and export schema and numeric summaries.",
    "Measure missing values, hidden missing tokens, uniqueness, and duplicates.",
    "Remove corrupted rows and create the canonical clean dataset.",
    "Investigate contradictory fields and identifier-ordering artifacts.",
    "Apply the feature policy and enforce the target-leakage gate.",
    "Fit development-only encoding and calculate diagnostic correlations.",
    "Create the temporal split and fit preprocessing on development data only.",
    "Compare five model families with expanding-window temporal validation.",
    "Tune the selected family, score the locked test, and compute importance.",
    "Serialize the fitted inference pipeline and its metadata contract.",
    "Export prediction evidence, report indexes, and the pipeline diagram.",
)


def configure_logging(level: str, console: Console = CONSOLE) -> logging.Logger:
    """Configure Rich terminal logging for the pipeline and training child loggers."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    return logger


def build_asset_output_dir(asset_root: Path, timestamp: str) -> Path:
    """Create the timestamped directory for generated asset images."""
    output = asset_root / f"output-{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def stage_header(number: int, title: str) -> None:
    CONSOLE.rule(f"[bold cyan]STAGE {number:02d}[/] - {title}")
    LOGGER.info("Stage %02d started: %s", number, title)
    LOGGER.debug("Stage %02d detail: %s", number, STAGE_DEBUG_DETAILS[number - 1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Job Market Salary Prediction - 12-stage pipeline"
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/ai_jobs_market_2025_2026.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--log-level",
        choices=("INFO", "DEBUG"),
        default="INFO",
        help="INFO shows stage/model progress; DEBUG adds fold, shape, and artifact details.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    project_root = PROJECT_ROOT
    data_path = args.data if args.data.is_absolute() else project_root / args.data
    output_root = (
        args.output if args.output.is_absolute() else project_root / args.output
    )
    paths = build_paths(project_root, output_root)
    start = time.time()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    asset_output = build_asset_output_dir(paths.assets, run_timestamp)
    LOGGER.debug(
        "Resolved inputs data=%s output=%s asset_output=%s",
        data_path,
        output_root,
        asset_output,
    )

    # Stage 1 stays here: resolve, load, and record source-data provenance.
    stage_header(1, STAGES[0])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    raw = pd.read_csv(data_path, low_memory=False)
    manifest = {
        "file": str(data_path.name),
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "file_size_bytes": int(data_path.stat().st_size),
        "sha256": sha256_file(data_path),
        "target": TARGET,
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    save_json(manifest, paths.basic / "01_raw_manifest.json")
    LOGGER.debug(
        "Loaded dataset rows=%d columns=%d sha256=%s",
        len(raw),
        raw.shape[1],
        manifest["sha256"],
    )

    stage_header(2, STAGES[1])
    project_scope_initial_inspection(raw, paths)
    stage_header(3, STAGES[2])
    data_quality_check(raw, paths)
    stage_header(4, STAGES[3])
    clean, dev, locked, corruption_mask = corrupted_row_removal(raw, paths)
    stage_header(5, STAGES[4])
    contradictory_feature_investigation(raw, clean, corruption_mask, paths)
    stage_header(6, STAGES[5])
    feature_selection_leakage_prevention(paths)
    stage_header(7, STAGES[6])
    dev, locked, skill_vocabulary = correlation_encoding_analysis(
        clean, dev, locked, paths
    )
    stage_header(8, STAGES[7])
    train_test_split(clean, dev, locked, paths)
    stage_header(9, STAGES[8])
    dev, folds, comparison = model_training_comparison(dev, paths)
    stage_header(10, STAGES[9])
    (
        selected_name,
        best_pipeline,
        locked_predictions,
        final_metrics,
        fitted_preprocessor,
        selection,
        interval_q90,
    ) = best_model_selection_feature_importance_review(
        dev, locked, folds, comparison, paths
    )
    stage_header(11, STAGES[10])
    save_deployable_pipeline_metadata(
        dev,
        locked,
        selected_name,
        best_pipeline,
        fitted_preprocessor,
        selection,
        interval_q90,
        final_metrics,
        paths,
    )
    stage_header(12, STAGES[11])
    streamlit_salary_prediction_dashboard(
        raw,
        clean,
        dev,
        locked,
        skill_vocabulary,
        selected_name,
        locked_predictions,
        final_metrics,
        interval_q90,
        paths,
        project_root,
        asset_output,
        start,
        CONSOLE,
    )


__all__ = [
    "PROJECT_ROOT",
    "build_asset_output_dir",
    "configure_logging",
    "main",
    "best_model_selection_feature_importance_review",
    "contradictory_feature_investigation",
    "correlation_encoding_analysis",
    "corrupted_row_removal",
    "data_quality_check",
    "feature_selection_leakage_prevention",
    "model_training_comparison",
    "project_scope_initial_inspection",
    "save_deployable_pipeline_metadata",
    "stage_header",
    "streamlit_salary_prediction_dashboard",
    "train_test_split",
]
