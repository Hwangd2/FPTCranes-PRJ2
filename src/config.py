from pathlib import Path


def resolve_latest_pipeline_image(asset_dir: Path, doc_dir: Path) -> Path:
    """Resolve the newest timestamped pipeline diagram with legacy fallbacks."""
    timestamped = sorted(
        (
            directory / "12_stage_pipeline.png"
            for directory in asset_dir.glob("output-*")
            if directory.is_dir()
        ),
        reverse=True,
    )
    return next(
        (
            path
            for path in (
                *timestamped,
                asset_dir / "12_stage_pipeline.png",
                doc_dir / "12_stage_pipeline.png",
            )
            if path.is_file()
        ),
        asset_dir / "output-latest" / "12_stage_pipeline.png",
    )


class Config:
    ROOT_DIR = Path(__file__).parent.parent
    STREAMLIT_DIR = ROOT_DIR / ".streamlit"
    ASSET_DIR = ROOT_DIR / "assets"
    DATA_DIR = ROOT_DIR / "data"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    ARTIFACT_DIR = ROOT_DIR / "artifacts"
    REPORT_DIR = ROOT_DIR / "reports"
    DOC_DIR = ROOT_DIR / "docs"
    PIPELINE_IMAGE = resolve_latest_pipeline_image(ASSET_DIR, DOC_DIR)
    PRESENTATION_REPORT = next(
        (
            path
            for path in (
                REPORT_DIR / "AI_Job_Market_Salary_Prediction_Report.docx",
                DOC_DIR / "AI_Job_Market_Salary_Prediction_Report.docx",
            )
            if path.is_file()
        ),
        REPORT_DIR / "AI_Job_Market_Salary_Prediction_Report.docx",
    )
