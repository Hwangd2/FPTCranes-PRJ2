from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import Config, resolve_latest_pipeline_image
from src.utils.artifacts import load_csv, load_json
from src.utils.auth import password_hash, resolve_auth_settings
from src.utils.formatting import format_money, model_comparison_comment
from src.utils.prediction import build_prediction_row, prediction_interval


def test_config_resolves_the_existing_pipeline_diagram() -> None:
    assert Config.PIPELINE_IMAGE.is_file()


def test_pipeline_image_resolver_uses_latest_timestamped_asset(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    docs = tmp_path / "docs"
    older = assets / "output-20260819T110000Z"
    latest = assets / "output-20260819T120000Z"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)
    docs.mkdir()
    (older / "12_stage_pipeline.png").write_bytes(b"older")
    (latest / "12_stage_pipeline.png").write_bytes(b"latest")

    assert resolve_latest_pipeline_image(assets, docs) == (
        latest / "12_stage_pipeline.png"
    )


def test_config_resolves_the_existing_presentation_report() -> None:
    assert Config.PRESENTATION_REPORT.is_file()


def test_artifact_loaders_refresh_when_files_change(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metadata.json"
    csv_path.write_text("metric,value\nMAE,10\n", encoding="utf-8")
    json_path.write_text(json.dumps({"model": "first"}), encoding="utf-8")

    assert load_csv(csv_path).iloc[0].to_dict() == {"metric": "MAE", "value": 10}
    assert load_json(json_path) == {"model": "first"}

    csv_path.write_text("metric,value\nMAE,12\nRMSE,15\n", encoding="utf-8")
    json_path.write_text(
        json.dumps({"model": "second", "version": 2}), encoding="utf-8"
    )

    assert load_csv(csv_path)["value"].tolist() == [12, 15]
    assert load_json(json_path) == {"model": "second", "version": 2}


def test_missing_optional_report_artifacts_return_empty_values(tmp_path: Path) -> None:
    assert load_csv(tmp_path / "missing.csv").empty
    assert load_json(tmp_path / "missing.json") == {}


def test_model_comment_handles_zero_one_and_multiple_candidates() -> None:
    assert "not available" in model_comparison_comment(pd.DataFrame()).lower()

    single = pd.DataFrame([{"model": "Ridge", "CV_MAE_mean": 1000.0}])
    assert "only evaluated candidate" in model_comparison_comment(single)

    comparison = pd.DataFrame(
        [
            {"model": "Ridge", "CV_MAE_mean": 1000.0},
            {"model": "Forest", "CV_MAE_mean": 800.0},
        ]
    )
    comment = model_comparison_comment(comparison)
    assert "Forest ranks first" in comment
    assert "20.0% lower" in comment
    assert format_money(1234.5) == "$1,234"


def test_auth_settings_prefer_environment_then_secrets_then_demo() -> None:
    env = {"AIJOB_APP_USER": "env-user", "AIJOB_APP_PASSWORD_SHA256": "ABCDEF"}
    secrets = {"auth": {"username": "secret-user", "password_sha256": "123456"}}

    configured = resolve_auth_settings(env, secrets)
    assert configured.username == "env-user"
    assert configured.password_sha256 == "abcdef"
    assert configured.demo_mode is False

    configured = resolve_auth_settings({}, secrets)
    assert configured.username == "secret-user"
    assert configured.password_sha256 == "123456"
    assert configured.demo_mode is False

    configured = resolve_auth_settings({}, {})
    assert configured.username == "admin"
    assert configured.password_sha256 == password_hash("AIJob2026!")
    assert configured.demo_mode is True


def test_prediction_row_preserves_serving_schema_and_distinct_skills() -> None:
    row = build_prediction_row(
        job_title="ML Engineer",
        job_category="AI",
        education="Master",
        years=4,
        city="Hanoi",
        country="Vietnam",
        remote="Hybrid",
        company="Medium",
        industry="Technology",
        demand=8,
        benefits=7,
        selected_skills=["Python", "SQL", "Python"],
    )

    assert row.iloc[0].to_dict() == {
        "job_title": "ML Engineer",
        "job_category": "AI",
        "years_of_experience": 4,
        "education_required": "Master",
        "city": "Hanoi",
        "country": "Vietnam",
        "remote_work": "Hybrid",
        "company_size": "Medium",
        "industry": "Technology",
        "demand_score": 8,
        "benefits_score_10": 7,
        "required_skills": "Python|SQL",
        "skill_count": 2,
    }
    assert prediction_interval(500.0, 750.0) == (0.0, 1250.0)
