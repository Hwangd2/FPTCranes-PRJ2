from __future__ import annotations

from typing import Final

SEED: Final = 42
TARGET: Final = "annual_salary_usd"
LOCKED_YEAR: Final = 2026
LOCKED_MONTH: Final = 3

EDUCATION_ORDER: Final = (
    "Bootcamp/Self-taught",
    "Associate's",
    "Bachelor's",
    "Master's",
    "PhD",
)

STAGES: Final = (
    "Load Data",
    "Project Scope & Initial Inspection",
    "Data Quality Check",
    "Corrupted Row Removal",
    "Contradictory-Feature Investigation",
    "Feature Selection & Leakage Prevention",
    "Correlation Encoding & Analysis",
    "Train-Test Split",
    "Model Training & Comparison",
    "Best Model Selection & Feature Importance Review",
    "Save Deployable Pipeline + Metadata",
    "Streamlit Salary Prediction Dashboard",
)

STRUCTURED_FEATURES: Final = (
    "job_title",
    "job_category",
    "years_of_experience",
    "education_required",
    "city",
    "country",
    "remote_work",
    "company_size",
    "industry",
    "demand_score",
    "benefits_score_10",
)
MODEL_FEATURES: Final = STRUCTURED_FEATURES + ("required_skills", "skill_count")
NOMINAL_FEATURES: Final = (
    "job_title",
    "job_category",
    "city",
    "country",
    "remote_work",
    "company_size",
    "industry",
)
ORDINAL_FEATURES: Final = ("education_required",)
NUMERIC_FEATURES: Final = (
    "years_of_experience",
    "demand_score",
    "benefits_score_10",
    "skill_count",
)
BLOCKED_FEATURES: Final = (
    "job_id",
    "salary_min_usd",
    "salary_max_usd",
    "salary_tier",
    "experience_level",
    "posting_year",
    "posting_month",
    "is_senior",
    "is_remote_friendly",
    "is_llm_role",
    "ai_salary_premium_pct",
    "demand_growth_yoy_pct",
)

PASTEL: Final = {
    "blue": "#A8DADC",
    "purple": "#CDB4DB",
    "pink": "#FFC8DD",
    "green": "#BDE0A8",
    "yellow": "#FFE5A5",
    "orange": "#FFD6A5",
    "ink": "#24324A",
    "muted": "#64748B",
}

__all__ = [
    "BLOCKED_FEATURES",
    "EDUCATION_ORDER",
    "LOCKED_MONTH",
    "LOCKED_YEAR",
    "MODEL_FEATURES",
    "NOMINAL_FEATURES",
    "NUMERIC_FEATURES",
    "ORDINAL_FEATURES",
    "PASTEL",
    "SEED",
    "STAGES",
    "STRUCTURED_FEATURES",
    "TARGET",
]
