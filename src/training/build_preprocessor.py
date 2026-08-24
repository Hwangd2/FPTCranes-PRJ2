from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from src.constants import COMPANY_SIZE_ORDER, EDUCATION_ORDER


NOMINAL_CANDIDATES = [
    "job_title",
    "job_category",
    "city",
    "country",
    "remote_work",
    "industry",
]
NUMERIC_CANDIDATES = [
    "years_of_experience",
    "demand_score",
    "benefits_score_10",
    "skill_count",
]


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(feature_columns: list[str], scale_numeric: bool) -> ColumnTransformer:
    transformers: list[tuple[str, Any, Any]] = []

    nominal = [c for c in NOMINAL_CANDIDATES if c in feature_columns]
    if nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", one_hot_encoder()),
                    ]
                ),
                nominal,
            )
        )

    if "education_required" in feature_columns:
        education = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[EDUCATION_ORDER],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("education", education, ["education_required"]))

    if "company_size" in feature_columns:
        company = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[COMPANY_SIZE_ORDER],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("company_size", company, ["company_size"]))

    # experience_level is used only in the explicit A2 ablation experiment.
    if "experience_level" in feature_columns:
        experience = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("experience_level", experience, ["experience_level"]))

    numeric = [c for c in NUMERIC_CANDIDATES if c in feature_columns]
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", RobustScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))

    if "required_skills" in feature_columns:
        # Scalar column selector intentionally passes a 1-D sequence to CountVectorizer.
        skills = CountVectorizer(
            token_pattern=r"[^|]+",
            lowercase=False,
            binary=True,
        )
        transformers.append(("skills", skills, "required_skills"))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
