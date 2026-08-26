from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.constants import (
    EDUCATION_ORDER,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)
from src.training.one_hot_encoder import one_hot_encoder

LOGGER = logging.getLogger("ml_pipeline.training")


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """Custom Transformer: Tự động chuẩn hóa tên cột thô trước khi ném vào ColumnTransformer."""
    def __init__(self, rename_mapping: dict[str, str]):
        self.rename_mapping = rename_mapping

    def fit(self, X: pd.DataFrame, y: Any = None) -> ColumnRenamer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.rename(columns=self.rename_mapping)


def build_preprocessor(scale_numeric: bool) -> Pipeline:
    """Construct an unfitted, leakage-safe training feature preprocessor."""
    LOGGER.debug("Building preprocessor scale_numeric=%s", scale_numeric)
    
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    education = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[list(EDUCATION_ORDER)],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    skills = CountVectorizer(token_pattern=r"[^|]+", lowercase=False, binary=True)

    # Core Transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ("nominal", one_hot_encoder(), list(NOMINAL_FEATURES)),
            ("education", education, list(ORDINAL_FEATURES)),
            ("numeric", Pipeline(steps=numeric_steps), list(NUMERIC_FEATURES)),
            ("skills", skills, "required_skills"),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )

    # Đóng gói Renamer và ColumnTransformer thành 1 Pipeline tổng
    return Pipeline(
        steps=[
            ("rename_legacy_columns", ColumnRenamer({"AI Engineering": "job_category"})),
            ("features", column_transformer)
        ]
    )


__all__ = ["build_preprocessor", "ColumnRenamer"]
