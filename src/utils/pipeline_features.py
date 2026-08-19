from __future__ import annotations

from typing import Any

import pandas as pd

from src.constants import BLOCKED_FEATURES, MODEL_FEATURES


def normalize_skill_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    seen: set[str] = set()
    normalized: list[str] = []
    for token in str(value).split("|"):
        skill = " ".join(token.strip().split())
        if skill and skill not in seen:
            seen.add(skill)
            normalized.append(skill)
    return "|".join(normalized)


def skill_count(value: Any) -> int:
    text = normalize_skill_string(value)
    return 0 if not text else len(text.split("|"))


def experience_bucket(years: Any) -> str:
    numeric_years = pd.to_numeric(pd.Series([years]), errors="coerce").iloc[0]
    if pd.isna(numeric_years):
        return "Unknown"
    if numeric_years <= 2:
        return "Entry (0-2 yrs)"
    if numeric_years <= 5:
        return "Mid (3-5 yrs)"
    if numeric_years <= 9:
        return "Senior (6-9 yrs)"
    return "Lead (10+ yrs)"


def salary_tier_expected(salary: float) -> str:
    if salary < 100_000:
        return "Entry (<$100k)"
    if salary < 150_000:
        return "Mid ($100-150k)"
    if salary < 200_000:
        return "Upper-Mid ($150-200k)"
    if salary <= 300_000:
        return "Senior ($200-300k)"
    return "Elite (>$300k)"


def data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "sample_values": " | ".join(
                    map(str, series.dropna().astype(str).unique()[:5])
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_policy_table() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for feature in MODEL_FEATURES:
        role = (
            "PHASE2_MULTI_HOT"
            if feature == "required_skills"
            else "ENGINEERED"
            if feature == "skill_count"
            else "KEEP"
        )
        rows.append(
            {
                "feature": feature,
                "policy": "ALLOW",
                "role": role,
                "reason": "Available before target; encoded inside TRAIN-fitted pipeline.",
            }
        )
    for feature in BLOCKED_FEATURES:
        reason = (
            "Target leakage / target-adjacent"
            if feature in {"salary_min_usd", "salary_max_usd", "salary_tier"}
            else "Identifier, redundant/contradictory, derived flag, or non-serving metadata"
        )
        rows.append(
            {
                "feature": feature,
                "policy": "BLOCK",
                "role": "EXCLUDED",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "data_dictionary",
    "experience_bucket",
    "feature_policy_table",
    "normalize_skill_string",
    "salary_tier_expected",
    "skill_count",
]
