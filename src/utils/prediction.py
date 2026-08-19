from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def build_prediction_row(
    *,
    job_title: str,
    job_category: str,
    education: str,
    years: int,
    city: str,
    country: str,
    remote: str,
    company: str,
    industry: str,
    demand: int,
    benefits: int,
    selected_skills: Iterable[str],
) -> pd.DataFrame:
    skills = list(dict.fromkeys(selected_skills))
    return pd.DataFrame(
        [
            {
                "job_title": job_title,
                "job_category": job_category,
                "years_of_experience": years,
                "education_required": education,
                "city": city,
                "country": country,
                "remote_work": remote,
                "company_size": company,
                "industry": industry,
                "demand_score": demand,
                "benefits_score_10": benefits,
                "required_skills": "|".join(skills),
                "skill_count": len(skills),
            }
        ]
    )


def prediction_interval(prediction: float, half_width: float) -> tuple[float, float]:
    return max(0.0, prediction - half_width), prediction + half_width
