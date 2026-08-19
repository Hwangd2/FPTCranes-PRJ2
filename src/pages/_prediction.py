from __future__ import annotations

import streamlit as st

from src.components._header import page_header
from src.config import Config
from src.utils.artifacts import load_csv, load_json, load_model
from src.utils.formatting import format_money
from src.utils.prediction import build_prediction_row, prediction_interval

page_header(
    "5. AI market job salary prediction",
    "Run inference with the exact preprocessing and model bundle evaluated offline.",
    "paid",
)

metadata = load_json(Config.ARTIFACT_DIR / "metadata.json")
categories = metadata["category_options"]
ranges = metadata["numeric_ranges"]
skills = metadata["skill_vocabulary"]

with st.form("prediction_form"):
    first, second, third = st.columns(3)
    with first:
        job_title = st.selectbox("Job title", categories["job_title"])
        job_category = st.selectbox("Job category", categories["job_category"])
        education = st.selectbox("Education required", metadata["education_order"])
        years = st.slider(
            "Years of experience",
            int(ranges["years_of_experience"]["min"]),
            int(ranges["years_of_experience"]["max"]),
            int(round(ranges["years_of_experience"]["median"])),
        )
    with second:
        city = st.selectbox("City", categories["city"])
        country = st.selectbox("Country", categories["country"])
        remote = st.selectbox("Remote work", categories["remote_work"])
        company = st.selectbox("Company size", categories["company_size"])
    with third:
        industry = st.selectbox("Industry", categories["industry"])
        demand = st.slider(
            "Demand score",
            int(ranges["demand_score"]["min"]),
            int(ranges["demand_score"]["max"]),
            int(round(ranges["demand_score"]["median"])),
        )
        benefits = st.slider(
            "Benefits score / 10",
            int(ranges["benefits_score_10"]["min"]),
            int(ranges["benefits_score_10"]["max"]),
            int(round(ranges["benefits_score_10"]["median"])),
        )
        selected_skills = st.multiselect("Required skills", skills, default=[])

    submitted = st.form_submit_button(
        "Predict salary",
        type="primary",
        icon=":material/calculate:",
        width="stretch",
    )

if submitted:
    row = build_prediction_row(
        job_title=job_title,
        job_category=job_category,
        education=education,
        years=years,
        city=city,
        country=country,
        remote=remote,
        company=company,
        industry=industry,
        demand=demand,
        benefits=benefits,
        selected_skills=selected_skills,
    )
    try:
        model = load_model(Config.ARTIFACT_DIR / "model_bundle.joblib")
        prediction = float(model.predict(row[metadata["model_features"]])[0])
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        st.error(
            f"Prediction artifacts are unavailable or incompatible: {error}. Run "
            "`python pipeline.py` to regenerate them.",
            icon=":material/error:",
        )
    else:
        half_width = float(metadata["prediction_interval_abs_error_q90"])
        lower, upper = prediction_interval(prediction, half_width)
        columns = st.columns(3)
        columns[0].metric("Predicted annual salary", format_money(prediction))
        columns[1].metric("Practical lower bound", format_money(lower))
        columns[2].metric("Practical upper bound", format_money(upper))
        st.success(
            f"Model: **{metadata['model_name']}**. The interval is an empirical "
            "±90th-percentile absolute validation error, not a formal guarantee.",
            icon=":material/check_circle:",
        )

examples = st.expander(
    "Final locked-test prediction examples",
    icon=":material/table_view:",
    on_change="rerun",
)
if examples.open:
    with examples:
        history = load_csv(
            Config.OUTPUT_DIR / "05_salary_prediction/11_locked_test_predictions.csv"
        )
        st.dataframe(
            history.head(30),
            width="stretch",
            hide_index=True,
            key="prediction_examples",
        )
