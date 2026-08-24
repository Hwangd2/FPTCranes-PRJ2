from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, fmt_money, guard, read_csv, read_json, show_image

BASIC = Config.OUTPUT_DIR / "01_data_basic_clean"


def render() -> None:
    page_header(
        "1. Data basic clean",
        "Stages 01–05 · raw ingestion → scope → quality → confirmed corruption removal → contradiction audit",
        "🧹",
    )
    tabs = st.tabs([
        "01 · Load Data",
        "02 · Scope & Inspection",
        "03 · Data Quality",
        "04 · Corrupted Row",
        "05 · Contradictions",
    ])

    with tabs[0]:
        stage_intro("Stage 01 · Load Data", "Fingerprint the raw CSV and preserve an auditable source snapshot.", "data/raw/ai_jobs_market_2025_2026.csv", "01_raw_manifest.json + 01_raw_preview.csv")
        manifest = read_json(BASIC / "01_raw_manifest.json")
        preview = read_csv(BASIC / "01_raw_preview.csv")
        if not manifest:
            st.warning("Stage 01 outputs are missing. Run `python pipeline.py` first.")
        else:
            cols = st.columns(5)
            cols[0].metric("Rows", f"{manifest.get('rows', 0):,}")
            cols[1].metric("Columns", manifest.get("columns", "—"))
            cols[2].metric("Period", f"{manifest.get('period_min', '—')} → {manifest.get('period_max', '—')}")
            cols[3].metric("File size", f"{manifest.get('file_size_bytes', 0)/1024:.1f} KB")
            cols[4].metric("SHA-256", str(manifest.get("sha256", "—"))[:12] + "…")
            if not preview.empty:
                st.dataframe(preview, use_container_width=True, hide_index=True, height=430)
            evidence("Stage 01 decision", "The source fingerprint, dimensions and preview are persisted before mutation.", "A reproducible pipeline must be able to identify exactly which source snapshot produced the artifacts.", "Carry the raw file unchanged into Stage 02–03; do not clean silently during ingestion.", "good")

    with tabs[1]:
        stage_intro("Stage 02 · Project Scope & Initial Inspection", "Frame the supervised-regression task and inspect schema, dtypes, descriptive statistics and target distribution.", "Raw dataset", "Data dictionary + numeric summary + target profile/outlier review")
        task = read_csv(BASIC / "02_formal_task_definition.csv")
        dictionary = read_csv(BASIC / "02_data_dictionary_raw.csv")
        summary = read_csv(BASIC / "02_numeric_summary_raw.csv")
        profile = read_csv(BASIC / "02_target_profile_raw.csv")
        outliers = read_csv(BASIC / "02_target_outliers_iqr.csv")
        if guard(profile, "Stage 02 outputs are missing. Run `python pipeline.py`."):
            return
        row = profile.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Mean salary", fmt_money(row["mean"]))
        cols[1].metric("Median", fmt_money(row["median"]))
        cols[2].metric("Std. dev.", fmt_money(row["std"]))
        cols[3].metric("Range", f"{fmt_money(row['min'])}–{fmt_money(row['max'])}")
        cols[4].metric("IQR flags", int(row["outlier_count"]))
        show_image(BASIC / "02_target_distribution_raw.png")
        left, right = st.columns([1.1, 1.4])
        with left:
            st.markdown("#### Formal task")
            st.dataframe(task, use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Schema / profiling dictionary")
            st.dataframe(dictionary, use_container_width=True, hide_index=True, height=390)
        with st.expander("Numeric descriptive statistics"):
            st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
        with st.expander(f"IQR-flagged salary records ({len(outliers):,})"):
            st.dataframe(outliers, use_container_width=True, hide_index=True)
        mean_v, median_v = float(row["mean"]), float(row["median"])
        shape_text = "right-tailed tendency (mean above median)" if mean_v > median_v else "no material mean/median right-tail signal"
        evidence(
            "Stage 02 finding",
            f"annual_salary_usd spans {fmt_money(row['min'])}–{fmt_money(row['max'])}; mean {fmt_money(mean_v)}, median {fmt_money(median_v)}; IQR flags {int(row['outlier_count'])} rows.",
            f"The target shows {shape_text}. IQR flags are statistical review signals, not proof of corruption.",
            "Retain salary extremes unless a later integrity rule proves they are invalid. Defer encoding/scaling and target-aware selection until leakage controls and the temporal holdout are governed.",
            "info",
        )

    with tabs[2]:
        stage_intro("Stage 03 · Data Quality Check", "Go beyond nulls/duplicates with hidden-token, cardinality and categorical value-by-value audits.", "Raw dataset", "Column quality profile + categorical value audit")
        quality = read_csv(BASIC / "03_data_quality_by_column.csv")
        summary = read_csv(BASIC / "03_data_quality_summary.csv")
        cat = read_csv(BASIC / "03_categorical_value_audit.csv")
        if guard(summary, "Stage 03 outputs are missing."):
            return
        q = summary.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Missing cells", int(q["missing_cells"]))
        cols[1].metric("Hidden missing", int(q["hidden_missing_tokens"]))
        cols[2].metric("Duplicate rows", int(q["duplicate_rows"]))
        cols[3].metric("Unique job IDs", int(q["unique_job_id"]))
        cols[4].metric("Invalid category rows", int(q["invalid_job_category_token_rows"]))
        left, right = st.columns(2)
        with left:
            show_image(BASIC / "03_missingness_by_feature.png")
        with right:
            show_image(BASIC / "03_high_cardinality_features.png")
        st.markdown("#### Column-level quality profile")
        st.dataframe(quality, use_container_width=True, hide_index=True, height=430)
        if not cat.empty:
            with st.expander("Categorical value-by-value sanity check", expanded=True):
                selected = st.selectbox("Categorical feature", sorted(cat["column"].unique()), key="cat_audit_feature")
                st.dataframe(cat.loc[cat["column"].eq(selected)].sort_values("count", ascending=False), use_container_width=True, hide_index=True, height=320)
        bad = int(q["invalid_job_category_token_rows"])
        evidence("Stage 03 finding", f"Missing={int(q['missing_cells'])}; duplicates={int(q['duplicate_rows'])}; semantic invalid header-token rows={bad}.", "Null/duplicate checks alone would incorrectly label this source as fully clean. Category-level auditing exposes corruption hidden inside an otherwise valid-looking field.", "Remove only the confidently corrupted, very-low-volume row at Stage 04; do not invent a replacement category.", "risk" if bad else "good")

    with tabs[3]:
        stage_intro("Stage 04 · Corrupted Row Removal", "Remove confirmed categorical corruption and create the single canonical basic-clean dataset.", "Raw dataset + Stage 03 evidence", "data_basic_clean.csv")
        summary = read_csv(BASIC / "04_cleaning_summary.csv")
        removed = read_csv(BASIC / "04_corrupted_rows_removed.csv")
        clean = read_csv(BASIC / "data_basic_clean.csv")
        if guard(summary, "Stage 04 outputs are missing."):
            return
        s = summary.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Raw rows", int(s["raw_rows"]))
        cols[1].metric("Removed", int(s["removed_corrupted_rows"]))
        cols[2].metric("Clean rows", int(s["clean_rows"]))
        cols[3].metric("Duplicates after", int(s["duplicate_rows_after"]))
        st.markdown("#### Confirmed removed row")
        st.dataframe(removed, use_container_width=True, hide_index=True)
        with st.expander("Canonical basic-clean preview"):
            st.dataframe(clean.head(30), use_container_width=True, hide_index=True)
        evidence("Stage 04 decision", f"{int(s['removed_corrupted_rows'])} confirmed corrupted row was removed; {int(s['clean_rows'])} canonical rows remain.", "The source column header had leaked into a row value, so imputation would fabricate a domain label.", "Use `data_basic_clean.csv` as the sole downstream source. The original raw file remains unchanged for lineage.", "good")

    with tabs[4]:
        stage_intro("Stage 05 · Contradictory-Feature Investigation", "Test whether related fields tell a coherent story before allowing them into modeling.", "Basic-clean data; target-aware diagnostics use DEV only", "Logic findings + contradiction and salary-integrity evidence")
        issues = read_csv(BASIC / "05_logic_integrity_findings.csv")
        by_level = read_csv(BASIC / "05_salary_by_experience_level_dev.csv")
        by_year = read_csv(BASIC / "05_salary_by_years_dev.csv")
        dep = read_csv(BASIC / "05_functional_dependency_report.csv")
        if guard(issues, "Stage 05 outputs are missing."):
            return
        lookup = issues.set_index("issue")
        def pct(name: str) -> float:
            return float(lookup.loc[name, "affected_pct"]) if name in lookup.index else np.nan
        cols = st.columns(4)
        cols[0].metric("Experience mismatch", f"{pct('experience_bucket_mismatch'):.1f}%")
        cols[1].metric("Salary outside min/max", f"{pct('salary_outside_min_max'):.1f}%")
        cols[2].metric("Salary-tier mismatch", f"{pct('salary_tier_mismatch'):.1f}%")
        cols[3].metric("Duplicate-skill rows", f"{pct('skill_rows_with_duplicate_tokens'):.1f}%")
        show_image(BASIC / "05_logic_issue_rates.png")
        c1, c2 = st.columns(2)
        with c1:
            show_image(BASIC / "05_years_by_experience_level.png")
            show_image(BASIC / "05_salary_by_experience_level.png")
        with c2:
            show_image(BASIC / "05_salary_by_years_experience.png")
            show_image(BASIC / "05_salary_by_job_category.png")
        show_image(BASIC / "05_salary_range_vs_target.png")
        with st.expander("Functional-dependency evidence"):
            st.dataframe(dep, use_container_width=True, hide_index=True)
        level_spread = float(by_level["mean_salary_usd"].max() - by_level["mean_salary_usd"].min()) if not by_level.empty else np.nan
        year_spread = float(by_year["mean_salary_usd"].max() - by_year["mean_salary_usd"].min()) if not by_year.empty else np.nan
        evidence(
            "Stage 05 decision",
            f"Experience semantic mismatch affects {pct('experience_bucket_mismatch'):.1f}% of clean rows. Mean-salary spread by experience bucket is {fmt_money(level_spread)} versus {fmt_money(year_spread)} across numeric years.",
            "The two experience fields cannot both be treated as trustworthy real-world salary signals. Compensation range/tier inconsistencies also make those fields unsafe as model inputs.",
            "Quarantine `experience_level`, block salary_min/max/tier, de-duplicate skills, and test `years_of_experience` only through explicit ablation with cautious interpretation.",
            "risk",
        )
