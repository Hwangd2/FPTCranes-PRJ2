from __future__ import annotations

import streamlit as st

from src.components._header import page_header
from src.config import Config
from src.utils.artifacts import load_csv

page_header(
    "1. Data basic clean",
    "Validate structure, detect hidden corruption, and produce a canonical clean dataset.",
    "cleaning_services",
)

output = Config.OUTPUT_DIR / "01_data_basic_clean"
quality = load_csv(output / "03_data_quality_summary.csv")
issues = load_csv(output / "05_logic_integrity_findings.csv")
corrupted = load_csv(output / "04_corrupted_rows_removed.csv")

if not quality.empty:
    row = quality.iloc[0]
    columns = st.columns(4)
    columns[0].metric("Rows", f"{int(row['rows']):,}")
    columns[1].metric("Columns", int(row["columns"]))
    columns[2].metric("Missing cells", int(row["missing_cells"]))
    columns[3].metric("Duplicate rows", int(row["duplicate_rows"]))

if not corrupted.empty:
    st.warning(
        f"Detected and removed {len(corrupted)} corrupted record(s). The hidden categorical "
        "value `job_category` appeared inside the original AI Engineering column.",
        icon=":material/warning:",
    )

left, right = st.columns(2)
with left:
    chart = output / "logic_issue_rates.png"
    if chart.is_file():
        st.image(str(chart), width="stretch")
with right:
    chart = output / "salary_distribution.png"
    if chart.is_file():
        st.image(str(chart), width="stretch")

st.subheader("Logic and integrity findings", anchor=False)
st.dataframe(issues, width="stretch", hide_index=True, key="data_clean_issues")
if not issues.empty:
    worst = issues.sort_values("affected_pct", ascending=False).iloc[0]
    st.warning(
        f"The largest integrity issue is **{worst['issue']}**, affecting "
        f"{worst['affected_pct']:.1f}% of clean rows. The pipeline blocks or ablates affected "
        "fields instead of treating a null-free dataset as automatically trustworthy.",
        icon=":material/policy:",
    )

preview = st.expander(
    "Preview and download clean data",
    icon=":material/table_view:",
    on_change="rerun",
)
if preview.open:
    with preview:
        clean = load_csv(output / "data_basic_clean.csv")
        st.dataframe(
            clean.head(50), width="stretch", hide_index=True, key="clean_preview"
        )
        st.download_button(
            "Download basic-clean data",
            data=clean.to_csv(index=False).encode("utf-8"),
            file_name="data_basic_clean.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
