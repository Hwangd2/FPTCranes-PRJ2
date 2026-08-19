from __future__ import annotations

import streamlit as st

from src.components._header import page_header
from src.config import Config

page_header(
    "12-stage pipeline",
    "The technical design sequence for leakage-safe validation and deployable artifacts.",
    "account_tree",
)

pipeline_image = Config.PIPELINE_IMAGE
if pipeline_image.is_file():
    st.image(str(pipeline_image), width="stretch")

stages = [
    "1. Load data",
    "2. Project scope and initial inspection",
    "3. Data quality check",
    "4. Corrupted-row removal",
    "5. Contradictory-feature investigation",
    "6. Feature selection and leakage prevention",
    "7. Correlation encoding and analysis",
    "8. Train-test split",
    "9. Model training and comparison",
    "10. Best-model selection and feature review",
    "11. Save deployable pipeline and metadata",
    "12. Streamlit salary prediction dashboard",
]

for start in range(0, len(stages), 3):
    for column, stage in zip(st.columns(3), stages[start : start + 3], strict=True):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{stage}**")
