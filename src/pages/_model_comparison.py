from __future__ import annotations

import streamlit as st

from src.components._header import page_header
from src.config import Config
from src.utils.artifacts import load_csv
from src.utils.formatting import model_comparison_comment

page_header(
    "3. Model comparison",
    "Compare five regression families using expanding temporal validation on development data.",
    "analytics",
)

output = Config.OUTPUT_DIR / "03_model_comparison"
comparison = load_csv(output / "09_model_comparison_temporal_cv.csv")
if comparison.empty:
    st.warning("No model comparison outputs. Run `python pipeline.py` first.")
else:
    st.dataframe(
        comparison.style.format(
            {
                "CV_MAE_mean": "${:,.0f}",
                "CV_RMSE_mean": "${:,.0f}",
                "CV_R2_mean": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
        key="model_comparison",
    )
    st.success(model_comparison_comment(comparison), icon=":material/check_circle:")

    left, right = st.columns(2)
    with left:
        st.image(str(output / "model_comparison_cv_mae.png"), width="stretch")
    with right:
        st.image(str(output / "model_comparison_cv_r2.png"), width="stretch")

    fold_details = st.expander(
        "Per-fold metrics",
        icon=":material/table_chart:",
        on_change="rerun",
    )
    if fold_details.open:
        with fold_details:
            folds = load_csv(output / "09_model_comparison_fold_metrics.csv")
            st.dataframe(folds, width="stretch", hide_index=True, key="fold_metrics")
