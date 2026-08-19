from __future__ import annotations

import streamlit as st
from src.components._header import page_header
from src.config import Config
from src.utils.artifacts import load_json
from src.utils.formatting import format_money

page_header(
    "AI job market salary prediction",
    "From data-quality audit to leakage-safe temporal validation and deployable inference.",
    "rocket_launch",
)

run = load_json(Config.OUTPUT_DIR / "run_summary.json")
metadata = load_json(Config.ARTIFACT_DIR / "metadata.json")
metrics = metadata.get("final_locked_test_metrics", {})

metric_columns = st.columns(4)
metric_columns[0].metric("Raw rows", f"{run.get('raw_rows', 0):,}")
metric_columns[1].metric("Clean rows", f"{run.get('clean_rows', 0):,}")
metric_columns[2].metric("Train / development", f"{run.get('train_rows', 0):,}")
metric_columns[3].metric("Locked test", f"{run.get('locked_test_rows', 0):,}")

left, right = st.columns([1.4, 1], vertical_alignment="top")
with left:
    pipeline_image = Config.PIPELINE_IMAGE
    if pipeline_image.is_file():
        st.image(str(pipeline_image), width="stretch")
    else:
        st.info("The pipeline diagram will appear after the offline pipeline runs.")

with right.container(border=True):
    st.subheader("Final model evidence", anchor=False)
    st.metric("Locked-test R²", f"{metrics.get('R2', float('nan')):.3f}")
    st.write(f"**Selected model:** {metadata.get('model_name', '—')}")
    st.write(f"**Locked-test MAE:** {format_money(metrics.get('MAE', 0))}")
    st.write(f"**Locked-test RMSE:** {format_money(metrics.get('RMSE', 0))}")
    st.write(f"**Locked-test MedAE:** {format_money(metrics.get('MedAE', 0))}")
    interval = metadata.get("prediction_interval_abs_error_q90", 0)
    st.write(f"**90% practical interval half-width:** {format_money(interval)}")

st.warning(
    "The supplied snapshot contains contradictory experience semantics, salary metadata "
    "inconsistencies, and synthetic-looking ordering. Strong performance describes fit to "
    "this dataset; it does not establish causal salary economics.",
    icon=":material/science:",
)
