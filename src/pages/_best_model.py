from __future__ import annotations

import streamlit as st

from src.components._header import page_header
from src.config import Config
from src.utils.artifacts import load_csv
from src.utils.formatting import format_money

page_header(
    "4. Best model selection and feature review",
    "Freeze the development-CV choice, open the locked test once, and inspect model behavior.",
    "trophy",
)

output = Config.OUTPUT_DIR / "04_best_model_and_feature_importance"
final = load_csv(output / "10_final_locked_test_metrics.csv")
raw_importance = load_csv(output / "10_raw_feature_permutation_importance.csv")

if not final.empty:
    row = final.iloc[0]
    columns = st.columns(4)
    columns[0].metric("Best model", row["selected_model"])
    columns[1].metric("MAE", format_money(row["MAE"]))
    columns[2].metric("RMSE", format_money(row["RMSE"]))
    columns[3].metric("R²", f"{row['R2']:.3f}")

left, right = st.columns(2)
with left:
    st.image(str(output / "raw_feature_permutation_importance.png"), width="stretch")
with right:
    st.image(str(output / "actual_vs_predicted_locked_test.png"), width="stretch")

st.subheader("Raw feature-family importance", anchor=False)
st.dataframe(raw_importance, width="stretch", hide_index=True, key="raw_importance")
if len(raw_importance) >= 2:
    first, second = raw_importance.iloc[0], raw_importance.iloc[1]
    st.warning(
        f"**{first['raw_feature']}** and **{second['raw_feature']}** dominate the permutation "
        "signal. This is fitted-model behavior, not proof that these factors causally "
        "determine real salaries.",
        icon=":material/science:",
    )

encoded_details = st.expander(
    "Top encoded importances",
    icon=":material/format_list_numbered:",
    on_change="rerun",
)
if encoded_details.open:
    with encoded_details:
        encoded = load_csv(output / "10_encoded_feature_importance.csv")
        st.dataframe(
            encoded.head(40), width="stretch", hide_index=True, key="encoded_importance"
        )
