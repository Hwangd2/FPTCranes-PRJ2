from __future__ import annotations

import streamlit as st

from src.components._header import page_header
from src.config import Config
from src.utils.artifacts import load_csv

page_header(
    "2. Data ready for machine learning",
    "Split first, fit preprocessing on training data, and enforce the leakage gate.",
    "model_training",
)

output = Config.OUTPUT_DIR / "02_data_ready_for_machine_learning"
split = load_csv(output / "08_split_summary.csv")
vocabulary = load_csv(output / "skill_vocabulary_train_only.csv")
correlation = load_csv(output / "07_train_encoded_feature_target_correlation.csv")
policy = load_csv(output / "06_feature_policy.csv")
feature_names = load_csv(output / "08_encoded_feature_names.csv")

if len(split) >= 2:
    train = split.iloc[0]
    test = split.iloc[1]
    columns = st.columns(4)
    columns[0].metric(
        "Train / development", f"{int(train['rows']):,}", f"{train['pct']:.1f}%"
    )
    columns[1].metric("Locked test", f"{int(test['rows']):,}", f"{test['pct']:.1f}%")
    columns[2].metric("Skill tokens", len(vocabulary))
    columns[3].metric("Encoded features", len(feature_names))

st.success(
    "Salary minimum, maximum, and tier are blocked. Required skills are normalized and fit "
    "to a training-only multi-hot vocabulary; `skill_count` is the distinct token count.",
    icon=":material/verified_user:",
)

chart = output / "top30_train_target_correlation.png"
if chart.is_file():
    st.image(str(chart), width="stretch")

if not correlation.empty:
    top = correlation.iloc[0]
    st.info(
        f"The strongest training-only encoded association is **{top['encoded_feature']}** "
        f"with Pearson r = {top['pearson_r']:+.3f}. Correlation is diagnostic evidence, not "
        "a causal claim or automatic keep/drop rule.",
        icon=":material/query_stats:",
    )

left, right = st.columns(2)
with left:
    st.subheader("Feature policy", anchor=False)
    st.dataframe(
        policy, width="stretch", hide_index=True, height=420, key="feature_policy"
    )
with right:
    st.subheader("Training skill vocabulary", anchor=False)
    st.dataframe(
        vocabulary,
        width="stretch",
        hide_index=True,
        height=420,
        key="skill_vocabulary",
    )
