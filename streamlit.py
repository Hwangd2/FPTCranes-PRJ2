"""Authenticated, read-only Streamlit router for the salary evidence packs.

Run from the project root with:
    streamlit run streamlit.py
"""

from __future__ import annotations

import streamlit as st

from src.components._nav import build_navigation, render_sidebar_context
from src.components._styles import configure_page
from src.config import Config
from src.pages._login import require_authentication
from src.utils.artifacts import load_json

configure_page()
require_authentication()

try:
    metadata = load_json(Config.ARTIFACT_DIR / "metadata.json")
except (OSError, ValueError) as error:
    st.error(
        f"The model metadata could not be read: {error}. Run `python pipeline.py` to "
        "regenerate the evaluated artifacts.",
        icon=":material/error:",
    )
    st.stop()

if not metadata:
    st.error(
        "Model metadata is missing. Run `python pipeline.py` before launching the dashboard.",
        icon=":material/database_off:",
    )
    st.stop()

page = build_navigation()
render_sidebar_context(metadata)
page.run()
