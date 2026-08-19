from __future__ import annotations

import streamlit as st


def configure_page() -> None:
    """Apply page-level settings; visual tokens live in `.streamlit/config.toml`."""
    st.set_page_config(
        page_title="AI job market salary prediction",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "Read-only technical report for the evaluated salary model."
        },
    )
