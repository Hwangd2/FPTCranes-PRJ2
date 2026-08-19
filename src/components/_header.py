from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str, icon: str) -> None:
    """Render consistent, theme-aware page hierarchy using native elements."""
    st.title(f":material/{icon}: {title}", anchor=False)
    st.caption(subtitle)
