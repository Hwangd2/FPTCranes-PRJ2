from __future__ import annotations

import mimetypes

import streamlit as st

from src.components._header import page_header
from src.config import Config
from src.utils.artifacts import load_json

page_header(
    "Project outputs",
    "Inspect the evidence index and download the application, diagram, and presentation report.",
    "folder_open",
)

index = load_json(Config.OUTPUT_DIR / "report_index.json")
st.json(index, expanded=False)

files = [
    Config.ROOT_DIR / "pipeline.py",
    Config.ROOT_DIR / "streamlit.py",
    Config.PIPELINE_IMAGE,
    Config.PRESENTATION_REPORT,
]
for file in files:
    if file.is_file():
        mime_type, _ = mimetypes.guess_type(file.name)
        st.download_button(
            f"Download {file.name}",
            data=file.read_bytes(),
            file_name=file.name,
            mime=mime_type or "application/octet-stream",
            icon=":material/download:",
            width="stretch",
            key=f"download_{file.name}",
        )

st.info(
    "Run `python pipeline.py` first to refresh data and model artifacts, then launch "
    "`streamlit run streamlit.py`.",
    icon=":material/terminal:",
)
