"""
Streamlit entry point. Launch with scripts/run_dashboard.py, which binds the
server to 127.0.0.1 and disables usage statistics.
"""

from __future__ import annotations

import streamlit as st

from stock_agent.dashboard.views import sidebar

st.set_page_config(page_title="Model diagnostics", layout="wide")
st.session_state["_view"] = sidebar()

PAGES = [
    st.Page("page_scripts/overview.py", title="Run overview", default=True),
    st.Page("page_scripts/folds.py", title="Fold construction"),
    st.Page("page_scripts/features.py", title="Data & feature diagnostics"),
    st.Page("page_scripts/nulls.py", title="Nulls & controls"),
    st.Page("page_scripts/validation.py", title="Validation results"),
    st.Page("page_scripts/calibration.py", title="Calibration"),
    st.Page("page_scripts/uncertainty.py", title="Statistical uncertainty"),
    st.Page("page_scripts/research_warnings.py", title="Research warnings"),
    st.Page("page_scripts/glossary.py", title="Glossary"),
]
st.navigation(PAGES).run()
