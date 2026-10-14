"""
Streamlit entry point of the standalone Model Health page. Launch with
scripts/run_model_health.py, which binds the server to 127.0.0.1, disables
usage statistics and passes the summary paths in the environment variables
named in `page` (unset means "no summary").
"""

from __future__ import annotations

import os

import streamlit as st

from stock_agent.health import page
from stock_agent.ui.components import apply_theme

st.set_page_config(page_title="Model health", layout="wide")
apply_theme()
page.render(page.sources_from_environment(os.environ))
