"""
Streamlit entry point of the Portfolio Lab, the product app (separate from
the research dashboard). Launch with scripts/run_portfolio_lab.py, which
binds the server to 127.0.0.1 and disables usage statistics.

Synthetic demo mode is the only mode until publications for real funds
exist: the funds come from the seeded generator (`synthetic.py`), and every
metric is unavailable until T5-A implements OD8 group A (`PendingMetrics`).
"""

from __future__ import annotations

import streamlit as st

from stock_agent.app import explore, synthetic
from stock_agent.app.provider import PendingMetrics
from stock_agent.ui.components import apply_theme

st.set_page_config(page_title="Portfolio Lab", layout="wide")
apply_theme()


def fund_history() -> None:
    explore.render(synthetic.demo_histories, PendingMetrics())


SECTIONS = {
    "Explore": [st.Page(fund_history, title="Fund history", url_path="fund-history", default=True)]
}
st.navigation(SECTIONS).run()
