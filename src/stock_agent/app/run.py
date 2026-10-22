"""
Streamlit entry point of the Portfolio Lab, the product app (separate from
the research dashboard). Launch with scripts/run_portfolio_lab.py, which
binds the server to 127.0.0.1 and disables usage statistics.

Eight destinations share one top navigation (`navigation.PAGES`, one page
script each): Overview (the landing page), Explore, Portfolio Builder,
Stock Probability, Premarket Briefing, Experiments, Model Health and
Evidence. The editorial theme comes from the
script-level `.streamlit/config.toml` next to this file (with the bundled
fonts in `static/`) and the CSS layer applied here once per run. The browser
tab shows the Lab's mark (`static/portfolio-lab-icon.png`, rendered from
`assets/brand/portfolio-lab-mark.svg`), the simplified version of the Mac app's icon.

Synthetic demo mode is the only mode until publications for real funds
exist: the funds come from the seeded generator (`synthetic.py`), and the
metrics are the approved OD8 group A calculations
(`stock_agent.portfolio.descriptive.DescriptiveCalculator`).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stock_agent.app import navigation, wording
from stock_agent.ui.components import apply_theme

ICON = Path(__file__).with_name("static") / "portfolio-lab-icon.png"

st.set_page_config(page_title=wording.LAB_NAME, page_icon=ICON, layout="wide")
apply_theme("lab")

PAGES = [
    st.Page(
        navigation.PAGES[key],
        title=title,
        url_path=url_path,
        default=key == navigation.DEFAULT,
    )
    for key, (title, url_path) in wording.NAV.items()
]
st.navigation(PAGES, position="top").run()
