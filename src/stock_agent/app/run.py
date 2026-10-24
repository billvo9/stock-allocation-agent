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

The launch key. Started by the Portfolio Lab Mac app, these research pages get
a key of their own for that launch (PORTFOLIO_LAB_RESEARCH_KEY; owner decision
of 2026-10-23), and show nothing, and change nothing, until a browser session
brings it: the Lab app's link to them carries it as `?key=...`. It is compared
in constant time, remembered for that browser session only, and removed from
the address at once. Started from a terminal (no key), the pages open as before.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st

from stock_agent.app import navigation, wording
from stock_agent.ui.components import apply_theme

ICON = Path(__file__).with_name("static") / "portfolio-lab-icon.png"
RESEARCH_KEY_VARIABLE = "PORTFOLIO_LAB_RESEARCH_KEY"  # set by scripts/portfolio_lab_app.py
UNLOCKED = "_launch_key_given"  # this browser session brought the launch key


def unlocked() -> bool:
    """Whether this browser session may see the pages (always, without a launch key)."""

    key = os.environ.get(RESEARCH_KEY_VARIABLE)
    if not key or st.session_state.get(UNLOCKED):
        return True
    given = str(st.query_params.get("key", ""))
    if not hmac.compare_digest(given.encode("utf-8", "surrogateescape"), key.encode("utf-8")):
        return False
    st.session_state[UNLOCKED] = True
    del st.query_params["key"]  # out of the address at once
    return True


st.set_page_config(page_title=wording.LAB_NAME, page_icon=ICON, layout="wide")
apply_theme("lab")
if not unlocked():
    st.markdown(f"### {wording.RESEARCH_LOCKED_TITLE}")
    st.markdown(wording.RESEARCH_LOCKED)
    st.stop()

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
