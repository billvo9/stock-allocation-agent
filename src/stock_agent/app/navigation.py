"""
The Lab's six destinations: each is a small page script in `destinations/`
(relative to run.py), so Streamlit's navigation and AppTest can reach every
one by its file. Titles and URL paths are copy (`wording.NAV`).
"""

from __future__ import annotations

from stock_agent.app import wording

DIRECTORY = "destinations"
DEFAULT = "overview"
PAGES = {key: f"{DIRECTORY}/{key}.py" for key in wording.NAV}


def links(*keys: str) -> dict[str, str]:
    """Page paths for `st.page_link`, by destination key."""

    return {key: PAGES[key] for key in keys}
