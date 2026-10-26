"""
The research pages' launch key (src/stock_agent/app/run.py; owner decision of
2026-10-23): started by the Portfolio Lab Mac app with PORTFOLIO_LAB_RESEARCH_KEY,
the Streamlit Lab shows and changes nothing until a browser session brings that
key (`?key=...`, from the Lab app's link), then remembers it for that session and
takes it out of the address. Started without a key, it opens as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from stock_agent.app import run, wording

RUN = Path(run.__file__)
TIMEOUT = 120
KEY = "q" * 43


def page_text(app: AppTest) -> str:
    return " ".join(str(item.value) for item in app.markdown)


def started(key: str | None, monkeypatch, given: str | None = None) -> AppTest:
    if key is None:
        monkeypatch.delenv(run.RESEARCH_KEY_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(run.RESEARCH_KEY_VARIABLE, key)
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT)
    if given is not None:
        app.query_params["key"] = given
    return app.run()


@pytest.mark.parametrize("given", [None, "", "q" * 42, "Q" * 43, KEY + "x", "kéy" * 15])
def test_without_the_launch_key_nothing_is_shown(monkeypatch, given):
    app = started(KEY, monkeypatch, given)
    assert not app.exception
    text = page_text(app)
    assert wording.RESEARCH_LOCKED_TITLE in text and "More, then Research pages" in text
    assert len(app.markdown) == 2 and not app.button and not app.text_input


def test_with_the_launch_key_the_pages_open_and_the_key_leaves_the_address(monkeypatch):
    app = started(KEY, monkeypatch, KEY)
    assert not app.exception
    assert wording.RESEARCH_LOCKED_TITLE not in page_text(app)
    assert "key" not in app.query_params
    assert app.session_state[run.UNLOCKED] is True
    app.run()  # the same browser session, without the key in the address any more
    assert not app.exception and wording.RESEARCH_LOCKED_TITLE not in page_text(app)


def test_started_from_a_terminal_the_pages_open_as_before(monkeypatch):
    app = started(None, monkeypatch)
    assert not app.exception and wording.RESEARCH_LOCKED_TITLE not in page_text(app)
