"""The private report: every label present, no ranking words, registered order, nothing remote."""

from __future__ import annotations

import copy
import re

import pytest
from showcase_fixtures import config, default_frames, loader, sessions_2024

from stock_agent.showcase.report import (
    WHAT_IT_DOES_NOT,
    WHAT_IT_SHOWS,
    _log_ticks,
    check_wording,
    money,
    pct,
    render_html,
    render_markdown,
)
from stock_agent.showcase.run import LABELS, run_showcase

SESSIONS = sessions_2024()


@pytest.fixture(scope="module")
def results():
    return run_showcase(
        config(), loader(default_frames(SESSIONS)), sessions=SESSIONS, code_sha="a" * 40
    )


REQUIRED = (
    "Exploratory",
    "Hindsight-selected",
    "Survivorship-biased",
    "2003-2024 only",
    "Not evidence of strategy skill, and not investment advice",
    "Vendor-reported dividends, not issuer-verified",
    "Before taxes",
    "no cash holding",
)


def test_every_label_is_on_the_page_and_in_the_markdown(results):
    page, markdown = render_html(results), render_markdown(results)
    for text in (*LABELS, *REQUIRED):
        assert text in page or text in page.replace("&#x27;", "'"), text
        assert text in markdown, text
    for item in (*WHAT_IT_SHOWS, *WHAT_IT_DOES_NOT):
        assert item in markdown
    assert "What this shows" in page and "What it doesn't show" in page


def test_the_report_states_what_happened_and_never_who_won(results):
    assert check_wording(render_html(results) + render_markdown(results)) == []
    assert check_wording("Equal weight beats buy and hold") == ["beats"]
    assert check_wording("the best rule outperformed") == ["best", "outperformed"]


def test_variants_appear_in_registered_order_not_by_result(results):
    reordered = copy.deepcopy(results)
    universe = reordered["universes"][0]
    # Make the last-registered rule the largest: the order must not change.
    universe["strategies"][-1]["final_wealth"] = 10**9
    page = render_html(reordered)
    labels = [s["label"] for s in universe["strategies"]]
    section = page[page.index('<section id="funds">') : page.index('<section id="stocks">')]
    table = section[section.index("<tbody>") :]
    positions = [table.index(f">{label}<") for label in labels]
    assert positions == sorted(positions)


def test_the_page_fetches_nothing_from_another_host(results):
    page = render_html(results, font_dir="fonts")
    assert not re.search(r'(src|href)\s*=\s*"(https?:)?//', page)
    assert 'url("fonts/SourceSerif4Variable-Roman.woff2")' in page
    assert "@import" not in page


def test_text_from_the_configuration_is_escaped(results):
    hostile = copy.deepcopy(results)
    hostile["universes"][0]["label"] = "<script>alert(1)</script>"
    page = render_html(hostile)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_a_stopped_variant_is_shown_as_stopped_with_no_total_return(results):
    stopped = copy.deepcopy(results)
    strategy = stopped["universes"][0]["strategies"][1]
    strategy.update(complete=False, status_reason="distribution_in_gap", final_wealth=None)
    strategy["total_return"] = None
    page, markdown = render_html(stopped), render_markdown(stopped)
    assert "stopped: distribution_in_gap" in page and "stopped: distribution_in_gap" in markdown
    assert "unavailable" in page


def test_formatting_helpers():
    assert pct(0.1234) == "+12.3%" and pct(-0.05) == "−5.0%" and pct(None) == "unavailable"
    assert money(12345.6) == "$12,346" and money(float("nan")) == "unavailable"
    assert _log_ticks(9000, 120000) == [10000, 20000, 50000, 100000]
    assert _log_ticks(9500, 14000) == [10000, 12000, 14000]  # a narrow range: linear ticks


def test_a_chart_breaks_at_a_gap_instead_of_bridging_it(results):
    gapped = copy.deepcopy(results)
    universe = gapped["universes"][0]
    for strategy in universe["strategies"]:
        strategy["wealth"][50] = None
    page = render_html(gapped)
    chart = page[page.index('data-chart="funds"') :]
    path = re.search(r'<path class="line rule-buy_and_hold" d="([^"]+)"', chart).group(1)
    assert path.count("M") == 2  # two runs: before and after the gap
