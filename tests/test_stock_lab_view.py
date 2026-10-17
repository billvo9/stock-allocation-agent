"""The Stock Probability page's pure view logic (stock_agent.stock_lab.view)."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import StableBand
from stock_agent.stock_lab.view import (
    demo_rows,
    feature_table,
    headline,
    price_table,
    recent_table,
    run_length_table,
    search,
    stock_detail,
    summary_table,
    threshold_table,
)


@pytest.fixture(scope="module")
def swing():
    bars = synthetic.demo_bars("DEMO_SWING")
    return stock_detail("DEMO_SWING", bars, name="Swing", synthetic_data=True)


def test_prices_are_shown_on_the_last_sessions_share_basis(swing):
    last = swing.frame.iloc[-1]
    assert last["display_level"] == pytest.approx(last["close"])
    before_split = swing.frame.iloc[899]
    assert before_split["display_level"] == pytest.approx(before_split["close"] / 3.0)
    table = price_table(swing)
    assert table.iloc[-1]["Price"] == f"{last['close']:,.2f}"


def test_cards_keep_values_short_and_put_the_rest_in_captions(swing):
    cards = {card.label: card for card in headline(swing)}
    assert cards["Last close"].caption.startswith("on 2024-")
    assert len(cards["Current streak"].value) <= 12
    assert cards["Typical daily move"].help
    assert cards["Stable sessions at ±0.5%"].caption.endswith("sessions")


def test_tables_say_what_their_numbers_count(swing):
    bands = threshold_table(swing)
    assert bands["Stable band"].tolist() == ["±0.25%", "±0.5%", "±1%"]
    shares = bands[["Increase", "Stable", "Decrease"]].map(lambda t: float(t.rstrip("%")))
    assert (shares.sum(axis=1).round(0) == 100).all()
    runs = run_length_table(swing)
    assert list(runs.columns)[1] == "1 session" and "6 or more" in runs.columns
    assert feature_table(swing)["Value"].str.endswith("×").any()
    assert recent_table(swing).index.name == "Session date"


def test_the_summary_explains_every_status_and_search_ignores_case():
    rows = demo_rows(synthetic.demo_histories(), band=StableBand())
    table = summary_table(rows, band=StableBand())
    assert list(table.columns[:5]) == ["Stock", "Status", "Data", "Why", "Last session"]
    assert (table["Status"] == "Synthetic demo").all() and table["Why"].str.len().gt(0).all()
    assert [row.ticker for row in search(rows, "swing")] == ["DEMO_SWING"]
    assert len(search(rows, "  ")) == 3
    assert all(pd.notna(table["Last session"]))
