"""
Ticker parsing and classification (stock_agent.stock_lab.symbols): US-listed
stocks only, every rewrite visible, and never a substituted listing.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.stock_lab.contracts import SymbolReason, SymbolStatus
from stock_agent.stock_lab.symbols import (
    MAX_TICKERS_PER_PASTE,
    ListingRow,
    classify,
    parse_ticker,
    parse_tickers,
    refused,
    unverified,
)

CUTOFF = date(2026, 10, 9)


def row(
    ticker, exchange="NASDAQ", asset="Stock", currency="USD", start="2000-01-03", end="2026-10-15"
):
    return ListingRow(
        ticker,
        exchange,
        asset,
        currency,
        date.fromisoformat(start) if start else None,
        date.fromisoformat(end) if end else None,
    )


def weekdays_since(day: date) -> int:
    import numpy as np

    return int(np.busday_count(day, CUTOFF))


def check(ticker, rows):
    return classify(ticker, rows, recent_cutoff=CUTOFF, sessions_since=weekdays_since)


@pytest.mark.parametrize(
    ("entered", "ticker"),
    [
        ("aapl", "AAPL"),
        ("  MSFT ", "MSFT"),
        ("brk.b", "BRK-B"),
        ("BRK/B", "BRK-B"),
        ("BRK-B", "BRK-B"),
        ("BF.A", "BF-A"),
        ("F", "F"),
    ],
)
def test_valid_tickers_are_normalized_to_the_providers_form(entered, ticker):
    parsed = parse_ticker(entered)
    assert parsed.ok and parsed.ticker == ticker and parsed.reason is None
    assert parsed.rewritten == (ticker != entered.strip().upper())


@pytest.mark.parametrize(
    ("entered", "reason"),
    [
        ("", SymbolReason.EMPTY),
        ("VOD.L", SymbolReason.NON_US_SUFFIX),
        ("SHOP.TO", SymbolReason.NON_US_SUFFIX),
        ("7203.T", SymbolReason.NON_US_SUFFIX),
        ("SAP.DE", SymbolReason.NON_US_SUFFIX),
        ("0700.HK", SymbolReason.NON_US_SUFFIX),
        ("^GSPC", SymbolReason.INDEX_OR_DERIVED),
        ("EURUSD=X", SymbolReason.INDEX_OR_DERIVED),
        ("ES=F", SymbolReason.INDEX_OR_DERIVED),
        ("TOOLONG", SymbolReason.INVALID_FORMAT),
        ("12345", SymbolReason.INVALID_FORMAT),
        ("DEMO_CALM", SymbolReason.INVALID_FORMAT),
        ("A$B", SymbolReason.INVALID_FORMAT),
    ],
)
def test_non_us_listings_and_non_stocks_are_refused_never_substituted(entered, reason):
    parsed = parse_ticker(entered)
    assert not parsed.ok and parsed.ticker is None and parsed.reason is reason
    assert parsed.detail
    assert refused(parsed).status is SymbolStatus.UNSUPPORTED


def test_a_paste_is_split_on_any_separator_and_deduplicated_in_order():
    parsed = parse_tickers("aapl, MSFT;brk.b\nBRK-B  msft\tVOD.L")
    assert [p.ticker for p in parsed] == ["AAPL", "MSFT", "BRK-B", None]
    assert parsed[-1].reason is SymbolReason.NON_US_SUFFIX


def test_a_paste_has_a_size_limit():
    with pytest.raises(ValueError):
        parse_tickers(" ".join(["AAPL"] * (MAX_TICKERS_PER_PASTE + 1)))


def test_refused_needs_a_failed_entry():
    with pytest.raises(ValueError):
        refused(parse_ticker("AAPL"))


def test_a_current_us_listed_stock_is_supported():
    result = check("AAPL", [row("AAPL")])
    assert result.status is SymbolStatus.SUPPORTED and result.reason is SymbolReason.LISTED_US_STOCK


@pytest.mark.parametrize(
    ("listing", "reason"),
    [
        (row("ABC", asset="ETF"), SymbolReason.NOT_A_STOCK),
        (row("ABC", asset="Mutual Fund"), SymbolReason.NOT_A_STOCK),
        (row("ABC", exchange="PINK"), SymbolReason.OVER_THE_COUNTER),
        (row("ABC", exchange="OTCMKTS"), SymbolReason.OVER_THE_COUNTER),
        (row("ABC", exchange="SHE"), SymbolReason.NON_US_EXCHANGE),
        (row("ABC", exchange="LSE"), SymbolReason.NON_US_EXCHANGE),
        (row("ABC", exchange="MARS"), SymbolReason.UNKNOWN_EXCHANGE),
        (row("ABC", exchange=""), SymbolReason.UNKNOWN_EXCHANGE),
        (row("ABC", currency="CNY"), SymbolReason.NON_USD),
        (row("ABC", end=None), SymbolReason.NO_PRICE_HISTORY),
    ],
)
def test_ineligible_listings_are_unsupported_with_their_reason(listing, reason):
    result = check("ABC", [listing])
    assert result.status is SymbolStatus.UNSUPPORTED and result.reason is reason


def test_a_ticker_the_provider_does_not_list_is_unsupported():
    result = check("ZZZZ", [row("AAPL")])
    assert result.status is SymbolStatus.UNSUPPORTED and result.reason is SymbolReason.NOT_LISTED


def test_several_current_eligible_listings_are_ambiguous_never_guessed():
    result = check("ABC", [row("ABC", exchange="NYSE"), row("ABC", exchange="NASDAQ")])
    assert result.status is SymbolStatus.AMBIGUOUS and result.row is None


def test_a_reused_ticker_with_one_current_listing_is_that_listing():
    old = row("ABC", exchange="NYSE", start="1995-01-03", end="2009-06-30")
    new = row("ABC", exchange="NASDAQ", start="2015-01-02", end="2026-10-15")
    result = check("ABC", [old, new])
    assert result.status is SymbolStatus.SUPPORTED and result.row == new


def test_an_eligible_listing_that_stopped_recently_is_inactive():
    result = check("ABC", [row("ABC", end="2026-09-25")])  # 10 weekdays before the cutoff
    assert result.status is SymbolStatus.INACTIVE and result.reason is SymbolReason.NO_RECENT_PRICES


def test_an_eligible_listing_that_stopped_long_ago_is_delisted_or_renamed():
    result = check("ABC", [row("ABC", end="2021-03-31")])
    assert result.status is SymbolStatus.DELISTED and result.reason is SymbolReason.COVERAGE_ENDED
    assert "does not say which" in result.detail


def test_an_etf_beside_a_current_stock_row_does_not_block_the_stock():
    result = check("ABC", [row("ABC", asset="ETF"), row("ABC")])
    assert result.status is SymbolStatus.SUPPORTED


def test_lowercase_listing_tickers_still_match():
    assert check("ABC", [row("abc")]).status is SymbolStatus.SUPPORTED


def test_unverified_is_the_status_before_any_listing_is_checked():
    result = unverified("AAPL")
    assert result.status is SymbolStatus.UNVERIFIED and result.reason is SymbolReason.NOT_CHECKED
