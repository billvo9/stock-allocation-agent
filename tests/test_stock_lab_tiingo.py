"""
The Stock Lab's Tiingo adapter (stock_agent.stock_lab.tiingo) on fakes only:
no request without authorization, a request budget, the documented fields
only (adjusted fields refused), and the key never in a message.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date

import numpy as np
import pytest

from stock_agent.data.vendors.tiingo import TiingoFailure, TiingoRequestFailed
from stock_agent.stock_lab.tiingo import (
    LISTING_URL,
    PRICE_FIELDS,
    NotAuthorized,
    StockAdapter,
    StockDataRefused,
    parse_listing,
    parse_prices,
    prices_url,
)

KEY = "s3cr3t-key-value"
START, END = date(2024, 1, 2), date(2024, 1, 4)


def price_row(day, close=100.0, **extra):
    row = {
        "date": f"{day}T00:00:00.000Z",
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000,
        "divCash": 0.0,
        "splitFactor": 1.0,
    }
    row.update(extra)
    return row


def body(rows):
    return json.dumps(rows).encode()


def listing_zip(text, name="supported_tickers.csv"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


LISTING = (
    "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
    "aapl,NASDAQ,Stock,USD,1980-12-12,2026-10-15\n"
    "VOO,NYSE ARCA,ETF,USD,2010-09-09,2026-10-15\n"
    "XYZ,PINK,Stock,USD,,\n"
)


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def adapter(responses, **kwargs):
    transport = FakeTransport(responses)
    sleeps = []
    built = StockAdapter(
        authorized=kwargs.pop("authorized", True),
        token_provider=lambda: KEY,
        transport=transport,
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
        **kwargs,
    )
    return built, transport, sleeps


def test_no_request_is_made_without_authorization():
    for flag in (False, "yes", 1, None):
        built, transport, _ = adapter([(200, body([]))], authorized=flag)
        with pytest.raises(NotAuthorized):
            built.prices("AAPL", START, END)
        with pytest.raises(NotAuthorized):
            built.listing()
        assert transport.calls == []


def test_a_prices_request_asks_for_the_documented_raw_fields_only():
    built, transport, _ = adapter([(200, body([price_row("2024-01-02"), price_row("2024-01-03")]))])
    response = built.prices("BRK-B", START, END)
    url, headers = transport.calls[0]
    assert url == prices_url("BRK-B", START, END)
    assert url.startswith("https://api.tiingo.com/tiingo/daily/BRK-B/prices?")
    assert f"columns={','.join(PRICE_FIELDS)}" in url and "adj" not in url
    assert headers == {"Authorization": f"Token {KEY}"}
    assert len(response.bars) == 2 and response.bars["close"].tolist() == [100.0, 100.0]


@pytest.mark.parametrize(
    "row",
    [
        price_row("2024-01-02", adjClose=99.0),
        {k: v for k, v in price_row("2024-01-02").items() if k != "volume"},
        price_row("2024-01-02", close="100"),
        price_row("2024-01-02", close=True),
        price_row("2024-01-02T10:00"),
        price_row("2024-01-02", close=-1.0),
    ],
)
def test_responses_outside_the_documented_format_are_refused(row):
    with pytest.raises(StockDataRefused):
        parse_prices("AAPL", body([row]))


def test_non_json_and_non_finite_numbers_are_refused():
    with pytest.raises(StockDataRefused):
        parse_prices("AAPL", b"<html>")
    with pytest.raises(StockDataRefused):
        parse_prices("AAPL", b'[{"close": NaN}]')
    with pytest.raises(StockDataRefused):
        parse_prices("AAPL", b'{"detail": "x"}')


def test_a_session_outside_the_request_is_refused():
    built, _, _ = adapter([(200, body([price_row("2024-01-08")]))])
    with pytest.raises(StockDataRefused, match="outside"):
        built.prices("AAPL", START, END)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (401, TiingoFailure.INVALID_TOKEN),
        (403, TiingoFailure.INVALID_TOKEN),
        (404, TiingoFailure.HTTP_ERROR),
        (429, TiingoFailure.RATE_LIMITED),
        (503, TiingoFailure.SERVER_ERROR),
    ],
)
def test_http_failures_name_the_reason_and_never_the_key(status, reason):
    built, _, _ = adapter([(status, b"")])
    with pytest.raises(TiingoRequestFailed) as raised:
        built.prices("AAPL", START, END)
    assert raised.value.reason is reason and KEY not in str(raised.value)


def test_a_transport_failure_reports_its_type_only_without_a_chained_cause():
    built, _, _ = adapter([OSError(f"connect failed for {KEY}")])
    with pytest.raises(TiingoRequestFailed) as raised:
        built.prices("AAPL", START, END)
    assert KEY not in str(raised.value) and "OSError" in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_a_run_has_a_request_budget_and_paces_requests():
    responses = [(200, body([price_row("2024-01-02")]))] * 3
    built, transport, sleeps = adapter(responses, max_requests=2, min_interval=2.0)
    built.prices("AAPL", START, END)
    built.prices("MSFT", START, END)
    with pytest.raises(TiingoRequestFailed, match="budget"):
        built.prices("NVDA", START, END)
    assert len(transport.calls) == 2 and sleeps == [2.0]


def test_the_listing_is_parsed_and_its_host_never_gets_the_key():
    built, transport, _ = adapter([(200, listing_zip(LISTING))])
    response = built.listing()
    assert transport.calls == [(LISTING_URL, {})]
    tickers = [row.ticker for row in response.rows]
    assert tickers == ["AAPL", "VOO", "XYZ"]
    assert response.rows[0].end_date == date(2026, 10, 15) and response.rows[2].end_date is None


@pytest.mark.parametrize(
    "archive",
    [
        b"not a zip",
        listing_zip("ticker,exchange\nAAPL,NASDAQ\n"),
        listing_zip(LISTING.replace("2026-10-15", "15/10/2026", 1)),
        listing_zip("ticker,exchange,assetType,priceCurrency,startDate,endDate\n"),
        listing_zip(LISTING + "BAD,ROW\n"),
    ],
)
def test_a_listing_outside_the_documented_format_is_refused(archive):
    with pytest.raises(StockDataRefused):
        parse_listing(archive)


def test_tickers_must_be_in_the_providers_form():
    for bad in ("brk.b", "DEMO_CALM", "../etc", "AAPL?x=1"):
        with pytest.raises(ValueError):
            prices_url(bad, START, END)
    with pytest.raises(ValueError):
        prices_url("AAPL", END, START)


@pytest.mark.parametrize(
    "text",
    [
        (
            '[{"date": "2024-01-02T00:00:00.000Z", "open": 1, "high": 1, "low": 1, "close": 1, '
            '"close": 999, "volume": 1, "divCash": 0, "splitFactor": 1}]'
        ),
        json.dumps([price_row("2024-01-02", divCash=None)]),
        json.dumps([price_row("2024-01-02", splitFactor=None)]),
        json.dumps([price_row("2024-01-02", close=10**400)]),
        json.dumps([price_row("2024-02-30")]),
        json.dumps([{**price_row("2024-01-02"), "date": "2024-01-02T10:00:00.000Z"}]),
    ],
    ids=["duplicate-key", "null-dividend", "null-split", "overflow", "no-such-day", "intraday"],
)
def test_every_malformed_price_response_is_a_structured_refusal(text):
    with pytest.raises(StockDataRefused):
        parse_prices("AAPL", text.encode())


def test_missing_prices_stay_missing_but_volume_and_prices_may_be_null():
    bars = parse_prices("AAPL", body([price_row("2024-01-02", volume=None, open=None)]))
    assert np.isnan(bars["volume"].iloc[0]) and np.isnan(bars["open"].iloc[0])


def test_a_corrupt_or_undecodable_listing_is_a_structured_refusal():
    good = listing_zip(LISTING)
    corrupt = bytearray(good)
    middle = len(corrupt) // 3
    corrupt[middle : middle + 8] = b"\xff" * 8
    for archive in (bytes(corrupt), listing_zip(LISTING.encode("utf-16").decode("latin-1"))):
        with pytest.raises(StockDataRefused):
            parse_listing(archive)


def test_an_unauthorized_adapter_never_reads_the_key():
    def provider():
        raise AssertionError("the key was read")

    built = StockAdapter(authorized=False, token_provider=provider, transport=FakeTransport([]))
    with pytest.raises(NotAuthorized):
        built.prices("AAPL", START, END)


def test_a_refused_listing_request_is_not_blamed_on_the_key():
    built, _, _ = adapter([(403, b"")])
    with pytest.raises(TiingoRequestFailed) as raised:
        built.listing()
    assert raised.value.reason is TiingoFailure.HTTP_ERROR
