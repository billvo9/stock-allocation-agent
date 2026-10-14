"""
Offline tests of the Tiingo adapter and the coverage-audit script. Every
response is an invented fixture shaped like Tiingo's documented JSON; tickers
are labels only and no value is a real price or distribution. The key is an
invented sentinel. Where a test needs `VERIFIED_DISTRIBUTIONS` entries, it
patches in invented amounts that match its own fixture; the real allowlist
stays empty.
"""

import ast
import dataclasses
import email.message
import hashlib
import http.client
import importlib.util
import inspect
import io
import json
import logging
import math
import os
import stat
import subprocess
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.calendars import nyse
from stock_agent.data.funds.adapter import FundDataAdapter
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.reader import FILE_SET, read_batch
from stock_agent.data.funds.schema import (
    FundDataRefused,
    Instrument,
    RefusalReason,
    dumps_canonical,
    loads_strict,
)
from stock_agent.data.vendors import tiingo
from stock_agent.data.vendors.tiingo import (
    ALLOWED_FIELDS,
    TiingoAdapter,
    TiingoFailure,
    TiingoRefusal,
    TiingoRequestFailed,
    parse_payload,
    verified_batch,
    verify_distribution_basis,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

TOKEN = "sentinel-tiingo-key-0123456789abcdef"  # invented; never a real key
RETRIEVED = pd.Timestamp("2026-10-14T09:30:00.000000Z")
START, END_EXCLUSIVE = "2014-06-02", "2014-06-14"  # ten sessions, 06-02 to 06-13
DEFAULT_DIVIDENDS = {"2014-06-05": 0.125, "2014-06-11": 0.25}
VTI_SPLIT = "2008-06-18"  # the one verified split
SPLIT_START, SPLIT_END = "2008-06-09", "2008-06-28"
AUDIT_START, AUDIT_END = "2007-06-01", "2009-07-01"
RAW = "unadjusted"
# Names that must never reach a stored artifact as data.
FORBIDDEN_NAMES = (
    "adjClose",
    "adjHigh",
    "adjLow",
    "adjOpen",
    "adjVolume",
    "endDate",
    "description",
    "exchangeCode",
)
SRC = Path(__file__).resolve().parents[1] / "src" / "stock_agent"


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC").as_unit("us")


def element(session: pd.Timestamp, close: float, **fields) -> dict:
    """One invented Tiingo element with the allowlisted fields."""

    row = {
        "date": f"{session:%Y-%m-%d}T00:00:00.000Z",
        "close": close,
        "divCash": 0.0,
        "splitFactor": 1.0,
    }
    row.update(fields)
    return row


def price_list(
    start=START, end=END_EXCLUSIVE, *, dividends=None, split=None, raw_falls=True, skip=()
) -> list[dict]:
    """
    Invented elements. A `split` (date, factor) reports the factor on its date;
    with `raw_falls` the raw close falls by the factor from that date on, as an
    unadjusted price does.
    """

    dividends = DEFAULT_DIVIDENDS if dividends is None else dividends
    rows = []
    sessions = nyse.sessions(day(start), day(end) - pd.Timedelta(days=1))
    for index, session in enumerate(sessions):
        text = f"{session:%Y-%m-%d}"
        if text in skip:
            continue
        close, factor = 40.0 + index % 5, 1.0
        if split is not None and text >= split[0]:
            close = close / split[1] if raw_falls else close
            factor = split[1] if text == split[0] else 1.0
        rows.append(element(session, close, divCash=dividends.get(text, 0.0), splitFactor=factor))
    return rows


def quarterly(start: str, end: str) -> list[str]:
    """The eleventh session of each March, June, September and December."""

    sessions = nyse.sessions(day(start), day(end) - pd.Timedelta(days=1))
    months = pd.Series(sessions).groupby([sessions.year, sessions.month])
    return [
        f"{group.iloc[10]:%Y-%m-%d}"
        for (_, month), group in months
        if month in (3, 6, 9, 12) and len(group) > 10
    ]


def vti_rows(pre_split_dividend: float, start=AUDIT_START) -> list[dict]:
    """VTI across its verified split, quarterly dividends of 0.30 after it."""

    dates = quarterly(start, AUDIT_END)
    dividends = {date: pre_split_dividend if date < VTI_SPLIT else 0.30 for date in dates}
    return price_list(
        start, AUDIT_END, dividends=dividends, split=(VTI_SPLIT, 2.0), skip=("2008-01-15",)
    )


def invented_vti_entries(
    pre_split: float, *, before: int = 1, after: int = 1, start: str = AUDIT_START
) -> dict[tuple[str, str], float]:
    """
    Invented allowlist entries equal to `vti_rows`' served amounts, `before`
    pre-split and `after` post-split ex-dates. Test fixtures, not real amounts.
    """

    dates = quarterly(start, AUDIT_END)
    pre, post = [d for d in dates if d < VTI_SPLIT], [d for d in dates if d > VTI_SPLIT]
    entries = {("VTI", date): pre_split for date in pre[:before]}
    entries.update({("VTI", date): 0.30 for date in post[:after]})
    return entries


def agg_rows() -> list[dict]:
    return price_list(
        AUDIT_START, AUDIT_END, dividends={date: 0.40 for date in quarterly(AUDIT_START, AUDIT_END)}
    )


def body(value) -> bytes:
    return value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")


class FakeTransport:
    """Routes a ticker to (status, body); records every call."""

    def __init__(self, prices=None, status=None, error=None):
        self.prices = prices or {}
        self.status = status or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        if self.error is not None:
            raise self.error
        ticker = url.removeprefix(tiingo.BASE_URL + "/").split("/")[0]
        status = self.status.get(ticker, 200)
        if status != 200:
            return status, b""
        return 200, body(self.prices.get(ticker, price_list()))


def adapter(transport, token=TOKEN, clock=RETRIEVED) -> TiingoAdapter:
    return TiingoAdapter(token_provider=lambda: token, transport=transport, clock=lambda: clock)


def retrieve(transport=None, funds=("VTI",), start=START, end=END_EXCLUSIVE, **kwargs):
    transport = transport if transport is not None else FakeTransport()
    return adapter(transport, **kwargs).retrieve(list(funds), day(start), day(end))


def fetch(transport=None, funds=("VTI",), start=START, end=END_EXCLUSIVE, **kwargs):
    transport = transport if transport is not None else FakeTransport()
    return adapter(transport, **kwargs).fetch(list(funds), day(start), day(end))


@pytest.fixture
def verified(monkeypatch):
    """Invented allowlist entries matching DEFAULT_DIVIDENDS (test-only, not real amounts)."""

    entries = {
        (ticker, date): amount
        for ticker in ("VTI", "AGG", "SHV")
        for date, amount in DEFAULT_DIVIDENDS.items()
    }
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
    return entries


def refused(reason, call, *args, **kwargs) -> FundDataRefused:
    with pytest.raises(FundDataRefused) as caught:
        call(*args, **kwargs)
    assert caught.value.reason is reason, caught.value
    return caught.value


def failed(reason, call, *args, **kwargs) -> TiingoRequestFailed:
    with pytest.raises(TiingoRequestFailed) as caught:
        call(*args, **kwargs)
    assert caught.value.reason is reason, caught.value
    return caught.value


def leaked(error: BaseException) -> bool:
    text = str(error) + repr(error) + "".join(traceback.format_exception(error))
    return TOKEN in text


def stored_bytes(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())


# --- requests ------------------------------------------------------------------


def test_the_adapter_implements_the_f1_protocol_with_injected_collaborators():
    assert isinstance(adapter(FakeTransport()), FundDataAdapter)
    defaults = inspect.signature(TiingoAdapter).parameters
    assert defaults["transport"].default is tiingo.urllib_transport
    assert defaults["token_provider"].default is tiingo.env_token
    assert TOKEN not in repr(adapter(FakeTransport()))
    shown = repr(retrieve())
    assert "40.0" not in shown and "prices" not in shown  # the tables never appear


def test_one_prices_call_per_fund_asks_only_for_allowlisted_columns():
    transport = FakeTransport()
    retrieve(transport, funds=("VTI", "AGG"))
    query = (
        "startDate=2014-06-02&endDate=2014-06-13&format=json&resampleFreq=daily"
        "&columns=date,close,divCash,splitFactor"
    )
    base = tiingo.BASE_URL
    assert [url for url, _ in transport.calls] == [
        f"{base}/VTI/prices?{query}",
        f"{base}/AGG/prices?{query}",
    ]  # no metadata call, no volume
    assert all(headers == {"Authorization": f"Token {TOKEN}"} for _, headers in transport.calls)
    assert not [url for url, _ in transport.calls if TOKEN in url]


def test_a_request_ending_at_the_holdout_start_asks_for_its_eve():
    transport = FakeTransport(prices={"SHV": price_list("2024-12-23", "2025-01-01")})
    retrieval = retrieve(
        transport, funds=("SHV",), start="2024-12-23", end=f"{MODEL_HOLDOUT_START.date()}"
    )
    assert "endDate=2024-12-31&" in transport.calls[0][0]
    assert retrieval.end_exclusive == MODEL_HOLDOUT_START
    assert retrieval.prices["session_date"].max() == day("2024-12-31")


@pytest.mark.parametrize(
    "funds, end, reason",
    [
        (["VTI"], MODEL_HOLDOUT_START + pd.Timedelta(days=1), RefusalReason.REQUEST_AFTER_CAP),
        (["VTI"], day("2026-10-14"), RefusalReason.REQUEST_AFTER_CAP),
        (["vti"], day(END_EXCLUSIVE), RefusalReason.INVALID_REQUEST),
        ([f"T{index}" for index in range(21)], day(END_EXCLUSIVE), RefusalReason.INVALID_REQUEST),
    ],
)
def test_a_post_cutoff_oversized_or_malformed_request_contacts_nothing(funds, end, reason):
    reads = []

    def provider():
        reads.append(1)
        return TOKEN

    transport = FakeTransport()
    late = TiingoAdapter(token_provider=provider, transport=transport, clock=lambda: RETRIEVED)
    refused(reason, late.retrieve, funds, day(START), end)
    assert transport.calls == [] and reads == []


def test_twenty_funds_are_allowed_in_one_batch():
    funds = [f"T{index}" for index in range(tiingo.MAX_TICKERS)]
    assert len(retrieve(funds=funds).fund_ids) == 20


def test_each_response_is_checked_before_the_next_request():
    rows = price_list()
    rows[0] = {**rows[0], "adjClose": 30.0}
    transport = FakeTransport(prices={"VTI": rows})
    refused(TiingoRefusal.NON_ALLOWLISTED_FIELD, retrieve, transport, funds=("VTI", "AGG"))
    assert len(transport.calls) == 1  # AGG is never requested


def test_the_clock_is_read_after_the_last_response():
    log = []
    inner = FakeTransport()

    def transport(url, headers):
        log.append("request")
        return inner(url, headers)

    def clock():
        log.append("clock")
        return RETRIEVED

    TiingoAdapter(token_provider=lambda: TOKEN, transport=transport, clock=clock).retrieve(
        ["VTI", "AGG"], day(START), day(END_EXCLUSIVE)
    )
    assert log == ["request", "request", "clock"]


# --- mapping -------------------------------------------------------------------


def test_each_session_maps_to_one_unadjusted_close_with_no_volume():
    retrieval = retrieve()
    prices, expected = retrieval.prices, price_list()
    assert len(prices) == len(nyse.sessions(day(START), day("2014-06-13"))) == 10
    assert set(prices["adjustment_basis"]) == {RAW}
    assert prices["close"].tolist() == [row["close"] for row in expected]
    assert prices["volume"].isna().all() and str(prices["volume"].dtype) == "Int64"
    assert set(prices["source_id"]) == {"tiingo"} and "batch_id" not in prices.columns
    assert retrieval.splits.empty  # every factor is 1


def test_div_cash_maps_to_distributions_with_no_invented_dates_or_type():
    distributions = retrieve().distributions
    assert distributions["ex_date"].tolist() == [day(date) for date in DEFAULT_DIVIDENDS]
    assert distributions["amount"].tolist() == list(DEFAULT_DIVIDENDS.values())
    assert set(distributions["amount_basis"]) == {"per_share_as_paid"}
    assert set(distributions["distribution_type"]) == {"unknown"}
    assert set(distributions["source_sequence"]) == {0}
    for column in ("declaration_date", "record_date", "pay_date"):
        assert distributions[column].isna().all()
        assert str(distributions[column].dtype) == "datetime64[us, UTC]"


def test_a_verified_batch_records_instruments_and_notes(verified):
    raw = fetch(funds=("VTI", "AGG"))
    assert raw.fund_ids == ("VTI", "AGG")
    assert raw.instruments == (Instrument("VTI", "VTI", None), Instrument("AGG", "AGG", None))
    assert raw.retrieved_at == RETRIEVED
    assert raw.adapter_id == "tiingo_daily" and raw.adapter_version == "4"
    assert "Internal Use Only" in raw.terms_note and "no redistribution" in raw.terms_note
    note = raw.vintage_note
    assert "not a point-in-time vintage" in note
    for name in ("metadata endpoint", "adjClose", "adjVolume", "high", "low", "open", "volume"):
        assert name in note  # the unrequested fields are recorded
    assert "ticker-to-security mapping and the set of sessions served" in note
    assert "neither that anchor nor the split checks detect a later split applied uniformly" in note
    assert note.endswith("distribution basis verified against primary-source amounts: VTI, AGG")


# --- the field allowlist ---------------------------------------------------------


@pytest.mark.parametrize("field", ["adjClose", "adjVolume", "high", "volume", "fxRate"])
def test_any_non_allowlisted_field_refuses_the_batch(field):
    rows = price_list()
    rows[2] = {**rows[2], field: 12345.678}
    error = refused(
        TiingoRefusal.NON_ALLOWLISTED_FIELD, retrieve, FakeTransport(prices={"VTI": rows})
    )
    assert field in str(error) and "12345" not in str(error)  # names only, never a value


def test_a_missing_allowlisted_field_refuses_the_batch():
    rows = price_list()
    del rows[3]["divCash"]
    refused(RefusalReason.SCHEMA_MISMATCH, retrieve, FakeTransport(prices={"VTI": rows}))


def _with_first(**changes) -> list[dict]:
    rows = price_list()
    rows[0] = {**rows[0], **changes}
    return rows


FAILURES = {
    "string close": (_with_first(close="40.0"), RefusalReason.INVALID_FIELD),
    "null close": (_with_first(close=None), RefusalReason.INVALID_FIELD),
    "boolean split factor": (_with_first(splitFactor=True), RefusalReason.INVALID_FIELD),
    "string dividend": (_with_first(divCash="0.1"), RefusalReason.INVALID_FIELD),
    "zero close": (_with_first(close=0.0), RefusalReason.INVALID_VALUE),
    "negative dividend": (_with_first(divCash=-0.1), RefusalReason.INVALID_VALUE),
    "zero split factor": (_with_first(splitFactor=0.0), RefusalReason.INVALID_VALUE),
    "NaN dividend": (_with_first(divCash=float("nan")), RefusalReason.INVALID_JSON),
    "infinite close": (_with_first(close=float("inf")), RefusalReason.INVALID_JSON),
    "overflowing dividend": (
        body(price_list()).replace(b'"divCash": 0.125', b'"divCash": 1e999'),
        RefusalReason.INVALID_JSON,
    ),
    "truncated JSON": (body(price_list())[:-20], RefusalReason.INVALID_JSON),
    "integer beyond float range": (_with_first(close=10**400), RefusalReason.INVALID_VALUE),
    "time of day": (_with_first(date="2014-06-02T14:00:00.000Z"), RefusalReason.INVALID_DATE),
    "plain date": (_with_first(date="2014-06-02"), RefusalReason.INVALID_DATE),
    "repeated date": (price_list() + price_list()[:1], RefusalReason.DUPLICATE_KEY),
    "outside request": (
        price_list() + [element(day("2014-06-16"), 41.0)],
        RefusalReason.OUTSIDE_REQUEST,
    ),
    "error object": ({"detail": "Error: invented"}, RefusalReason.SCHEMA_MISMATCH),
    "not an object": ([["2014-06-02", 40.0]], RefusalReason.SCHEMA_MISMATCH),
    "not UTF-8": (b"[\xff]", RefusalReason.INVALID_JSON),
}


@pytest.mark.parametrize("case", list(FAILURES))
def test_a_malformed_response_refuses_the_batch_without_chaining_it(case):
    payload, reason = FAILURES[case]
    assert b'"divCash": 0.125' in body(price_list())  # the overflow case edits a real token
    error = refused(reason, retrieve, FakeTransport(prices={"VTI": payload}))
    assert TOKEN not in str(error)
    assert error.__cause__ is None and error.__context__ is None  # no response body is chained


def test_strict_json_refusals_chain_no_decoder_error():
    error = refused(RefusalReason.INVALID_JSON, loads_strict, '[{"close": 40.0, "secret": 1')
    assert error.__cause__ is None and error.__context__ is None
    assert "secret" not in str(error)


def test_a_row_on_or_after_the_holdout_start_refuses_the_batch():
    rows = price_list("2024-12-23", "2025-01-01") + [element(day("2025-01-02"), 41.0)]
    error = refused(
        RefusalReason.LOCKBOX_VIOLATION,
        retrieve,
        FakeTransport(prices={"SHV": rows}),
        funds=("SHV",),
        start="2024-12-23",
        end=f"{MODEL_HOLDOUT_START.date()}",
    )
    assert "2025-01-02" in str(error)


# --- splits ----------------------------------------------------------------------


def test_a_verified_split_is_kept_with_unadjusted_closes():
    rows = price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 2.0))
    retrieval = retrieve(FakeTransport(prices={"VTI": rows}), start=SPLIT_START, end=SPLIT_END)
    assert retrieval.splits[["fund_id", "effective_date", "ratio_new_per_old"]].values.tolist() == [
        ["VTI", day(VTI_SPLIT), 2.0]
    ]
    assert set(retrieval.prices["adjustment_basis"]) == {RAW}
    assert tiingo.VERIFIED_SPLITS == {("VTI", "2008-06-18"): 2.0}


@pytest.mark.parametrize(
    "fund, rows, reason",
    [
        # A factor on a date outside the allowlist, for another fund or another date.
        ("AGG", price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 2.0)), "unverified"),
        ("VTI", price_list(SPLIT_START, SPLIT_END, split=("2008-06-17", 2.0)), "unverified"),
        ("VTI", _with_first(splitFactor=0.5), "unverified"),
        # The verified session with another factor, no factor, or no row at all.
        ("VTI", price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 3.0)), "mismatch"),
        ("VTI", price_list(SPLIT_START, SPLIT_END, dividends={}), "mismatch"),
        (
            "VTI",
            price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 2.0), skip=(VTI_SPLIT,)),
            "mismatch",
        ),
        # A raw close that is already split-adjusted, or no previous close to check.
        (
            "VTI",
            price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 2.0), raw_falls=False),
            "price",
        ),
        (
            "VTI",
            price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 2.0), skip=("2008-06-17",)),
            "price",
        ),
    ],
)
def test_an_unverified_or_inconsistent_split_refuses_the_batch(fund, rows, reason):
    expected = {
        "unverified": TiingoRefusal.UNVERIFIED_SPLIT_FACTOR,
        "mismatch": TiingoRefusal.SPLIT_FACTOR_MISMATCH,
        "price": TiingoRefusal.SPLIT_PRICE_MISMATCH,
    }[reason]
    start = START if rows[0]["date"].startswith("2014") else SPLIT_START
    end = END_EXCLUSIVE if start == START else SPLIT_END
    transport = FakeTransport(prices={fund: rows})
    refused(expected, retrieve, transport, funds=(fund,), start=start, end=end)


def test_a_request_starting_on_the_split_cannot_verify_it():
    rows = price_list(VTI_SPLIT, SPLIT_END, split=(VTI_SPLIT, 2.0))
    transport = FakeTransport(prices={"VTI": rows})
    refused(TiingoRefusal.SPLIT_PRICE_MISMATCH, retrieve, transport, start=VTI_SPLIT, end=SPLIT_END)


@pytest.mark.parametrize("key", sorted(tiingo.VERIFIED_SPLITS))
def test_every_verified_split_is_discriminable(key):
    factor = tiingo.VERIFIED_SPLITS[key]
    assert tiingo.discriminable(factor)
    assert abs(math.log(factor)) >= 3 * tiingo.BAND  # the rule, and nothing more


def test_the_fixed_band_and_the_discriminability_rule():
    assert tiingo.BAND == math.log(1.25)
    assert all(tiingo.discriminable(factor) for factor in (2.0, 1.96, 0.5, 3.0, 0.25))
    assert not any(tiingo.discriminable(factor) for factor in (1.95, 1.5, 1.1, 0.52, 1.0))
    assert tiingo.split_close_consistent(1.0) and tiingo.split_close_consistent(1.24)
    assert tiingo.split_close_consistent(1 / 1.24)
    assert not tiingo.split_close_consistent(1.26) and not tiingo.split_close_consistent(2.0)


def test_a_listed_split_that_is_not_discriminable_refuses_the_batch(monkeypatch):
    monkeypatch.setattr(tiingo, "VERIFIED_SPLITS", {("VTI", VTI_SPLIT): 1.5})
    rows = price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 1.5))
    transport = FakeTransport(prices={"VTI": rows})
    refused(
        TiingoRefusal.SPLIT_FACTOR_NOT_DISCRIMINABLE,
        retrieve,
        transport,
        start=SPLIT_START,
        end=SPLIT_END,
    )


# --- distribution-basis verification ---------------------------------------------


def test_the_real_distribution_allowlist_starts_empty():
    assert tiingo.VERIFIED_DISTRIBUTIONS == {}  # owner step: add sourced entries


def basis(rows, fund="VTI", start=AUDIT_START, end=AUDIT_END):
    transport = FakeTransport(prices={fund: rows})
    [check] = verify_distribution_basis(retrieve(transport, funds=(fund,), start=start, end=end))
    return check


@pytest.mark.parametrize(
    "pre_split, passed",
    [(0.60, True), (0.66, True), (0.54, True), (0.30, False), (0.45, False), (0.80, False)],
)
def test_the_split_ratio_can_only_fail_a_fund(monkeypatch, pre_split, passed):
    assert not basis(vti_rows(pre_split)).passed  # no primary-source amounts: never stored
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", invented_vti_entries(pre_split))
    check = basis(vti_rows(pre_split))
    assert check.passed is passed
    assert check.route == ("verified_distributions" if passed else None)
    assert check.split_check == ("agreed" if passed else "failed")
    ratio = pre_split / (2 * 0.30)
    assert f"median before / (factor x median after) = {ratio:.2f}" in check.detail


def test_split_adjusted_pre_split_amounts_fail_despite_listed_post_split_matches(monkeypatch):
    # The statistical probe: pre-split amounts served split-adjusted, and two listed
    # post-split amounts that match. Post-split matches say nothing about the other side.
    entries = invented_vti_entries(0.30, before=0, after=2)
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
    check = basis(vti_rows(0.30))
    assert not check.passed and check.route is None and check.split_check == "failed"
    assert "2 (needs 2), 2 match, 0 do not" in check.detail
    assert "listed matches 0 before, 2 after (needs 1 each)" in check.detail


@pytest.mark.parametrize("before, after", [(0, 2), (2, 0)])
def test_listed_matches_are_needed_on_both_sides_of_each_split(monkeypatch, before, after):
    entries = invented_vti_entries(0.60, before=before, after=after)
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
    check = basis(vti_rows(0.60))
    assert not check.passed and check.split_check == "agreed"  # fails on (ii) alone
    assert f"listed matches {before} before, {after} after" in check.detail


def test_a_distribution_on_a_split_session_fails_the_fund(monkeypatch):
    rows = vti_rows(0.60)
    for row in rows:
        if row["date"].startswith(VTI_SPLIT):
            row["divCash"] = 0.15  # an invented amount whose share basis cannot be known
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", invented_vti_entries(0.60))
    check = basis(rows)
    assert not check.passed and check.split_check == "agreed"
    assert "distributions on the split session 1" in check.detail


def test_a_split_ratio_that_is_not_computable_neither_passes_nor_fails(monkeypatch):
    start = "2007-12-01"  # three pre-split distributions only
    assert not basis(vti_rows(0.60, start=start), start=start).passed
    entries = invented_vti_entries(0.60, start=start)
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
    check = basis(vti_rows(0.60, start=start), start=start)
    assert check.passed and check.split_check == "not_computable"
    assert "3 before, 4 after (needs 4 each), not computable" in check.detail


def test_the_amount_route_needs_two_exact_primary_source_matches(monkeypatch):
    dates = quarterly(AUDIT_START, AUDIT_END)
    cases = [
        ({("AGG", dates[0]): 0.40, ("AGG", dates[1]): 0.40 + 5e-7}, True),
        ({("AGG", dates[0]): 0.40}, False),  # one entry is not enough
        ({("AGG", dates[0]): 0.40, ("AGG", dates[1]): 0.40 + 1e-5}, False),  # a mismatch
        ({("AGG", dates[0]): 0.40, ("AGG", "2007-06-01"): 0.40}, False),  # nothing served
        ({("AGG", dates[0]): 0.40, ("AGG", "2006-03-15"): 0.40}, False),  # outside the window
        ({("VTI", dates[0]): 0.40, ("VTI", dates[1]): 0.40}, False),  # another fund
    ]
    for entries, passed in cases:
        monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
        check = basis(agg_rows(), fund="AGG")
        assert check.passed is passed, (entries, check)
        assert check.route == ("verified_distributions" if passed else None)
        assert check.split_check == "no_split"


def test_a_mismatched_primary_source_amount_fails_the_fund(monkeypatch):
    pre_split_date = quarterly(AUDIT_START, AUDIT_END)[1]
    entries = invented_vti_entries(0.60) | {("VTI", pre_split_date): 0.61}
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
    check = basis(vti_rows(0.60))
    assert not check.passed and check.split_check == "agreed"
    assert "3 (needs 2), 2 match, 1 do not" in check.detail


def test_a_fund_without_distributions_cannot_be_verified(verified):
    check = basis(price_list(dividends={}), fund="SHV", start=START, end=END_EXCLUSIVE)
    assert not check.passed  # the invented entries list amounts that were not served


def test_direct_use_of_the_adapter_stores_nothing_unverified(tmp_path):
    error = refused(TiingoRefusal.UNVERIFIED_DISTRIBUTION_BASIS, fetch, funds=("VTI", "AGG"))
    assert "VTI:" in str(error) and "AGG:" in str(error) and "0.125" not in str(error)
    with pytest.raises(FundDataRefused):
        ingest_batch(tmp_path, fetch(funds=("VTI",)))
    assert list(tmp_path.iterdir()) == []


def test_verified_batch_stores_only_tables_rebuilt_from_the_payload(verified):
    retrieval = retrieve(funds=("VTI", "AGG"))
    batch = verified_batch(retrieval)
    assert batch.distributions.equals(parse_payload(retrieval.raw_payload).distributions)
    changed = retrieval.distributions.copy()
    changed.loc[changed.index[0], "amount"] = 0.5  # an in-memory edit the payload lacks
    tampered = dataclasses.replace(retrieval, distributions=changed)
    for call in (verified_batch, verify_distribution_basis):
        refused(TiingoRefusal.PAYLOAD_MISMATCH, call, tampered)
    for change in (
        {"fund_ids": ("VTI",)},
        {"start": day("2014-06-03")},
        {"end_exclusive": day("2014-06-13")},
    ):
        mismatched = dataclasses.replace(retrieval, **change)
        refused(TiingoRefusal.PAYLOAD_MISMATCH, verified_batch, mismatched)


# --- the stored payload ----------------------------------------------------------


def test_the_payload_holds_only_the_allowlisted_fields_and_f1_hashes_it(tmp_path, verified):
    raw = fetch(funds=("VTI", "AGG"))
    envelope = json.loads(raw.raw_payload)
    assert set(envelope) == {"envelope_version", "source_id", "start_date", "end_date"} | {
        "fields",
        "prices",
    }  # no original-response hash, URL or metadata
    assert envelope["fields"] == list(ALLOWED_FIELDS)
    assert all(
        set(row) == set(ALLOWED_FIELDS) for rows in envelope["prices"].values() for row in rows
    )
    assert envelope["prices"]["VTI"] == price_list()  # the values used, unchanged
    assert raw.raw_payload == dumps_canonical(envelope).encode("utf-8")  # canonical
    assert raw.raw_payload == fetch(funds=("VTI", "AGG")).raw_payload  # deterministic
    for name in (*FORBIDDEN_NAMES, "volume", TOKEN):
        assert name.encode() not in raw.raw_payload
    parsed = parse_payload(raw.raw_payload)  # re-parsing needs no request
    assert parsed.prices.equals(raw.prices) and parsed.splits.equals(raw.splits)
    assert parsed.distributions.equals(raw.distributions)
    record = ingest_batch(tmp_path, raw).record
    assert record.raw_payload_sha256 == hashlib.sha256(raw.raw_payload).hexdigest()


def _tampered(change) -> bytes:
    envelope = json.loads(retrieve().raw_payload)
    change(envelope)
    return dumps_canonical(envelope).encode("utf-8")


def _adjusted_field(envelope) -> None:
    envelope["prices"]["VTI"][0]["adjClose"] = 30.0


def _unverified_split(envelope) -> None:
    envelope["prices"]["VTI"][4]["splitFactor"] = 2.0


def _past_the_cap(envelope) -> None:
    envelope["end_date"] = f"{MODEL_HOLDOUT_START.date()}"


def _reversed_dates(envelope) -> None:
    envelope["start_date"] = "2014-06-14"  # after its end_date, 2014-06-13


@pytest.mark.parametrize(
    "change, reason",
    [
        (_adjusted_field, TiingoRefusal.NON_ALLOWLISTED_FIELD),
        (_unverified_split, TiingoRefusal.UNVERIFIED_SPLIT_FACTOR),
        (_past_the_cap, RefusalReason.REQUEST_AFTER_CAP),
        (_reversed_dates, RefusalReason.INVALID_REQUEST),
    ],
)
def test_a_stored_payload_is_rechecked_on_re_parse(change, reason):
    refused(reason, parse_payload, _tampered(change))


# --- transport, failures and redaction ---------------------------------------------


@pytest.mark.parametrize(
    "status, reason",
    [
        (429, TiingoFailure.RATE_LIMITED),
        (500, TiingoFailure.SERVER_ERROR),
        (503, TiingoFailure.SERVER_ERROR),
        (401, TiingoFailure.HTTP_ERROR),
        (404, TiingoFailure.HTTP_ERROR),
        (302, TiingoFailure.HTTP_ERROR),
    ],
)
def test_an_error_status_refuses_the_batch_with_no_retry(status, reason):
    transport = FakeTransport(status={"VTI": status})
    error = failed(reason, retrieve, transport, funds=("VTI", "AGG"))
    assert len(transport.calls) == 1  # one call for VTI, no retry; AGG never requested
    assert (error.ticker, error.status) == ("VTI", status)
    assert str(error) == f"{reason.value}; ticker VTI; HTTP {status}"
    assert not leaked(error) and "api.tiingo.com" not in str(error)


class FakeResponse:
    def __init__(self, status: int, content: bytes) -> None:
        self.status, self._content = status, content

    def read(self) -> bytes:
        return self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class FakeOpener:
    """Stands in for the urllib opener; never touches the network."""

    def __init__(self, error_status: int | None = None) -> None:
        self.error_status = error_status
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error_status is not None:
            raise urllib.error.HTTPError(
                request.full_url,
                self.error_status,
                "Found",
                email.message.Message(),
                io.BytesIO(b"moved to https://elsewhere.example/"),
            )
        return FakeResponse(200, body(price_list()))


def _proxies(opener) -> list[dict]:
    return [
        item.proxies for item in opener.handlers if isinstance(item, urllib.request.ProxyHandler)
    ]


def test_the_default_transport_follows_no_redirect_and_ignores_proxies(monkeypatch):
    handler = tiingo._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere/") is None
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    [default] = _proxies(urllib.request.build_opener())  # the stdlib default reads it
    assert default["https"] == "http://proxy.invalid:3128"
    opener = tiingo.build_opener()  # ProxyHandler({}) replaces the environment-reading one
    assert _proxies(opener) == []
    assert any(isinstance(item, tiingo._NoRedirect) for item in opener.handlers)
    request = tiingo.build_request("https://api.tiingo.com/x", {"Authorization": f"Token {TOKEN}"})
    assert request.unredirected_hdrs == {"Authorization": f"Token {TOKEN}"}
    assert request.headers == {}  # never copied onto a redirected request


def test_a_redirect_maps_to_a_refusal_with_its_status_and_no_body():
    url = "https://api.tiingo.com/tiingo/daily/VTI/prices"
    assert tiingo.urllib_transport(url, {}, opener=FakeOpener(302)) == (302, b"")
    opener = FakeOpener()
    assert tiingo.urllib_transport(url, {}, opener=opener) == (200, body(price_list()))
    assert opener.requests[0][1] == tiingo.TIMEOUT_SECONDS
    openers = []

    def transport(address, headers):
        openers.append(FakeOpener(302))
        return tiingo.urllib_transport(address, headers, opener=openers[-1])

    error = failed(TiingoFailure.HTTP_ERROR, retrieve, transport)
    assert error.status == 302 and "elsewhere" not in str(error) and not leaked(error)
    [(request, _)] = openers[0].requests
    assert request.unredirected_hdrs == {"Authorization": f"Token {TOKEN}"}
    assert TOKEN not in request.full_url


@pytest.mark.parametrize(
    "error",
    [
        ValueError(f"Invalid header value b'Token {TOKEN}'"),
        http.client.IncompleteRead(b"partial"),
        urllib.error.URLError(f"refused for {TOKEN}"),
        ConnectionError(f"Authorization: Token {TOKEN}"),
        TimeoutError(),
    ],
)
def test_each_expected_network_failure_is_reported_by_type_only(error):
    failure = failed(TiingoFailure.TRANSPORT_ERROR, retrieve, FakeTransport(error=error))
    assert str(failure) == f"transport_error; ticker VTI; {type(error).__name__}"
    assert failure.__cause__ is None and failure.__context__ is None
    assert not leaked(failure)


@pytest.mark.parametrize("error", [KeyError("bug"), RuntimeError("bug"), TypeError("bug")])
def test_an_unexpected_exception_propagates_unchanged(error):
    with pytest.raises(type(error)) as caught:
        retrieve(FakeTransport(error=error))
    assert caught.value is error


def test_a_transport_returning_the_wrong_shape_is_refused():
    failed(TiingoFailure.TRANSPORT_ERROR, retrieve, lambda url, headers: (200, "text"))


def test_the_key_comes_from_the_environment_and_is_never_echoed(monkeypatch):
    monkeypatch.delenv(tiingo.TOKEN_ENV, raising=False)
    error = failed(TiingoFailure.MISSING_TOKEN, tiingo.env_token)
    assert tiingo.TOKEN_ENV in str(error)
    monkeypatch.setenv(tiingo.TOKEN_ENV, TOKEN)
    assert tiingo.env_token() == TOKEN
    transport = FakeTransport()
    TiingoAdapter(transport=transport, clock=lambda: RETRIEVED).retrieve(
        ["VTI"], day(START), day(END_EXCLUSIVE)
    )
    assert transport.calls[0][1] == {"Authorization": f"Token {TOKEN}"}
    spaced = f"{TOKEN} \r\nX-Injected: 1"
    error = failed(TiingoFailure.INVALID_TOKEN, retrieve, FakeTransport(), token=spaced)
    assert not leaked(error)


def test_no_log_or_output_carries_the_key(caplog, capsys):
    caplog.set_level(logging.DEBUG)
    retrieve()
    with pytest.raises(TiingoRequestFailed):
        retrieve(FakeTransport(status={"VTI": 429}))
    with pytest.raises(TiingoRequestFailed):
        retrieve(FakeTransport(error=OSError(TOKEN)))
    captured = capsys.readouterr()
    assert TOKEN not in caplog.text + captured.out + captured.err


# --- boundaries ----------------------------------------------------------------


def _direct_imports(path: Path) -> set[str]:
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_vendor_code_uses_no_third_party_client_and_the_store_no_vendor():
    for path in (SRC / "data" / "vendors").glob("*.py"):
        roots = {name.split(".")[0] for name in _direct_imports(path)}
        assert not roots & {"requests", "urllib3", "yfinance", "httpx", "aiohttp"}, path
    for path in (SRC / "data" / "funds").glob("*.py"):
        imported = _direct_imports(path)
        assert not [name for name in imported if name.startswith("stock_agent.data.vendors")]
        assert not {name.split(".")[0] for name in imported} & {"urllib", "http", "socket"}


def test_loading_the_store_loads_no_vendor_or_network_module():
    script = (
        "import json, sys\n"
        "import stock_agent.data.funds.ingest, stock_agent.data.funds.reader\n"
        "print(json.dumps(sorted(sys.modules)))"
    )
    environment = {**os.environ, "PYTHONPATH": str(SRC.parent)}
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=environment, check=True
    )
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert not [name for name in loaded if name.startswith("stock_agent.data.vendors")]
    assert not loaded & {"urllib.request", "http.client", "requests", "yfinance"}


# --- the audit script (offline; the adapter runs over the fake transport) --------

SCRIPT = SRC.parents[1] / "scripts" / "audit_fund_coverage.py"
_spec = importlib.util.spec_from_file_location("stock_agent_audit_fund_coverage", SCRIPT)
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(audit)


def never():
    raise AssertionError("the adapter must not be built")


@pytest.fixture
def repo(tmp_path) -> Path:
    path = tmp_path / "repo"
    (path / "data" / "raw").mkdir(parents=True)
    return path


def _tree(path: Path) -> set[Path]:
    return set(path.rglob("*"))


def run_audit(root, repo, transport, funds=("VTI", "AGG"), start=AUDIT_START, end=AUDIT_END):
    argv = ["--root", str(root), "--funds", *funds, "--start", start]
    argv += ["--end-exclusive", end, "--authorized"]
    return audit.main(argv, adapter_factory=lambda: adapter(transport), project_root=repo)


def test_the_audit_refuses_to_run_without_authorization(tmp_path, repo, capsys):
    root = tmp_path / "private"
    code = audit.main(["--root", str(root)], adapter_factory=never, project_root=repo)
    assert code == 2 and not root.exists()
    assert "--authorized" in capsys.readouterr().err


@pytest.mark.parametrize("inside", ["store", "data", "data/raw", "src/data/raw/x", ".."])
def test_the_audit_refuses_a_root_that_would_put_vendor_data_in_the_repository(
    tmp_path, repo, inside, capsys
):
    before = _tree(tmp_path)
    argv = ["--root", str(repo / inside), "--authorized"]
    assert audit.main(argv, adapter_factory=never, project_root=repo) == 2
    assert "Refused" in capsys.readouterr().err
    assert _tree(tmp_path) == before


def test_private_roots_lie_outside_every_repository_or_under_its_data_raw(tmp_path, repo):
    allowed = repo / "data" / "raw" / "tiingo"
    assert audit.private_root(allowed, repo) == allowed.resolve()
    assert audit.private_root(tmp_path / "elsewhere", repo) == (tmp_path / "elsewhere").resolve()
    (tmp_path / "link").symlink_to(repo / "src")
    with pytest.raises(ValueError):
        audit.private_root(tmp_path / "link", repo)
    # Another clone or worktree: an ancestor holding a `.git` file or directory.
    clone = tmp_path / "clone"
    (clone / "data" / "raw").mkdir(parents=True)
    (clone / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    for refused_root in (clone, clone / "store", clone / "data" / "raw", clone / "data" / "x"):
        with pytest.raises(ValueError):
            audit.private_root(refused_root, repo)
    inside_raw = clone / "data" / "raw" / "tiingo"
    assert audit.private_root(inside_raw, repo) == inside_raw.resolve()


def test_a_case_variant_path_into_the_repository_is_refused(tmp_path, repo):
    variant = repo.parent / repo.name.upper()
    if variant.exists():  # a case-insensitive file system, as macOS uses by default
        with pytest.raises(ValueError):
            audit.private_root(variant / "store", repo)
        assert audit.private_root(variant / "data" / "raw" / "x", repo)
    else:  # a case-sensitive file system: a different directory, not the repository
        assert audit.private_root(variant / "store", repo) == (variant / "store").resolve()


def test_an_existing_root_open_to_others_is_refused_and_never_changed(tmp_path, repo, capsys):
    existing = tmp_path / "existing"
    existing.mkdir()
    for mode in (0o755, 0o710, 0o701):
        os.chmod(existing, mode)
        argv = ["--root", str(existing), "--authorized"]
        assert audit.main(argv, adapter_factory=never, project_root=repo) == 2
        error = capsys.readouterr().err
        assert "restrict it yourself" in error and "chmod 700" in error
        assert stat.S_IMODE(existing.stat().st_mode) == mode  # the audit never changes it
    afile = tmp_path / "afile"
    afile.write_text("", encoding="utf-8")
    argv = ["--root", str(afile), "--authorized"]
    assert audit.main(argv, adapter_factory=never, project_root=repo) == 2
    assert "not a directory" in capsys.readouterr().err
    # A private existing root is used as it is; a missing root is created private.
    os.chmod(existing, 0o700)
    fresh = tmp_path / "fresh" / "store"
    for root in (existing, fresh):
        transport = FakeTransport(status={"VTI": 429})
        code = run_audit(root, repo, transport, funds=("VTI",), start=START, end=END_EXCLUSIVE)
        assert code == 1 and len(transport.calls) == 1  # past the root check, then refused
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
    capsys.readouterr()


def _section(lines: list[str], fund: str, following: str | None) -> list[str]:
    end = lines.index(following) if following else len(lines)
    return lines[lines.index(fund) + 1 : end]


def test_an_offline_audit_verifies_every_fund_then_stores(tmp_path, repo, capsys, monkeypatch):
    dates = quarterly(AUDIT_START, AUDIT_END)
    # Invented primary-source entries matching the fixtures (test-only, not real amounts).
    entries = invented_vti_entries(0.60) | {("AGG", d): 0.40 for d in dates[:2]}
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
    rows = {"VTI": vti_rows(0.60), "AGG": agg_rows()}
    transport = FakeTransport(prices=rows)
    root = tmp_path / "private"
    code = run_audit(root, repo, transport)
    out = capsys.readouterr().out
    assert code == 0, out
    assert len(transport.calls) == 2
    lines = out.splitlines()
    vti, agg = _section(lines, "VTI", "AGG"), _section(lines, "AGG", None)
    assert "field allowlist: every element had exactly date, close, divCash, splitFactor" in out
    assert f"  price rows (unadjusted): {len(rows['VTI'])}" in vti
    assert any("missing NYSE sessions: 1 " in line and "2008-01-15" in line for line in vti)
    assert "  distributions per year: 2007:3 2008:4 2009:2" in vti
    split_row = next(row for row in rows["VTI"] if row["date"].startswith(VTI_SPLIT))
    previous = rows["VTI"][rows["VTI"].index(split_row) - 1]
    ratio = split_row["close"] * 2 / previous["close"]
    assert (
        f"  splits: 2008-06-18 x2 (verified; raw close x factor / previous raw close {ratio:.2f})"
        in vti
    )
    assert any(
        line.startswith(
            "  distribution basis: verified against primary-source amounts; split check agreed ("
        )
        and "listed matches 1 before, 1 after" in line
        and "= 1.00, within" in line
        for line in vti
    )
    assert any(
        line.startswith(
            "  distribution basis: verified against primary-source amounts; split check no_split ("
        )
        and "2 (needs 2), 2 match, 0 do not" in line
        for line in agg
    )
    assert "vendor-adjusted reconciliation: not performed" in out
    assert "total return: not computed by the audit (OD8 group A descriptive calculations)" in out
    assert "stored tables match a re-parse of the stored raw payload: True" in out
    assert TOKEN not in out and "40.0" not in out and "0.6" not in out
    batch = read_batch(next(root.iterdir()))
    assert batch.record.vintage_note.endswith(
        "distribution basis verified against primary-source amounts: VTI, AGG"
    )
    assert _tree(repo) == {repo / "data", repo / "data" / "raw"}  # nothing written in the repo


def test_vti_alone_is_refused_without_primary_source_amounts(tmp_path, repo, capsys):
    root = tmp_path / "private"
    transport = FakeTransport(prices={"VTI": vti_rows(0.60)})
    code = run_audit(root, repo, transport, funds=("VTI",))  # the real, empty allowlist
    captured = capsys.readouterr()
    assert code == 1 and list(root.iterdir()) == []
    assert "NOT STORED unverified_distribution_basis: VTI;" in captured.err
    assert "  distribution basis: NOT verified; split check agreed (" in captured.out


def test_one_unverified_fund_means_nothing_is_stored(tmp_path, repo, capsys, monkeypatch):
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", invented_vti_entries(0.60))
    root = tmp_path / "private"
    transport = FakeTransport(prices={"VTI": vti_rows(0.60), "AGG": agg_rows()})
    code = run_audit(root, repo, transport)  # AGG has no listed amounts
    captured = capsys.readouterr()
    assert code == 1 and list(root.iterdir()) == []
    assert "NOT STORED unverified_distribution_basis: AGG;" in captured.err
    assert (
        "  distribution basis: NOT verified; split check no_split (verified distributions in "
        "the window: 0 (needs 2)" in captured.out
    )
    verified_vti = "  distribution basis: verified against primary-source amounts; split check"
    assert verified_vti in captured.out  # VTI passed
    assert "stored" not in captured.out.replace("NOT STORED", "")


def test_split_adjusted_div_cash_is_never_stored(tmp_path, repo, capsys, monkeypatch):
    # Two listed post-split amounts match, but the pre-split side was served split-adjusted.
    entries = invented_vti_entries(0.30, before=0, after=2)
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", entries)
    root = tmp_path / "private"
    transport = FakeTransport(prices={"VTI": vti_rows(pre_split_dividend=0.30)})
    code = run_audit(root, repo, transport, funds=("VTI",))
    captured = capsys.readouterr()
    assert code == 1 and list(root.iterdir()) == []
    assert "NOT STORED unverified_distribution_basis: VTI;" in captured.err
    assert "split check failed" in captured.out and "= 0.50, outside" in captured.out


def test_a_failed_re_parse_comparison_exits_with_an_error(tmp_path, repo, capsys, monkeypatch):
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", invented_vti_entries(0.60))
    monkeypatch.setattr(audit, "reparse_matches", lambda batch: False)
    transport = FakeTransport(prices={"VTI": vti_rows(0.60)})
    code = run_audit(tmp_path / "private", repo, transport, funds=("VTI",))
    captured = capsys.readouterr()
    assert code == 1
    assert "stored tables match a re-parse of the stored raw payload: False" in captured.out
    assert "FAILED the stored tables differ from a re-parse" in captured.err


def test_a_stored_audit_holds_no_key_adjusted_level_or_metadata(
    tmp_path, repo, capsys, monkeypatch
):
    monkeypatch.setattr(tiingo, "VERIFIED_DISTRIBUTIONS", invented_vti_entries(0.60))
    root = tmp_path / "private"
    assert run_audit(root, repo, FakeTransport(prices={"VTI": vti_rows(0.60)}), funds=("VTI",)) == 0
    capsys.readouterr()
    [batch] = list(root.iterdir())
    assert {path.name for path in batch.iterdir()} == FILE_SET  # only the F1 files
    assert TOKEN.encode() not in stored_bytes(root)  # the sentinel key never reaches the store
    for path in batch.iterdir():
        content = path.read_bytes()
        names = [name for name in FORBIDDEN_NAMES if name.encode() in content]
        if path.name == "batch.json":
            # Named only in the notes, as fields that were not requested; never as a key.
            record = json.loads(content)
            assert all(name in record["vintage_note"] for name in names)
            assert not [name for name in names if f'"{name}":'.encode() in content]
        else:  # the payload, the tables and the manifest never name them at all
            assert names == [], path.name
    stored = read_batch(batch)
    assert set(stored.prices["adjustment_basis"]) == {RAW} and stored.prices["volume"].isna().all()


def test_a_refused_fetch_reports_its_reason_and_stores_nothing(tmp_path, repo, capsys):
    root = tmp_path / "private"
    rows = price_list()
    rows[1] = {**rows[1], "adjClose": 98765.4321}
    cases = [
        (FakeTransport(status={"VTI": 429}), "REFUSED rate_limited; ticker VTI; HTTP 429"),
        (FakeTransport(prices={"VTI": rows}), "REFUSED non_allowlisted_field: VTI prices[1]"),
    ]
    for transport, message in cases:
        code = run_audit(root, repo, transport, funds=("VTI",), start=START, end=END_EXCLUSIVE)
        captured = capsys.readouterr()
        assert code == 1 and message in captured.err
        assert "98765" not in captured.out + captured.err  # a refused response is never shown
        assert list(root.iterdir()) == [] and TOKEN not in captured.out + captured.err


def test_an_audit_of_a_fund_with_no_prices_reports_and_stores_nothing(tmp_path, repo, capsys):
    root = tmp_path / "private"
    transport = FakeTransport(prices={"VTI": []})
    code = run_audit(root, repo, transport, funds=("VTI",), start=START, end=END_EXCLUSIVE)
    captured = capsys.readouterr()
    assert code == 1 and list(root.iterdir()) == []
    assert "  sessions: none; missing NYSE sessions: 10" in captured.out.splitlines()
    assert "  distributions per year: none" in captured.out and "  splits: none" in captured.out
