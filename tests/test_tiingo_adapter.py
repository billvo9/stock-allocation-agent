"""
Offline tests of the Tiingo adapter and the coverage-audit script. Every
response is an invented fixture shaped like Tiingo's documented JSON; tickers
are labels only and no value is a real price. The default transport is never
called, and the key is an invented sentinel.
"""

import ast
import hashlib
import http.client
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
import traceback
import urllib.error
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
)
from stock_agent.data.vendors import tiingo
from stock_agent.data.vendors.tiingo import (
    ALLOWED_FIELDS,
    TiingoAdapter,
    TiingoFailure,
    TiingoRefusal,
    TiingoRequestFailed,
    parse_payload,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

TOKEN = "sentinel-tiingo-key-0123456789abcdef"  # invented; never a real key
RETRIEVED = pd.Timestamp("2026-10-14T09:30:00.000000Z")
START, END_EXCLUSIVE = "2014-06-02", "2014-06-14"  # ten sessions, 06-02 to 06-13
EX_DATE = "2014-06-05"
VTI_SPLIT = "2008-06-18"  # the one verified split
SPLIT_START, SPLIT_END = "2008-06-09", "2008-06-28"
RAW = "unadjusted"
# Names that must never reach a stored artifact.
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
        "volume": 1_000,
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

    dividends = {EX_DATE: 0.125} if dividends is None else dividends
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
        rows.append(
            element(
                session,
                close,
                volume=1_000 + index,
                divCash=dividends.get(text, 0.0),
                splitFactor=factor,
            )
        )
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


def fetch(transport=None, funds=("VTI",), start=START, end=END_EXCLUSIVE, **kwargs):
    transport = transport if transport is not None else FakeTransport()
    return adapter(transport, **kwargs).fetch(list(funds), day(start), day(end))


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
    assert defaults["transport"].default is tiingo.urllib_transport  # never called in tests
    assert defaults["token_provider"].default is tiingo.env_token
    assert TOKEN not in repr(adapter(FakeTransport()))


def test_one_prices_call_per_fund_asks_only_for_allowlisted_columns():
    transport = FakeTransport()
    fetch(transport, funds=("VTI", "AGG"))
    query = (
        "startDate=2014-06-02&endDate=2014-06-13&format=json&resampleFreq=daily"
        "&columns=date,close,volume,divCash,splitFactor"
    )
    base = tiingo.BASE_URL
    assert [url for url, _ in transport.calls] == [
        f"{base}/VTI/prices?{query}",
        f"{base}/AGG/prices?{query}",
    ]  # no metadata call
    assert all(headers == {"Authorization": f"Token {TOKEN}"} for _, headers in transport.calls)
    assert not [url for url, _ in transport.calls if TOKEN in url]


def test_a_request_ending_at_the_holdout_start_asks_for_its_eve():
    transport = FakeTransport(prices={"SHV": price_list("2024-12-23", "2025-01-01")})
    raw = fetch(transport, funds=("SHV",), start="2024-12-23", end=f"{MODEL_HOLDOUT_START.date()}")
    assert "endDate=2024-12-31&" in transport.calls[0][0]
    assert raw.end_exclusive == MODEL_HOLDOUT_START
    assert raw.prices["session_date"].max() == day("2024-12-31")


@pytest.mark.parametrize(
    "funds, end, reason",
    [
        (["VTI"], MODEL_HOLDOUT_START + pd.Timedelta(days=1), RefusalReason.REQUEST_AFTER_CAP),
        (["VTI"], day("2026-10-14"), RefusalReason.REQUEST_AFTER_CAP),
        (["vti"], day(END_EXCLUSIVE), RefusalReason.INVALID_REQUEST),
    ],
)
def test_a_post_cutoff_or_malformed_request_contacts_nothing_and_reads_no_key(funds, end, reason):
    reads = []

    def provider():
        reads.append(1)
        return TOKEN

    transport = FakeTransport()
    late = TiingoAdapter(token_provider=provider, transport=transport, clock=lambda: RETRIEVED)
    refused(reason, late.fetch, funds, day(START), end)
    assert transport.calls == [] and reads == []


# --- mapping -------------------------------------------------------------------


def test_each_session_maps_to_one_unadjusted_close():
    raw = fetch()
    prices, expected = raw.prices, price_list()
    assert len(prices) == len(nyse.sessions(day(START), day("2014-06-13"))) == 10
    assert set(prices["adjustment_basis"]) == {RAW}
    assert prices["close"].tolist() == [row["close"] for row in expected]
    assert prices["volume"].tolist() == [row["volume"] for row in expected]
    assert set(prices["source_id"]) == {"tiingo"} and "batch_id" not in prices.columns
    assert raw.splits.empty  # every factor is 1


def test_div_cash_maps_to_distributions_with_no_invented_dates_or_type():
    distributions = fetch().distributions
    assert len(distributions) == 1
    row = distributions.iloc[0]
    assert row["ex_date"] == day(EX_DATE) and row["amount"] == 0.125
    assert row["amount_basis"] == "per_share_as_paid"
    assert row["distribution_type"] == "unknown" and row["source_sequence"] == 0
    for column in ("declaration_date", "record_date", "pay_date"):
        assert distributions[column].isna().all()
        assert str(distributions[column].dtype) == "datetime64[us, UTC]"


def test_the_batch_records_instruments_notes_and_retrieval_time():
    raw = fetch(funds=("VTI", "AGG"))
    assert raw.fund_ids == ("VTI", "AGG")
    assert raw.instruments == (Instrument("VTI", "VTI", None), Instrument("AGG", "AGG", None))
    assert raw.retrieved_at == RETRIEVED
    assert raw.adapter_id == "tiingo_daily" and raw.adapter_version == "3"
    assert "Internal Use Only" in raw.terms_note and "no redistribution" in raw.terms_note
    assert "not a point-in-time vintage" in raw.vintage_note
    for name in ("metadata endpoint", "adjClose", "adjVolume", "high", "low", "open"):
        assert name in raw.vintage_note  # the unrequested fields are recorded


# --- the field allowlist ---------------------------------------------------------


@pytest.mark.parametrize("field", ["adjClose", "adjVolume", "high", "fxRate"])
def test_any_non_allowlisted_field_refuses_the_batch(field):
    rows = price_list()
    rows[2] = {**rows[2], field: 12345.678}
    error = refused(TiingoRefusal.NON_ALLOWLISTED_FIELD, fetch, FakeTransport(prices={"VTI": rows}))
    assert field in str(error) and "12345" not in str(error)  # names only, never a value


def test_a_missing_allowlisted_field_refuses_the_batch():
    rows = price_list()
    del rows[3]["divCash"]
    refused(RefusalReason.SCHEMA_MISMATCH, fetch, FakeTransport(prices={"VTI": rows}))


def _with_first(**changes) -> list[dict]:
    rows = price_list()
    rows[0] = {**rows[0], **changes}
    return rows


FAILURES = {
    "string close": (_with_first(close="40.0"), RefusalReason.INVALID_FIELD),
    "null volume": (_with_first(volume=None), RefusalReason.INVALID_FIELD),
    "boolean split factor": (_with_first(splitFactor=True), RefusalReason.INVALID_FIELD),
    "string dividend": (_with_first(divCash="0.1"), RefusalReason.INVALID_FIELD),
    "fractional volume": (_with_first(volume=10.5), RefusalReason.INVALID_VALUE),
    "zero close": (_with_first(close=0.0), RefusalReason.INVALID_VALUE),
    "negative dividend": (_with_first(divCash=-0.1), RefusalReason.INVALID_VALUE),
    "zero split factor": (_with_first(splitFactor=0.0), RefusalReason.INVALID_VALUE),
    "NaN dividend": (_with_first(divCash=float("nan")), RefusalReason.INVALID_JSON),
    "infinite close": (_with_first(close=float("inf")), RefusalReason.INVALID_JSON),
    "overflowing dividend": (
        body(price_list()).replace(b'"divCash": 0.125', b'"divCash": 1e999'),
        RefusalReason.INVALID_JSON,
    ),
    "integer beyond float range": (_with_first(close=10**400), RefusalReason.INVALID_VALUE),
    "volume beyond int64": (_with_first(volume=2**64), RefusalReason.INVALID_VALUE),
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
def test_a_malformed_response_refuses_the_batch(case):
    payload, reason = FAILURES[case]
    assert b'"divCash": 0.125' in body(price_list())  # the overflow case edits a real token
    error = refused(reason, fetch, FakeTransport(prices={"VTI": payload}))
    assert TOKEN not in str(error)


def test_a_row_on_or_after_the_holdout_start_refuses_the_batch():
    rows = price_list("2024-12-23", "2025-01-01") + [element(day("2025-01-02"), 41.0)]
    error = refused(
        RefusalReason.LOCKBOX_VIOLATION,
        fetch,
        FakeTransport(prices={"SHV": rows}),
        funds=("SHV",),
        start="2024-12-23",
        end=f"{MODEL_HOLDOUT_START.date()}",
    )
    assert "2025-01-02" in str(error)


# --- splits ----------------------------------------------------------------------


def split_fetch(rows, fund="VTI", start=SPLIT_START):
    return fetch(FakeTransport(prices={fund: rows}), funds=(fund,), start=start, end=SPLIT_END)


def test_a_verified_split_is_stored_with_unadjusted_closes():
    raw = split_fetch(price_list(SPLIT_START, SPLIT_END, split=(VTI_SPLIT, 2.0)))
    assert raw.splits[["fund_id", "effective_date", "ratio_new_per_old"]].values.tolist() == [
        ["VTI", day(VTI_SPLIT), 2.0]
    ]
    assert set(raw.prices["adjustment_basis"]) == {RAW}
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
    refused(
        expected, fetch, FakeTransport(prices={fund: rows}), funds=(fund,), start=start, end=end
    )


def test_a_request_starting_on_the_split_cannot_verify_it():
    rows = price_list(VTI_SPLIT, SPLIT_END, split=(VTI_SPLIT, 2.0))
    refused(TiingoRefusal.SPLIT_PRICE_MISMATCH, split_fetch, rows, start=VTI_SPLIT)


# --- the stored payload ----------------------------------------------------------


def test_the_payload_holds_only_the_allowlisted_fields_and_f1_hashes_it(tmp_path):
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
    for name in (*FORBIDDEN_NAMES, TOKEN):
        assert name.encode() not in raw.raw_payload
    parsed = parse_payload(raw.raw_payload)  # re-parsing needs no request
    assert parsed.prices.equals(raw.prices) and parsed.splits.equals(raw.splits)
    assert parsed.distributions.equals(raw.distributions)
    record = ingest_batch(tmp_path, raw).record
    assert record.raw_payload_sha256 == hashlib.sha256(raw.raw_payload).hexdigest()


def _tampered(change) -> bytes:
    envelope = json.loads(fetch().raw_payload)
    change(envelope)
    return dumps_canonical(envelope).encode("utf-8")


def _adjusted_field(envelope) -> None:
    envelope["prices"]["VTI"][0]["adjClose"] = 30.0


def _unverified_split(envelope) -> None:
    envelope["prices"]["VTI"][4]["splitFactor"] = 2.0


def _past_the_cap(envelope) -> None:
    envelope["end_date"] = f"{MODEL_HOLDOUT_START.date()}"


@pytest.mark.parametrize(
    "change, reason",
    [
        (_adjusted_field, TiingoRefusal.NON_ALLOWLISTED_FIELD),
        (_unverified_split, TiingoRefusal.UNVERIFIED_SPLIT_FACTOR),
        (_past_the_cap, RefusalReason.REQUEST_AFTER_CAP),
    ],
)
def test_a_stored_payload_is_rechecked_on_re_parse(change, reason):
    refused(reason, parse_payload, _tampered(change))


# --- failures and redaction ----------------------------------------------------


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
    error = failed(reason, fetch, transport, funds=("VTI", "AGG"))
    assert len(transport.calls) == 1  # one call for VTI, no retry; AGG never requested
    assert (error.ticker, error.status) == ("VTI", status)
    assert str(error) == f"{reason.value}; ticker VTI; HTTP {status}"
    assert not leaked(error) and "api.tiingo.com" not in str(error)


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
    failure = failed(TiingoFailure.TRANSPORT_ERROR, fetch, FakeTransport(error=error))
    assert str(failure) == f"transport_error; ticker VTI; {type(error).__name__}"
    assert failure.__cause__ is None and failure.__context__ is None
    assert not leaked(failure)


@pytest.mark.parametrize("error", [KeyError("bug"), RuntimeError("bug"), TypeError("bug")])
def test_an_unexpected_exception_propagates_unchanged(error):
    with pytest.raises(type(error)) as caught:
        fetch(FakeTransport(error=error))
    assert caught.value is error


def test_a_transport_returning_the_wrong_shape_is_refused():
    failed(TiingoFailure.TRANSPORT_ERROR, fetch, lambda url, headers: (200, "text"))


def test_the_key_comes_from_the_environment_and_is_never_echoed(monkeypatch):
    monkeypatch.delenv(tiingo.TOKEN_ENV, raising=False)
    error = failed(TiingoFailure.MISSING_TOKEN, tiingo.env_token)
    assert tiingo.TOKEN_ENV in str(error)
    monkeypatch.setenv(tiingo.TOKEN_ENV, TOKEN)
    assert tiingo.env_token() == TOKEN
    transport = FakeTransport()
    TiingoAdapter(transport=transport, clock=lambda: RETRIEVED).fetch(
        ["VTI"], day(START), day(END_EXCLUSIVE)
    )
    assert transport.calls[0][1] == {"Authorization": f"Token {TOKEN}"}
    spaced = f"{TOKEN} \r\nX-Injected: 1"
    error = failed(TiingoFailure.INVALID_TOKEN, fetch, FakeTransport(), token=spaced)
    assert not leaked(error)


def test_no_log_or_output_carries_the_key(caplog, capsys):
    caplog.set_level(logging.DEBUG)
    fetch()
    with pytest.raises(TiingoRequestFailed):
        fetch(FakeTransport(status={"VTI": 429}))
    with pytest.raises(TiingoRequestFailed):
        fetch(FakeTransport(error=OSError(TOKEN)))
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
AUDIT_START, AUDIT_END = "2007-06-01", "2009-07-01"


def never():
    raise AssertionError("the adapter must not be built")


@pytest.fixture
def repo(tmp_path) -> Path:
    path = tmp_path / "repo"
    (path / "data" / "raw").mkdir(parents=True)
    return path


def _tree(path: Path) -> set[Path]:
    return set(path.rglob("*"))


def audit_rows(pre_split_dividend: float) -> dict[str, list[dict]]:
    """VTI with its verified split and quarterly dividends; AGG without a split."""

    dates = quarterly(AUDIT_START, AUDIT_END)
    vti = {date: pre_split_dividend if date < VTI_SPLIT else 0.30 for date in dates}
    return {
        "VTI": price_list(
            AUDIT_START, AUDIT_END, dividends=vti, split=(VTI_SPLIT, 2.0), skip=("2008-01-15",)
        ),
        "AGG": price_list(AUDIT_START, AUDIT_END, dividends={date: 0.40 for date in dates}),
    }


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


def test_private_roots_lie_outside_the_repository_or_under_data_raw(tmp_path, repo):
    allowed = repo / "data" / "raw" / "tiingo"
    assert audit.private_root(allowed, repo) == allowed.resolve()
    assert audit.private_root(tmp_path / "elsewhere", repo) == (tmp_path / "elsewhere").resolve()
    (tmp_path / "link").symlink_to(repo / "src")
    with pytest.raises(ValueError):
        audit.private_root(tmp_path / "link", repo)


def test_an_offline_audit_reports_coverage_splits_and_the_div_cash_basis(tmp_path, repo, capsys):
    rows = audit_rows(pre_split_dividend=0.60)
    transport = FakeTransport(prices=rows)
    root = tmp_path / "private"
    code = run_audit(root, repo, transport)
    out = capsys.readouterr().out
    assert code == 0, out
    assert len(transport.calls) == 2
    lines = out.splitlines()
    vti = lines[lines.index("VTI") + 1 : lines.index("AGG")]
    agg = lines[lines.index("AGG") + 1 :]
    assert "field allowlist: every element had exactly date, close, volume, divCash, " in out
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
    basis = "median of 4 before / median of 4 after = 2.00 -> per share as paid"
    assert f"  divCash share basis at 2008-06-18 x2: {basis}" in vti
    assert "  splits: none" in agg and not [line for line in agg if "share basis" in line]
    assert "vendor-adjusted reconciliation: not performed" in out
    assert "total return: pending OD8 group A (T5-A)" in out
    assert "stored tables match a re-parse of the stored raw payload: True" in out
    assert TOKEN not in out and "40.0" not in out and "0.6" not in out
    assert _tree(repo) == {repo / "data", repo / "data" / "raw"}  # nothing written in the repo


def test_an_offline_audit_stores_no_key_adjusted_level_or_metadata(tmp_path, repo, capsys):
    root = tmp_path / "private"
    assert run_audit(root, repo, FakeTransport(prices=audit_rows(0.60))) == 0
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
    assert set(read_batch(batch).prices["adjustment_basis"]) == {RAW}


def test_split_adjusted_div_cash_is_refused_before_anything_is_stored(tmp_path, repo, capsys):
    root = tmp_path / "private"
    code = run_audit(root, repo, FakeTransport(prices=audit_rows(pre_split_dividend=0.30)))
    captured = capsys.readouterr()
    assert code == 1
    assert "REFUSED divcash_basis_mismatch: VTI divCash looks split-adjusted" in captured.err
    assert list(root.iterdir()) == []


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


def test_an_audit_of_a_fund_with_no_prices_reports_every_session_missing(tmp_path, repo, capsys):
    transport = FakeTransport(prices={"VTI": []})
    code = run_audit(
        tmp_path / "private", repo, transport, funds=("VTI",), start=START, end=END_EXCLUSIVE
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "  sessions: none; missing NYSE sessions: 10" in out.splitlines()
    assert "  distributions per year: none" in out and "  splits: none" in out
