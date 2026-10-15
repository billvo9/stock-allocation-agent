"""
The Tiingo end-of-day adapter for the F1 fund-data store.

Built to the owner's requirements of 2026-10-14 and the independent review
that followed. Tiingo's endpoint shape and field meanings are recalled from its
documentation [unverified]; the owner-authorized coverage audit
(`scripts/audit_fund_coverage.py`) checks them.

Two phases (nothing reaches the store unverified)
    `TiingoAdapter.retrieve` requests, checks and parses in memory and returns
    a `Retrieval`. `verified_batch` turns a retrieval into an F1 `RawBatch` only
    when every fund passes `verify_distribution_basis`; otherwise it refuses
    with `unverified_distribution_basis`. It never trusts the retrieval's
    in-memory tables: it rebuilds them from the raw payload, refuses
    (`payload_mismatch`) if they differ or the payload's tickers and dates do
    not match the request, and stores the rebuilt tables. `fetch` (the F1 seam)
    does both, so a caller that passes its result to `ingest_batch` stores only
    verified data.

Request (prices only; the metadata endpoint is never called)
    The F1 request check refuses, before the key is read or any request is
    made, an `end_exclusive` after `MODEL_HOLDOUT_START` (so `endDate` is at
    most the day before it) and more than `MAX_TICKERS` funds. One call per
    fund, in order, each response checked before the next request:

        GET https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=<start>
            &endDate=<end_exclusive - 1 day>&format=json&resampleFreq=daily
            &columns=date,close,divCash,splitFactor

    `retrieved_at` is read from the injected clock after the last response.

Field allowlist (the lockbox information-content rule)
    `docs/product/architecture.md`, section 4, rule 7: "dated" means
    information content. Adjusted levels and vendor metadata carry information
    from after the holdout start, so they are never requested, parsed or
    stored; volume is not requested either (its split basis is unverified).
    Each element must hold exactly `ALLOWED_FIELDS`. Any other field refuses
    the batch with `non_allowlisted_field`, which is how the audit learns
    whether Tiingo honours `columns`. A refused response is never stored or
    logged; refusal details name fields, dates and indices, never a value.

Mapping
    close        -> `prices`, basis `unadjusted` only; volume NULL
    divCash > 0  -> `distributions`: amount = divCash, `per_share_as_paid`
                    (stored only after the basis is verified, below), type
                    `unknown`, sequence 0, declaration, record and pay dates NULL
    splitFactor  -> `splits` only for a share-count split in the reviewed
                    `VERIFIED_SPLITS` allowlist, on the same session, with the
                    same factor, a `discriminable` factor, and a raw close
                    consistent with an unadjusted price
                    (`split_close_consistent`). Any other factor != 1 refuses
                    the batch.

Distribution-basis verification (per fund; every fund must pass)
    Storage requires primary-source amounts for every fund (route (b)); the
    split ratio (route (a)) is a consistency check that can only fail a fund:
    (i)   at least `MIN_VERIFIED_AMOUNTS` of the owner's hand-filled anchors
          (`anchors.py`, a private git-ignored file) in the window, each
          matching the vendor's divCash within `AMOUNT_TOLERANCE`; a listed date
          with no served divCash is a mismatch, and any mismatch fails the fund;
    (ii)  for every verified split in the window, at least one listed match
          strictly before and one strictly after the split session (a vendor
          rescales every amount between two splits by one factor, so matches
          on one side prove nothing about the other);
    (iii) where at least `MIN_SIDE` distributions lie on each side of a
          verified split, median(before) / (factor x median(after)) must lie
          within `BAND` (log scale) of 1;
    (iv)  no distribution may have a verified split session as its ex-date
          (its share basis is unverifiable).
    Evidence level: a spot check, not a validation of the history. A pass
    shows that the listed ex-dates match the issuer's amounts and, assuming
    the vendor uses one basis between consecutive splits, that each split
    segment's amounts are per share as paid. It does not show that any other
    amount is correct, or that no distribution is missing or extra.
    Neither route detects a later split applied uniformly to the whole
    closing-price history; route (b) anchors amounts only. The anchors are
    passed in explicitly (`DistributionAnchors`, research scope here); no
    amount lives in code, and the adapter's default is an empty allowlist, so
    nothing is stored without them. The batch's vintage note records the
    anchors file's sha256, never an amount.

Raw payload
    The canonical JSON of the allowlisted fields actually used, keyed by
    ticker. F1 hashes it. Nothing else from the response is kept.

Secrets and failures
    The key is sent only in the `Authorization: Token <key>` header, never in a
    URL. It is read once per retrieval from the injected provider (by default
    `TIINGO_API_KEY`), held in a wrapper whose repr is redacted, and never
    stored on the adapter. Any status other than 200, including 429 and 5xx,
    refuses the batch with no retry. An expected network failure
    (`TRANSPORT_FAILURES`) is reported by type only, raised outside the except
    block (no chained exception, header or body); other exception types
    propagate unchanged. The default transport ignores environment proxies,
    follows no redirects and sends the key as an unredirected header.

Prospective mode
    `tiingo_prospective.py` reuses the request loop (`TiingoAdapter.request_elements`),
    the element checks (`checked_sessions`, with its own window refusal), the
    mapping and the gate. This module's research path is unchanged: it still
    refuses anything on or after `MODEL_HOLDOUT_START`.

Nothing here computes a return, adjusts a value or fills a gap.
"""

from __future__ import annotations

import http.client
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.funds.adapter import RawBatch
from stock_agent.data.funds.schema import (
    DISTRIBUTIONS,
    PRICES,
    SPLITS,
    AdjustmentBasis,
    AmountBasis,
    DistributionType,
    FundDataRefused,
    Instrument,
    RefusalReason,
    dumps_canonical,
    loads_strict,
    parse_date,
    refuse,
    utc_timestamp,
)
from stock_agent.data.funds.validation import validate_request
from stock_agent.data.vendors.anchors import AnchorScope, DistributionAnchors
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

TOKEN_ENV = "TIINGO_API_KEY"
BASE_URL = "https://api.tiingo.com/tiingo/daily"
SOURCE_ID = "tiingo"
TIMEOUT_SECONDS = 30.0
ENVELOPE_VERSION = "4"
MAX_TICKERS = 20

# The only fields requested, parsed and stored.
ALLOWED_FIELDS = ("date", "close", "divCash", "splitFactor")
NOT_REQUESTED = (
    "the metadata endpoint; price fields adjClose, adjHigh, adjLow, adjOpen, adjVolume, "
    "high, low, open, volume"
)

# Externally verified share-count splits: (ticker, session) -> new shares per old.
# Code-reviewed; add an entry only with a primary source.
VERIFIED_SPLITS = {
    # SEC EDGAR, Vanguard Index Funds (CIK 36405), accession 0000932471-08-001251:
    # Form 497 filed 2008-06-09, U.S. Stock ETF supplement. 2-for-1 split, record date
    # 2008-06-13, new shares distributed 2008-06-17, split-adjusted trading expected to
    # begin 2008-06-18. The FY2008 annual report confirms the record and distribution
    # dates; no issuer document confirms the first trading date after the event.
    ("VTI", "2008-06-18"): 2.0,
}

# Primary-source distribution amounts are never in code: issuer terms forbid
# redistribution and this repository is public. They come from the owner's
# private, hand-filled anchors file (`anchors.py`).
NO_ANCHORS = DistributionAnchors.empty(AnchorScope.RESEARCH)

MIN_SIDE = 4  # distributions required on each side of a verified split
MIN_VERIFIED_AMOUNTS = 2  # matching anchors required in the window
AMOUNT_TOLERANCE = 1e-6

TERMS_NOTE = (
    "Tiingo free tier, Internal Use Only; no redistribution; "
    "terms https://www.tiingo.com/about/terms accessed 2026-10-13"
)
VINTAGE_NOTE = (
    "close, divCash and splitFactor as served on the retrieval date; not a point-in-time "
    "vintage; the ticker-to-security mapping and the set of sessions served are Tiingo's "
    "view at retrieval; distribution amounts are stored only after a spot check against "
    "sampled primary-source amounts (the owner's private, hand-filled anchors), which tests the "
    "share basis but does not validate any other amount or that no distribution is missing; "
    "neither that "
    "spot check nor the split checks detect a "
    "later split applied uniformly to the whole closing-price history, because closes are "
    "checked against an external fact only at verified split sessions; "
    f"requested columns {', '.join(ALLOWED_FIELDS)} only; not requested: {NOT_REQUESTED}; "
    "splits stored only from the verified share-count allowlist"
)

# The network failures reported by type only. URLError and TimeoutError are
# OSErrors; http.client raises ValueError for an invalid header value, whose
# text would contain the key.
TRANSPORT_FAILURES = (
    urllib.error.URLError,
    http.client.HTTPException,
    OSError,
    TimeoutError,
    ValueError,
)

_ENVELOPE_FIELDS = ("envelope_version", "source_id", "start_date", "end_date", "fields", "prices")
_TICKER = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,15}")
_TOKEN = re.compile(r"[\x21-\x7e]{1,256}")  # printable ASCII, no spaces or line breaks
_SESSION_STAMP = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})T00:00:00\.000Z")
_ONE_DAY = pd.Timedelta(days=1)

TokenProvider = Callable[[], str]
Transport = Callable[[str, Mapping[str, str]], tuple[int, bytes]]
Clock = Callable[[], pd.Timestamp]


# The tolerance on |log ratio| for every check expected near 1 (the raw close
# across a split, and the split's distribution ratio): a 25% move either way.
BAND = math.log(1.25)


def discriminable(factor: float) -> bool:
    """
    Whether a split factor can be told from no split by a check with `BAND`:
    |log factor| >= 3 x BAND (about factor >= 1.953 or <= 0.512). A ratio
    within BAND of the factor is then at least 2 x BAND away from 1.
    """

    return factor > 0 and abs(math.log(factor)) >= 3 * BAND


def split_close_consistent(ratio: float) -> bool:
    """(raw close x factor) / previous raw close near 1, as an unadjusted price gives."""

    return ratio > 0 and abs(math.log(ratio)) < BAND


class TiingoRefusal(StrEnum):
    NON_ALLOWLISTED_FIELD = "non_allowlisted_field"
    UNVERIFIED_SPLIT_FACTOR = "unverified_split_factor"
    SPLIT_FACTOR_MISMATCH = "split_factor_mismatch"
    SPLIT_PRICE_MISMATCH = "split_price_mismatch"
    SPLIT_FACTOR_NOT_DISCRIMINABLE = "split_factor_not_discriminable"
    UNVERIFIED_DISTRIBUTION_BASIS = "unverified_distribution_basis"
    PAYLOAD_MISMATCH = "payload_mismatch"


class TiingoDataRefused(FundDataRefused):
    """
    A Tiingo-specific refusal of the whole batch. `reason` is a `TiingoRefusal`
    (F1's `RefusalReason` stays unchanged); F1 callers catch it as `FundDataRefused`.
    """

    def __init__(self, reason: TiingoRefusal, detail: str) -> None:
        super().__init__(reason, detail)


class TiingoFailure(StrEnum):
    MISSING_TOKEN = "missing_token"
    INVALID_TOKEN = "invalid_token"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    HTTP_ERROR = "http_error"
    TRANSPORT_ERROR = "transport_error"


class TiingoRequestFailed(RuntimeError):
    """
    A Tiingo request failed and the batch is refused. The message is built
    only from the reason, ticker, status and a fixed detail; never from a
    header, the key, a URL or a response body.
    """

    def __init__(
        self,
        reason: TiingoFailure,
        *,
        ticker: str | None = None,
        status: int | None = None,
        detail: str | None = None,
    ) -> None:
        parts = [reason.value]
        if ticker is not None:
            parts.append(f"ticker {ticker}")
        if status is not None:
            parts.append(f"HTTP {status}")
        if detail is not None:
            parts.append(detail)
        super().__init__("; ".join(parts))
        self.reason = reason
        self.ticker = ticker
        self.status = status


def env_token() -> str:
    """The key from `TIINGO_API_KEY`; refuses (without echoing anything) when unset."""

    value = os.environ.get(TOKEN_ENV, "")
    if not value.strip():
        raise TiingoRequestFailed(
            TiingoFailure.MISSING_TOKEN, detail=f"set {TOKEN_ENV} in the local environment"
        )
    return value


class _Token:
    """The key with a redacted repr; only `headers()` reveals it."""

    __slots__ = ("_value",)

    def __init__(self, value: object) -> None:
        if not isinstance(value, str) or not _TOKEN.fullmatch(value):
            del value  # keep the rejected value out of traceback locals
            raise TiingoRequestFailed(
                TiingoFailure.INVALID_TOKEN, detail="the key must be printable ASCII, no spaces"
            )
        self._value = value

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._value}"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError with the 3xx status


def build_opener() -> urllib.request.OpenerDirector:
    """An opener that ignores environment proxies and follows no redirects."""

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def build_request(url: str, headers: Mapping[str, str]) -> urllib.request.Request:
    """A GET whose headers are unredirected: never forwarded to another URL."""

    request = urllib.request.Request(url, method="GET")
    for name, value in headers.items():
        request.add_unredirected_header(name, value)
    return request


def urllib_transport(
    url: str,
    headers: Mapping[str, str],
    *,
    timeout: float = TIMEOUT_SECONDS,
    opener: urllib.request.OpenerDirector | None = None,
) -> tuple[int, bytes]:
    """
    GET with the standard library. Returns (status, body); the body is empty
    for an error status (a redirect included). Network failures propagate; the
    adapter reports only their type.
    """

    request = build_request(url, headers)
    try:
        with (opener or build_opener()).open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
    return status, b""


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("us")


def prices_url(ticker: str, start_date: str, end_date: str) -> str:
    """The one request per ticker; `end_date` is Tiingo's inclusive endDate."""

    return (
        f"{BASE_URL}/{ticker}/prices?startDate={start_date}&endDate={end_date}"
        f"&format=json&resampleFreq=daily&columns={','.join(ALLOWED_FIELDS)}"
    )


# --- checks shared by the response and the stored payload -----------------------


def _number(label: str, value: object) -> float:
    if type(value) not in (int, float):
        raise refuse(RefusalReason.INVALID_FIELD, f"{label} is a {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:  # an integer literal beyond the float range
        number = math.inf
    if not math.isfinite(number):
        raise refuse(RefusalReason.INVALID_VALUE, f"{label} is not finite")
    return number


def _session(label: str, value: object) -> pd.Timestamp:
    match = _SESSION_STAMP.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise refuse(RefusalReason.INVALID_DATE, f"{label} is not a midnight UTC timestamp")
    return parse_date(label, match.group(1))


def _utf8(label: str, data: object) -> str:
    """Strict UTF-8 text; the refusal is raised outside the except block (no chained bytes)."""

    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            pass
    raise refuse(RefusalReason.INVALID_JSON, f"{label} is not UTF-8 bytes")


def _element(label: str, element: object) -> dict:
    if not isinstance(element, dict):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{label} is a {type(element).__name__}")
    extra = sorted(set(element) - set(ALLOWED_FIELDS))
    if extra:
        raise TiingoDataRefused(
            TiingoRefusal.NON_ALLOWLISTED_FIELD, f"{label} has non-allowlisted fields {extra}"
        )
    missing = [name for name in ALLOWED_FIELDS if name not in element]
    if missing:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{label} lacks {missing}")
    return element


@dataclass(frozen=True)
class CheckedSession:
    """One element after every check, numbers as floats."""

    session: pd.Timestamp
    close: float
    div_cash: float
    split_factor: float


SessionGuard = Callable[[str, pd.Timestamp], None]
PreviousSession = Callable[[pd.Timestamp], pd.Timestamp]


def _refuse_lockbox(label: str, session: pd.Timestamp) -> None:
    if session >= MODEL_HOLDOUT_START:
        raise refuse(
            RefusalReason.LOCKBOX_VIOLATION,
            f"{label}.date {session.date()} is on or after the holdout start; batch refused",
        )


def _frozen_previous(session: pd.Timestamp) -> pd.Timestamp:
    return nyse.sessions(nyse.COVERAGE.first, session - _ONE_DAY)[-1]


def check_elements(
    ticker: str, data: object, first: pd.Timestamp, last: pd.Timestamp
) -> list[CheckedSession]:
    """
    The fail-closed checks for one ticker's elements, in [first, last] (both
    inclusive), including the lockbox, the split allowlist and the raw-close
    check (on the frozen research calendar).
    """

    return checked_sessions(
        ticker, data, first, last, guard=_refuse_lockbox, previous_session=_frozen_previous
    )


def checked_sessions(
    ticker: str,
    data: object,
    first: pd.Timestamp,
    last: pd.Timestamp,
    *,
    guard: SessionGuard,
    previous_session: PreviousSession,
) -> list[CheckedSession]:
    """
    `check_elements` with the date guard and the calendar as parameters: the
    research path refuses the lockbox, the prospective path refuses dates
    before its window. `guard(label, session)` runs right after each date is
    parsed, before the request-range check.
    """

    if not isinstance(data, list):
        raise refuse(
            RefusalReason.SCHEMA_MISMATCH,
            f"{ticker} prices response is a {type(data).__name__}, expected a list",
        )
    checked: dict[pd.Timestamp, CheckedSession] = {}
    for index, item in enumerate(data):
        label = f"{ticker} prices[{index}]"
        element = _element(label, item)
        session = _session(f"{label}.date", element["date"])
        guard(label, session)
        if not first <= session <= last:
            raise refuse(RefusalReason.OUTSIDE_REQUEST, f"{label}.date {session.date()}")
        if session in checked:
            raise refuse(RefusalReason.DUPLICATE_KEY, f"{label}.date {session.date()} repeats")
        close, div_cash, split_factor = (
            _number(f"{label}.{name}", element[name]) for name in ALLOWED_FIELDS[1:]
        )
        if close <= 0:
            raise refuse(RefusalReason.INVALID_VALUE, f"{label}.close must be > 0")
        if div_cash < 0 or split_factor <= 0:
            raise refuse(RefusalReason.INVALID_VALUE, f"{label} divCash < 0 or splitFactor <= 0")
        if split_factor != 1 and (ticker, f"{session:%Y-%m-%d}") not in VERIFIED_SPLITS:
            raise TiingoDataRefused(
                TiingoRefusal.UNVERIFIED_SPLIT_FACTOR,
                f"{label} reports a split factor on {session.date()}, which is not a "
                "verified share-count split",
            )
        checked[session] = CheckedSession(session, close, div_cash, split_factor)
    _check_verified_splits(ticker, checked, first, last, previous_session)
    return [checked[session] for session in sorted(checked)]


def _check_verified_splits(
    ticker: str,
    checked: dict[pd.Timestamp, CheckedSession],
    first: pd.Timestamp,
    last: pd.Timestamp,
    previous_session: PreviousSession,
) -> None:
    for (symbol, date), factor in sorted(VERIFIED_SPLITS.items()):
        split = parse_date("verified split", date)
        if symbol != ticker or not first <= split <= last:
            continue
        if not discriminable(factor):
            raise TiingoDataRefused(
                TiingoRefusal.SPLIT_FACTOR_NOT_DISCRIMINABLE,
                f"{ticker} verified split {date} x{factor:g}: |log factor| < 3 x log 1.25, so "
                "a raw-close check cannot tell it from no split",
            )
        row = checked.get(split)
        if row is None or row.split_factor != factor:
            raise TiingoDataRefused(
                TiingoRefusal.SPLIT_FACTOR_MISMATCH,
                f"{ticker} verified split {date} x{factor:g}: the vendor factor is "
                f"{'missing' if row is None else 'different'}",
            )
        previous = previous_session(split)
        before = checked.get(previous)
        if before is None or not split_close_consistent(row.close * factor / before.close):
            raise TiingoDataRefused(
                TiingoRefusal.SPLIT_PRICE_MISMATCH,
                f"{ticker} verified split {date}: the raw close is not consistent with an "
                f"unadjusted price across the split (previous session {previous.date()}, "
                f"tolerance |log ratio| < {BAND:.3f})",
            )


# --- the stored payload ---------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ParsedPayload:
    """The parsed envelope. Tables carry every schema column except `batch_id`."""

    start: pd.Timestamp
    end_date: pd.Timestamp  # inclusive, as requested from Tiingo
    fund_ids: tuple[str, ...]
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame


def _dates(values: list) -> pd.Series:
    return pd.to_datetime(pd.Series(values, dtype=object), utc=True).astype("datetime64[us, UTC]")


def _texts(value: str, count: int) -> pd.Series:
    return pd.Series([value] * count, dtype=object)


def frames(
    checked: dict[str, list[CheckedSession]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """F1 rows: unadjusted closes, divCash distributions and verified splits."""

    rows = [(ticker, row) for ticker, items in checked.items() for row in items]
    paid = [(ticker, row) for ticker, row in rows if row.div_cash > 0]
    split = [(ticker, row) for ticker, row in rows if row.split_factor != 1]
    prices = pd.DataFrame(
        {
            "fund_id": pd.Series([ticker for ticker, _ in rows], dtype=object),
            "session_date": _dates([row.session for _, row in rows]),
            "close": pd.Series([row.close for _, row in rows], dtype="float64"),
            "adjustment_basis": _texts(AdjustmentBasis.UNADJUSTED.value, len(rows)),
            "volume": pd.Series([pd.NA] * len(rows), dtype="Int64"),  # not requested
            "source_id": _texts(SOURCE_ID, len(rows)),
        }
    )
    no_dates = pd.Series([pd.NaT] * len(paid), dtype="datetime64[us, UTC]")
    distributions = pd.DataFrame(
        {
            "fund_id": pd.Series([ticker for ticker, _ in paid], dtype=object),
            "ex_date": _dates([row.session for _, row in paid]),
            "declaration_date": no_dates,
            "record_date": no_dates,
            "pay_date": no_dates,
            "amount": pd.Series([row.div_cash for _, row in paid], dtype="float64"),
            "amount_basis": _texts(AmountBasis.PER_SHARE_AS_PAID.value, len(paid)),
            "distribution_type": _texts(DistributionType.UNKNOWN.value, len(paid)),
            "source_sequence": pd.Series([0] * len(paid), dtype="int64"),
            "source_id": _texts(SOURCE_ID, len(paid)),
        }
    )
    splits = pd.DataFrame(
        {
            "fund_id": pd.Series([ticker for ticker, _ in split], dtype=object),
            "effective_date": _dates([row.session for _, row in split]),
            "ratio_new_per_old": pd.Series([row.split_factor for _, row in split], dtype="float64"),
            "source_id": _texts(SOURCE_ID, len(split)),
        }
    )
    return tuple(
        frame[[name for name in spec.names if name != "batch_id"]]
        for frame, spec in ((prices, PRICES), (distributions, DISTRIBUTIONS), (splits, SPLITS))
    )


def parse_payload(payload: bytes) -> ParsedPayload:
    """
    The transformation step: the stored envelope bytes to F1 rows, without a
    request. It re-runs every element check, so a payload with any
    non-allowlisted field or unverified split is refused here too.
    """

    envelope = loads_strict(_utf8("the payload", payload))
    if not isinstance(envelope, dict) or set(envelope) != set(_ENVELOPE_FIELDS):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "the payload is not a Tiingo envelope")
    if (
        envelope["envelope_version"] != ENVELOPE_VERSION
        or envelope["source_id"] != SOURCE_ID
        or envelope["fields"] != list(ALLOWED_FIELDS)
    ):
        raise refuse(RefusalReason.INVALID_FIELD, "the payload is not a Tiingo envelope v4")
    first = parse_date("payload start_date", envelope["start_date"])
    last = parse_date("payload end_date", envelope["end_date"])
    if last >= MODEL_HOLDOUT_START:
        raise refuse(RefusalReason.REQUEST_AFTER_CAP, "the payload end_date passes the cap")
    if first > last:
        raise refuse(RefusalReason.INVALID_REQUEST, "the payload start_date is after its end_date")
    prices = envelope["prices"]
    if not isinstance(prices, dict) or not prices:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "payload prices must be a non-empty object")
    checked = {}
    for ticker, elements in prices.items():
        if not _TICKER.fullmatch(ticker):
            raise refuse(RefusalReason.INVALID_FIELD, "a payload key is not a Tiingo ticker")
        checked[ticker] = check_elements(ticker, elements, first, last)
    return ParsedPayload(first, last, tuple(checked), *frames(checked))


# --- distribution-basis verification ---------------------------------------------


@dataclass(frozen=True, eq=False)
class Retrieval:
    """
    One retrieval, in memory only: checked for the field allowlist and the
    verified splits, but NOT for distribution basis. `verified_batch` is the
    only way to an F1 `RawBatch`. The tables never appear in its repr.
    """

    fund_ids: tuple[str, ...]
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    retrieved_at: pd.Timestamp
    raw_payload: bytes = field(repr=False)
    prices: pd.DataFrame = field(repr=False)
    distributions: pd.DataFrame = field(repr=False)
    splits: pd.DataFrame = field(repr=False)


class SplitCheck(StrEnum):
    """What the split-ratio consistency check (route (a)) found for one fund."""

    NO_SPLIT = "no_split"  # no verified split in the window
    NOT_COMPUTABLE = "not_computable"  # fewer than MIN_SIDE distributions on a side
    AGREED = "agreed"  # every computable ratio within BAND of 1
    FAILED = "failed"  # a ratio outside BAND: the fund fails, whatever route (b) says


@dataclass(frozen=True)
class BasisCheck:
    """
    One fund's distribution-basis verification. `route` is
    "primary_source_spot_check" when the fund passed, else None; `split_check`
    is a `SplitCheck` value; `detail` holds counts and ratios only.
    """

    fund_id: str
    passed: bool
    route: str | None
    split_check: str
    detail: str


def check_fund(
    fund: str,
    first: pd.Timestamp,
    last: pd.Timestamp,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
    anchors: Mapping[tuple[str, str], float],
) -> BasisCheck:
    """One fund's rules (i) to (iv) over [first, last], against `anchors` entries."""

    own = distributions[distributions["fund_id"] == fund].sort_values("ex_date")
    served = dict(zip(own["ex_date"], own["amount"], strict=True))
    listed = {}
    for (symbol, text), amount in anchors.items():
        date = parse_date("verified distribution", text)
        if symbol == fund and first <= date <= last:
            listed[date] = amount
    matches = sorted(
        date
        for date, amount in listed.items()
        if date in served and abs(served[date] - amount) <= AMOUNT_TOLERANCE
    )
    mismatched = len(listed) - len(matches)
    anchored = len(listed) >= MIN_VERIFIED_AMOUNTS and mismatched == 0  # (i)
    counts = f"{len(listed)} (needs {MIN_VERIFIED_AMOUNTS}), {len(matches)} match"
    notes = [f"verified distributions in the window: {counts}, {mismatched} do not"]
    outcomes = []
    for split in splits[splits["fund_id"] == fund].itertuples():
        date, factor = split.effective_date, split.ratio_new_per_old
        label = f"split {date:%Y-%m-%d} x{factor:g}"
        listed_before = sum(1 for match in matches if match < date)
        listed_after = sum(1 for match in matches if match > date)
        on_split = int((own["ex_date"] == date).sum())
        anchored = anchored and listed_before >= 1 and listed_after >= 1  # (ii)
        anchored = anchored and on_split == 0  # (iv)
        notes.append(
            f"{label}: listed matches {listed_before} before, {listed_after} after (needs 1 "
            f"each); distributions on the split session {on_split}"
        )
        before = own.loc[own["ex_date"] < date, "amount"].iloc[-MIN_SIDE:]
        after = own.loc[own["ex_date"] > date, "amount"].iloc[:MIN_SIDE]
        if len(before) < MIN_SIDE or len(after) < MIN_SIDE:
            outcomes.append(SplitCheck.NOT_COMPUTABLE)
            notes.append(
                f"{label} ratio check: {len(before)} before, {len(after)} after (needs "
                f"{MIN_SIDE} each), not computable"
            )
            continue
        ratio = float(before.median() / (factor * after.median()))
        inside = abs(math.log(ratio)) < BAND  # (iii)
        outcomes.append(SplitCheck.AGREED if inside else SplitCheck.FAILED)
        notes.append(
            f"{label} ratio check: median before / (factor x median after) = {ratio:.2f}, "
            f"{'within' if inside else 'outside'} |log| < {BAND:.3f}"
        )
    split_check = (
        SplitCheck.NO_SPLIT
        if not outcomes
        else SplitCheck.FAILED
        if SplitCheck.FAILED in outcomes
        else SplitCheck.NOT_COMPUTABLE
        if SplitCheck.NOT_COMPUTABLE in outcomes
        else SplitCheck.AGREED
    )
    passed = anchored and split_check is not SplitCheck.FAILED
    route = "primary_source_spot_check" if passed else None
    return BasisCheck(fund, passed, route, split_check.value, "; ".join(notes))


def _rebuilt(retrieval: Retrieval) -> ParsedPayload:
    """
    The tables rebuilt from the raw payload, which is what F1 stores and
    hashes. Refuses when the payload's tickers or dates differ from the
    retrieval's request, or when the in-memory tables differ from it.
    """

    parsed = parse_payload(retrieval.raw_payload)
    if (
        sorted(parsed.fund_ids) != sorted(retrieval.fund_ids)
        or parsed.start != retrieval.start
        or parsed.end_date != retrieval.end_exclusive - _ONE_DAY
    ):
        raise TiingoDataRefused(
            TiingoRefusal.PAYLOAD_MISMATCH,
            "the raw payload's tickers or dates differ from the retrieval's request",
        )
    for name in ("prices", "distributions", "splits"):
        if not getattr(parsed, name).equals(getattr(retrieval, name)):
            raise TiingoDataRefused(
                TiingoRefusal.PAYLOAD_MISMATCH,
                f"the in-memory {name} table differs from the raw payload",
            )
    return parsed


def require_scope(anchors: DistributionAnchors, scope: AnchorScope) -> DistributionAnchors:
    """The anchors, if loaded for `scope`; research and prospective anchors never mix."""

    if not isinstance(anchors, DistributionAnchors) or anchors.scope is not scope:
        found = getattr(anchors, "scope", type(anchors).__name__)
        raise ValueError(f"expected {scope.value}-scope DistributionAnchors, got {found}")
    return anchors


def basis_checks(
    fund_ids: Sequence[str], parsed: ParsedPayload, anchors: DistributionAnchors
) -> tuple[BasisCheck, ...]:
    return tuple(
        check_fund(
            fund,
            parsed.start,
            parsed.end_date,
            parsed.distributions,
            parsed.splits,
            anchors.entries,
        )
        for fund in fund_ids
    )


def refuse_failing(checks: tuple[BasisCheck, ...]) -> None:
    """`unverified_distribution_basis` unless every fund passed (counts and ratios only)."""

    failing = [check for check in checks if not check.passed]
    if failing:
        raise TiingoDataRefused(
            TiingoRefusal.UNVERIFIED_DISTRIBUTION_BASIS,
            "; ".join(
                f"{check.fund_id}: split check {check.split_check}; {check.detail}"
                for check in failing
            )
            + "; nothing may be stored under an unverified basis",
        )


def verify_distribution_basis(
    retrieval: Retrieval, *, anchors: DistributionAnchors
) -> tuple[BasisCheck, ...]:
    """
    Per fund, on the tables rebuilt from the raw payload: (i) to (iv) in the
    module docstring, against research-scope anchors. Only primary-source
    amounts can pass a fund; the split ratio can only fail one.
    """

    anchors = require_scope(anchors, AnchorScope.RESEARCH)
    return basis_checks(retrieval.fund_ids, _rebuilt(retrieval), anchors)


def verified_batch(retrieval: Retrieval, *, anchors: DistributionAnchors) -> RawBatch:
    """
    The F1 batch, built from the tables rebuilt from the raw payload, only
    when every fund's distribution basis is verified against research-scope
    anchors. The vintage note records the anchors file's sha256.
    """

    anchors = require_scope(anchors, AnchorScope.RESEARCH)
    parsed = _rebuilt(retrieval)
    refuse_failing(basis_checks(retrieval.fund_ids, parsed, anchors))
    funds = ", ".join(retrieval.fund_ids)
    return RawBatch(
        adapter_id=TiingoAdapter.adapter_id,
        adapter_version=TiingoAdapter.adapter_version,
        fund_ids=retrieval.fund_ids,
        start=retrieval.start,
        end_exclusive=retrieval.end_exclusive,
        instruments=tuple(Instrument(ticker, ticker, None) for ticker in retrieval.fund_ids),
        retrieved_at=retrieval.retrieved_at,
        raw_payload=retrieval.raw_payload,
        terms_note=TERMS_NOTE,
        vintage_note=(
            f"{VINTAGE_NOTE}; {anchors.provenance_note()}; distribution basis spot-checked "
            f"against primary-source amounts (a sample, not a validation of every "
            f"distribution): {funds}"
        ),
        prices=parsed.prices,
        distributions=parsed.distributions,
        splits=parsed.splits,
    )


# --- the adapter ----------------------------------------------------------------


class TiingoAdapter:
    """
    `FundDataAdapter` for Tiingo daily prices. Fund ids are the Tiingo tickers
    (upper case). Every collaborator is injected: the token provider, the
    transport `(url, headers) -> (status, body)`, the clock for
    `retrieved_at` and the research-scope distribution anchors `fetch`
    verifies against (default: none, so `fetch` stores nothing).
    """

    adapter_id = "tiingo_daily"
    adapter_version = "5"
    terms_note = TERMS_NOTE
    vintage_note = VINTAGE_NOTE

    def __init__(
        self,
        *,
        token_provider: TokenProvider = env_token,
        transport: Transport = urllib_transport,
        clock: Clock = utc_now,
        anchors: DistributionAnchors = NO_ANCHORS,
    ) -> None:
        self._token_provider = token_provider
        self._transport = transport
        self._clock = clock
        self._anchors = require_scope(anchors, AnchorScope.RESEARCH)

    def now(self) -> pd.Timestamp:
        """The injected clock, as a UTC timestamp."""

        return utc_timestamp("clock", self._clock())

    def _get(self, token: _Token, ticker: str, url: str) -> str:
        # An expected network failure is reported by type only and raised outside
        # the except block, so no exception text or chained traceback carries the
        # header. Any other exception type propagates unchanged. No retry.
        failure: str | None = None
        try:
            response = self._transport(url, token.headers())
        except TRANSPORT_FAILURES as error:
            failure = type(error).__name__
        if failure is not None:
            raise TiingoRequestFailed(TiingoFailure.TRANSPORT_ERROR, ticker=ticker, detail=failure)
        if (
            not isinstance(response, tuple)
            or len(response) != 2
            or type(response[0]) is not int
            or not isinstance(response[1], bytes)
        ):
            raise TiingoRequestFailed(
                TiingoFailure.TRANSPORT_ERROR,
                ticker=ticker,
                detail="the transport returned no (status, bytes) pair",
            )
        status, body = response
        if status != 200:
            reason = (
                TiingoFailure.RATE_LIMITED
                if status == 429
                else TiingoFailure.SERVER_ERROR
                if 500 <= status <= 599
                else TiingoFailure.HTTP_ERROR
            )
            raise TiingoRequestFailed(reason, ticker=ticker, status=status)
        return _utf8(f"{ticker} prices response", body)

    def request_elements(
        self,
        ids: Sequence[str],
        first: pd.Timestamp,
        last: pd.Timestamp,
        check: Callable[[str, object], object],
    ) -> dict[str, list]:
        """
        One prices call per ticker for [first, last] (Tiingo's endDate is
        inclusive). The key is read once; each response is parsed strictly and
        passed to `check` before the next request, so a refusal stops the
        batch at once. Callers validate the request before calling this.
        """

        start_date, end_date = f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}"
        token = _Token(self._token_provider())
        elements = {}
        for ticker in ids:  # each response is checked before the next request
            data = loads_strict(self._get(token, ticker, prices_url(ticker, start_date, end_date)))
            check(ticker, data)
            elements[ticker] = data
        return elements

    def retrieve(
        self, fund_ids: Sequence[str], start: pd.Timestamp, end_exclusive: pd.Timestamp
    ) -> Retrieval:
        """Request and check every fund in memory; distribution basis NOT yet verified."""

        ids, first, end = validate_request(fund_ids, start, end_exclusive)
        if len(ids) > MAX_TICKERS:
            raise refuse(
                RefusalReason.INVALID_REQUEST, f"{len(ids)} funds; at most {MAX_TICKERS} a batch"
            )
        invalid = [fund_id for fund_id in ids if not _TICKER.fullmatch(fund_id)]
        if invalid:
            raise refuse(
                RefusalReason.INVALID_REQUEST, f"fund ids must be upper-case tickers: {invalid}"
            )
        last = end - _ONE_DAY
        start_date, end_date = f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}"
        elements = self.request_elements(
            ids, first, last, lambda ticker, data: check_elements(ticker, data, first, last)
        )
        retrieved_at = utc_timestamp("retrieved_at", self._clock())
        envelope = {
            "envelope_version": ENVELOPE_VERSION,
            "source_id": SOURCE_ID,
            "start_date": start_date,
            "end_date": end_date,
            "fields": list(ALLOWED_FIELDS),
            "prices": elements,
        }
        payload = dumps_canonical(envelope).encode("utf-8")
        parsed = parse_payload(payload)
        return Retrieval(
            ids,
            first,
            end,
            retrieved_at,
            payload,
            parsed.prices,
            parsed.distributions,
            parsed.splits,
        )

    def fetch(
        self, fund_ids: Sequence[str], start: pd.Timestamp, end_exclusive: pd.Timestamp
    ) -> RawBatch:
        """The F1 seam: a batch only when every fund's distribution basis is verified."""

        return verified_batch(self.retrieve(fund_ids, start, end_exclusive), anchors=self._anchors)


__all__ = [
    "ALLOWED_FIELDS",
    "BAND",
    "NO_ANCHORS",
    "VERIFIED_SPLITS",
    "BasisCheck",
    "ParsedPayload",
    "Retrieval",
    "SplitCheck",
    "TiingoAdapter",
    "TiingoDataRefused",
    "TiingoFailure",
    "TiingoRefusal",
    "TiingoRequestFailed",
    "basis_checks",
    "check_elements",
    "check_fund",
    "checked_sessions",
    "discriminable",
    "env_token",
    "parse_payload",
    "prices_url",
    "refuse_failing",
    "require_scope",
    "urllib_transport",
    "verified_batch",
    "verify_distribution_basis",
]
