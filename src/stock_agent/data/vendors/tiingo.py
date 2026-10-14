"""
The Tiingo end-of-day adapter for the F1 fund-data store.

Built to the owner's requirements of 2026-10-14. Tiingo's endpoint shape and
field meanings are recalled from its documentation [unverified]; the
owner-authorized coverage audit (`scripts/audit_fund_coverage.py`) checks them.

Request (prices only; the metadata endpoint is never called)
    The F1 request check runs first, and the adapter refuses before any
    request unless `end_exclusive` is at most `MODEL_HOLDOUT_START`, so the
    requested `endDate` is at most the day before it. One call per fund:

        GET https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=<start>
            &endDate=<end_exclusive - 1 day>&format=json&resampleFreq=daily
            &columns=date,close,volume,divCash,splitFactor

    Tiingo's `endDate` is inclusive. `retrieved_at` is read from the injected
    clock after the last response arrives.

Field allowlist (the lockbox information-content rule)
    `docs/product/architecture.md`, section 4, rule 7: "dated" means
    information content. Adjusted levels and vendor metadata carry information
    from after the holdout start, so they are never requested, parsed or
    stored. Each element must hold exactly `ALLOWED_FIELDS`. Any other field
    (for example `adjClose`) refuses the batch with `non_allowlisted_field`:
    that is how the audit learns whether Tiingo honours `columns`. A refused
    response is never stored or logged, and refusal details name fields,
    dates and indices only, never a value.

Mapping
    close, volume   -> `prices`, basis `unadjusted` only
    divCash > 0     -> `distributions`: amount = divCash, `per_share_as_paid`
                       [unverified until the audit], type `unknown`, sequence 0,
                       declaration, record and pay dates NULL (Tiingo has none)
    splitFactor     -> `splits` only for a share-count split in the reviewed
                       `VERIFIED_SPLITS` allowlist, on the same session, with the
                       same factor, and with a raw close consistent with an
                       unadjusted price (`SPLIT_PRICE_TOLERANCE`). Any other
                       factor != 1 refuses the batch.

Raw payload
    The canonical JSON of the allowlisted fields actually used, keyed by
    ticker. F1 hashes it. Nothing else from the response is kept.

Secrets and failures
    The key is sent only in the `Authorization: Token <key>` header, never in a
    URL. It is read once per fetch from the injected provider (by default
    `TIINGO_API_KEY`), held in a wrapper whose repr is redacted, and never
    stored on the adapter. Any status other than 200, including 429 and 5xx,
    refuses the batch with no retry. An expected network failure
    (`TRANSPORT_FAILURES`) is reported by type only, raised outside the except
    block (no chained exception, header or body); other exception types
    propagate unchanged. The default transport follows no redirects.

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
from dataclasses import dataclass
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
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

TOKEN_ENV = "TIINGO_API_KEY"
BASE_URL = "https://api.tiingo.com/tiingo/daily"
SOURCE_ID = "tiingo"
TIMEOUT_SECONDS = 30.0
ENVELOPE_VERSION = "3"

# The only fields requested, parsed and stored.
ALLOWED_FIELDS = ("date", "close", "volume", "divCash", "splitFactor")
NOT_REQUESTED = (
    "the metadata endpoint; price fields adjClose, adjHigh, adjLow, adjOpen, adjVolume, "
    "high, low, open"
)

# Externally verified share-count splits: (ticker, session) -> new shares per old.
# Code-reviewed; add an entry only with a primary source.
VERIFIED_SPLITS = {
    # SEC EDGAR 0000036405/000093247108001251: split-adjusted trading from 2008-06-18.
    ("VTI", "2008-06-18"): 2.0,
}
# At a verified split, (raw close x factor) / previous raw close must lie within
# 1 +/- this. An unadjusted close gives about 1 (the price falls by the factor);
# a close that is already split-adjusted gives about the factor (2.0 for VTI).
SPLIT_PRICE_TOLERANCE = 0.25

TERMS_NOTE = (
    "Tiingo free tier, Internal Use Only; no redistribution; "
    "terms https://www.tiingo.com/about/terms accessed 2026-10-13"
)
VINTAGE_NOTE = (
    "close, volume, divCash and splitFactor as served on the retrieval date; not a "
    "point-in-time vintage; divCash share basis unverified until the coverage audit; "
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


class TiingoRefusal(StrEnum):
    NON_ALLOWLISTED_FIELD = "non_allowlisted_field"
    UNVERIFIED_SPLIT_FACTOR = "unverified_split_factor"
    SPLIT_FACTOR_MISMATCH = "split_factor_mismatch"
    SPLIT_PRICE_MISMATCH = "split_price_mismatch"


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


def urllib_transport(
    url: str, headers: Mapping[str, str], *, timeout: float = TIMEOUT_SECONDS
) -> tuple[int, bytes]:
    """
    GET with the standard library: a timeout, no redirects, headers that are
    never forwarded. Returns (status, body); the body is empty for an error
    status. Network failures propagate; the adapter reports only their type.
    """

    request = urllib.request.Request(url, method="GET")
    for name, value in headers.items():
        request.add_unredirected_header(name, value)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
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
    missing = [field for field in ALLOWED_FIELDS if field not in element]
    if missing:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{label} lacks {missing}")
    return element


@dataclass(frozen=True)
class CheckedSession:
    """One element after every check; numbers as floats, volume as an int."""

    session: pd.Timestamp
    close: float
    volume: int
    div_cash: float
    split_factor: float


def check_elements(
    ticker: str, data: object, first: pd.Timestamp, last: pd.Timestamp
) -> list[CheckedSession]:
    """
    The fail-closed checks for one ticker's elements, in [first, last] (both
    inclusive), including the split allowlist and the raw-close check.
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
        if session >= MODEL_HOLDOUT_START:
            raise refuse(
                RefusalReason.LOCKBOX_VIOLATION,
                f"{label}.date {session.date()} is on or after the holdout start; batch refused",
            )
        if not first <= session <= last:
            raise refuse(RefusalReason.OUTSIDE_REQUEST, f"{label}.date {session.date()}")
        if session in checked:
            raise refuse(RefusalReason.DUPLICATE_KEY, f"{label}.date {session.date()} repeats")
        close, div_cash, split_factor = (
            _number(f"{label}.{field}", element[field])
            for field in ("close", "divCash", "splitFactor")
        )
        _number(f"{label}.volume", element["volume"])
        volume = element["volume"]
        if close <= 0:
            raise refuse(RefusalReason.INVALID_VALUE, f"{label}.close must be > 0")
        if type(volume) is not int or not 0 <= volume < 2**63:
            raise refuse(RefusalReason.INVALID_VALUE, f"{label}.volume must be a whole number")
        if div_cash < 0 or split_factor <= 0:
            raise refuse(RefusalReason.INVALID_VALUE, f"{label} divCash < 0 or splitFactor <= 0")
        if split_factor != 1:
            expected = VERIFIED_SPLITS.get((ticker, f"{session:%Y-%m-%d}"))
            if expected is None:
                raise TiingoDataRefused(
                    TiingoRefusal.UNVERIFIED_SPLIT_FACTOR,
                    f"{label} reports a split factor on {session.date()}, which is not a "
                    "verified share-count split",
                )
        checked[session] = CheckedSession(session, close, volume, div_cash, split_factor)
    _check_verified_splits(ticker, checked, first, last)
    return [checked[session] for session in sorted(checked)]


def _check_verified_splits(
    ticker: str,
    checked: dict[pd.Timestamp, CheckedSession],
    first: pd.Timestamp,
    last: pd.Timestamp,
) -> None:
    for (symbol, date), factor in sorted(VERIFIED_SPLITS.items()):
        split = parse_date("verified split", date)
        if symbol != ticker or not first <= split <= last:
            continue
        row = checked.get(split)
        if row is None or row.split_factor != factor:
            raise TiingoDataRefused(
                TiingoRefusal.SPLIT_FACTOR_MISMATCH,
                f"{ticker} verified split {date} x{factor:g}: the vendor factor is "
                f"{'missing' if row is None else 'different'}",
            )
        previous = nyse.sessions(nyse.COVERAGE.first, split - _ONE_DAY)[-1]
        before = checked.get(previous)
        consistent = (
            before is not None
            and abs(row.close * factor / before.close - 1) <= SPLIT_PRICE_TOLERANCE
        )
        if not consistent:
            raise TiingoDataRefused(
                TiingoRefusal.SPLIT_PRICE_MISMATCH,
                f"{ticker} verified split {date}: the raw close is not consistent with an "
                f"unadjusted price across the split (previous session {previous.date()}, "
                f"tolerance {SPLIT_PRICE_TOLERANCE:g})",
            )


# --- the stored payload ---------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ParsedPayload:
    """The parsed envelope. Tables carry every schema column except `batch_id`."""

    start: pd.Timestamp
    end_date: pd.Timestamp  # inclusive, as requested from Tiingo
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame


def _dates(values: list) -> pd.Series:
    return pd.to_datetime(pd.Series(values, dtype=object), utc=True).astype("datetime64[us, UTC]")


def _texts(value: str, count: int) -> pd.Series:
    return pd.Series([value] * count, dtype=object)


def _frames(
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
            "volume": pd.Series([row.volume for _, row in rows], dtype="Int64"),
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
    request. It re-runs every check, so a payload with any non-allowlisted
    field or unverified split is refused here too.
    """

    envelope = loads_strict(_utf8("the payload", payload))
    if not isinstance(envelope, dict) or set(envelope) != set(_ENVELOPE_FIELDS):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "the payload is not a Tiingo envelope")
    if (
        envelope["envelope_version"] != ENVELOPE_VERSION
        or envelope["source_id"] != SOURCE_ID
        or envelope["fields"] != list(ALLOWED_FIELDS)
    ):
        raise refuse(RefusalReason.INVALID_FIELD, "the payload is not a Tiingo envelope v3")
    first = parse_date("payload start_date", envelope["start_date"])
    last = parse_date("payload end_date", envelope["end_date"])
    if last >= MODEL_HOLDOUT_START:
        raise refuse(RefusalReason.REQUEST_AFTER_CAP, "the payload end_date passes the cap")
    prices = envelope["prices"]
    if not isinstance(prices, dict) or not prices:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "payload prices must be a non-empty object")
    checked = {}
    for ticker, elements in prices.items():
        if not _TICKER.fullmatch(ticker):
            raise refuse(RefusalReason.INVALID_FIELD, "a payload key is not a Tiingo ticker")
        checked[ticker] = check_elements(ticker, elements, first, last)
    return ParsedPayload(first, last, *_frames(checked))


# --- the adapter ----------------------------------------------------------------


class TiingoAdapter:
    """
    `FundDataAdapter` for Tiingo daily prices. Fund ids are the Tiingo tickers
    (upper case). Every collaborator is injected: the token provider, the
    transport `(url, headers) -> (status, body)` and the clock for
    `retrieved_at`.
    """

    adapter_id = "tiingo_daily"
    adapter_version = "3"
    terms_note = TERMS_NOTE
    vintage_note = VINTAGE_NOTE

    def __init__(
        self,
        *,
        token_provider: TokenProvider = env_token,
        transport: Transport = urllib_transport,
        clock: Clock = utc_now,
    ) -> None:
        self._token_provider = token_provider
        self._transport = transport
        self._clock = clock

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

    def fetch(
        self, fund_ids: Sequence[str], start: pd.Timestamp, end_exclusive: pd.Timestamp
    ) -> RawBatch:
        ids, first, end = validate_request(fund_ids, start, end_exclusive)
        last = end - _ONE_DAY
        if last >= MODEL_HOLDOUT_START:  # validate_request already caps; kept explicit
            raise refuse(RefusalReason.REQUEST_AFTER_CAP, f"endDate {last.date()} passes the cap")
        invalid = [fund_id for fund_id in ids if not _TICKER.fullmatch(fund_id)]
        if invalid:
            raise refuse(
                RefusalReason.INVALID_REQUEST, f"fund ids must be upper-case tickers: {invalid}"
            )
        start_date, end_date = f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}"
        token = _Token(self._token_provider())
        texts = {
            ticker: self._get(token, ticker, prices_url(ticker, start_date, end_date))
            for ticker in ids
        }
        retrieved_at = utc_timestamp("retrieved_at", self._clock())
        elements = {}
        for ticker in ids:
            data = loads_strict(texts[ticker])
            check_elements(ticker, data, first, last)  # refuses before anything is kept
            elements[ticker] = data
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
        return RawBatch(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            fund_ids=ids,
            start=first,
            end_exclusive=end,
            instruments=tuple(Instrument(ticker, ticker, None) for ticker in ids),
            retrieved_at=retrieved_at,
            raw_payload=payload,
            terms_note=self.terms_note,
            vintage_note=self.vintage_note,
            prices=parsed.prices,
            distributions=parsed.distributions,
            splits=parsed.splits,
        )


__all__ = [
    "ALLOWED_FIELDS",
    "VERIFIED_SPLITS",
    "ParsedPayload",
    "TiingoAdapter",
    "TiingoDataRefused",
    "TiingoFailure",
    "TiingoRefusal",
    "TiingoRequestFailed",
    "check_elements",
    "env_token",
    "parse_payload",
    "prices_url",
    "urllib_transport",
]
