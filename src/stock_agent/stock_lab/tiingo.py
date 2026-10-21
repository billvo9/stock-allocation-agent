"""
The Tiingo end-of-day stock adapter for the Stock Lab (replaceable).

It is separate from the ETF research adapter (`stock_agent.data.vendors.
tiingo`), whose allowlist, split verification and lockbox rules stay as they
are. It reuses only that module's audited key handling and transport: the key
comes from TIINGO_API_KEY, is sent only in the Authorization header to
api.tiingo.com, never appears in a message, URL or traceback, and requests
ignore environment proxies and follow no redirect.

Authorization. Nothing here makes a request unless the adapter is built with
`authorized=True`, which only an owner-run script passes (after the owner's
explicit authorization of the exact scope). Every request counts against a
per-run budget (`max_requests`), and requests are paced (`min_interval`).
The Lab never imports this module (tested).

Requests
    listing  GET https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip
             (no key sent: another host). One CSV with exactly the columns
             ticker, exchange, assetType, priceCurrency, startDate, endDate;
             any other header refuses the listing.
    prices   GET https://api.tiingo.com/tiingo/daily/<ticker>/prices
             ?startDate=..&endDate=..&format=json&resampleFreq=daily
             &columns=date,open,high,low,close,volume,divCash,splitFactor

Field semantics (as Tiingo documents them; recorded with every batch)
    open, high, low, close   raw, as traded on the session (not adjusted)
    volume                   raw shares traded (not adjusted for splits)
    divCash                  cash dividend per share, on its ex-date only
    splitFactor              new shares per old share on the split's ex-date,
                             1.0 on every other session
    adjClose, adjOpen, ...   NOT requested and refused if served: adjusted
                             history is recomputed after every later event, so
                             it is not a point-in-time vintage.

Every batch records its retrieval time and is stored as a new vintage
(store.py); a session served differently by a later retrieval is reported
as revised, never overwritten. The ticker-to-security mapping is Tiingo's
view at retrieval; it is not a point-in-time security master (tickers are
reused), and Tiingo does not say whether a listing is an ADR.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import time
import zipfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from stock_agent.data.vendors.tiingo import (
    TRANSPORT_FAILURES,
    TiingoFailure,
    TiingoRequestFailed,
    _Token,
    env_token,
    urllib_transport,
    utc_now,
)
from stock_agent.stock_lab.prices import BAR_COLUMNS, validate_bars
from stock_agent.stock_lab.symbols import ListingRow

SOURCE_ID = "tiingo_eod_stocks"
PRICES_BASE = "https://api.tiingo.com/tiingo/daily"
LISTING_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
PRICE_FIELDS = ("date", "open", "high", "low", "close", "volume", "divCash", "splitFactor")
LISTING_HEADER = ("ticker", "exchange", "assetType", "priceCurrency", "startDate", "endDate")
DEFAULT_MAX_REQUESTS = 40
DEFAULT_MIN_INTERVAL = 2.0  # seconds between requests
MAX_LISTING_BYTES = 64 * 1024 * 1024

VINTAGE_NOTE = (
    "Tiingo EOD as served on the retrieval date: raw open, high, low, close and volume, divCash "
    "on ex-dates and splitFactor on split ex-dates; adjusted fields not requested; not a "
    "point-in-time vintage; the ticker-to-security mapping is Tiingo's view at retrieval"
)
TERMS_NOTE = (
    "Tiingo Starter (free) plan for private, local use; no redistribution; terms "
    "https://www.tiingo.com/about/terms"
)

_TICKER = re.compile(r"[A-Z]{1,5}(-[A-Z]{1,2})?")
_SESSION_STAMP = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})T00:00:00(\.000)?Z")

TokenProvider = Callable[[], str]
Transport = Callable[[str, Mapping[str, str]], tuple[int, bytes]]


class StockDataRefused(ValueError):
    """A response that breaks the documented format; nothing from it is stored."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class NotAuthorized(RuntimeError):
    """A request from an adapter that was not built with authorized=True."""


@dataclass(frozen=True)
class PriceResponse:
    ticker: str
    bars: pd.DataFrame  # prices.BAR_COLUMNS
    retrieved_at: pd.Timestamp
    start: date
    end: date


@dataclass(frozen=True)
class ListingResponse:
    rows: tuple[ListingRow, ...]
    retrieved_at: pd.Timestamp


def prices_url(ticker: str, start: date, end: date) -> str:
    if not _TICKER.fullmatch(ticker):
        raise ValueError(f"not a US ticker in the provider's form: {ticker!r}")
    if start > end:
        raise ValueError("start is after end")
    return (
        f"{PRICES_BASE}/{ticker}/prices?startDate={start.isoformat()}&endDate={end.isoformat()}"
        f"&format=json&resampleFreq=daily&columns={','.join(PRICE_FIELDS)}"
    )


REQUIRED_NUMBERS = ("divCash", "splitFactor")  # never null: null would be guessed as 0 or 1


def _number(value: object, field: str, ticker: str) -> float:
    if value is None:
        if field in REQUIRED_NUMBERS:
            raise StockDataRefused("null_field", f"{ticker} {field}")
        return math.nan
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StockDataRefused("non_numeric_field", f"{ticker} {field}")
    try:
        number = float(value)
    except OverflowError:
        raise StockDataRefused("non_finite_field", f"{ticker} {field}") from None
    if not math.isfinite(number):
        raise StockDataRefused("non_finite_field", f"{ticker} {field}")
    return number


def parse_prices(ticker: str, body: bytes) -> pd.DataFrame:
    """The bars in a prices response (module docstring); refuses anything else."""

    try:
        rows = json.loads(
            body.decode("utf-8"), parse_constant=_refuse_constant, object_pairs_hook=_no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise StockDataRefused("not_json", f"{ticker}: {type(error).__name__}") from None
    if not isinstance(rows, list):
        raise StockDataRefused("not_a_list", ticker)
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise StockDataRefused("not_an_object", ticker)
        if set(row) != set(PRICE_FIELDS):
            extra = sorted(set(row) - set(PRICE_FIELDS))
            missing = sorted(set(PRICE_FIELDS) - set(row))
            raise StockDataRefused("unexpected_fields", f"{ticker} extra {extra} missing {missing}")
        stamp = row["date"]
        match = _SESSION_STAMP.fullmatch(stamp) if isinstance(stamp, str) else None
        if match is None:
            raise StockDataRefused("bad_date", ticker)
        records.append(
            {
                "session": match.group(1),
                "open": _number(row["open"], "open", ticker),
                "high": _number(row["high"], "high", ticker),
                "low": _number(row["low"], "low", ticker),
                "close": _number(row["close"], "close", ticker),
                "volume": _number(row["volume"], "volume", ticker),
                "div_cash": _number(row["divCash"], "divCash", ticker),
                "split_factor": _number(row["splitFactor"], "splitFactor", ticker),
            }
        )
    frame = pd.DataFrame(records, columns=list(BAR_COLUMNS))
    try:
        frame["session"] = pd.to_datetime(frame["session"], utc=True, format="%Y-%m-%d")
        frame["session"] = frame["session"].dt.as_unit("us")
        return validate_bars(frame)
    except (ValueError, OverflowError) as error:  # an impossible date, or bad bars
        raise StockDataRefused("invalid_bars", f"{ticker}: {type(error).__name__}") from None


def _refuse_constant(token: str) -> None:
    raise StockDataRefused("non_finite_number", token)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise StockDataRefused("duplicate_key", "a JSON object repeats a key")
    return dict(pairs)


def _listing_date(text: str) -> date | None:
    text = text.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise StockDataRefused("bad_listing_date", text[:20]) from None


def parse_listing(body: bytes) -> tuple[ListingRow, ...]:
    """The rows of the supported-tickers zip (module docstring); refuses anything else."""

    if len(body) > MAX_LISTING_BYTES:
        raise StockDataRefused("listing_too_large", f"{len(body)} bytes")
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise StockDataRefused("listing_files", f"expected one CSV, found {len(names)}")
        info = archive.getinfo(names[0])
        if info.file_size > MAX_LISTING_BYTES:
            raise StockDataRefused("listing_too_large", f"{info.file_size} bytes uncompressed")
        text = archive.read(names[0]).decode("utf-8-sig")
    except StockDataRefused:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError,
            EOFError, OSError, UnicodeDecodeError, zlib.error) as error:  # fmt: skip
        raise StockDataRefused("not_a_listing", type(error).__name__) from None
    try:
        return _listing_rows(csv.reader(io.StringIO(text)))
    except csv.Error as error:
        raise StockDataRefused("listing_row", type(error).__name__) from None


def _listing_rows(reader) -> tuple[ListingRow, ...]:
    header = tuple(next(reader, ()))
    if header != LISTING_HEADER:
        raise StockDataRefused("listing_header", f"unexpected columns {list(header)}")
    rows = []
    for record in reader:
        if not record:
            continue
        if len(record) != len(LISTING_HEADER):
            raise StockDataRefused("listing_row", f"{len(record)} fields")
        ticker, exchange, asset_type, currency, start, end = record
        rows.append(
            ListingRow(
                ticker=ticker.strip().upper(),
                exchange=exchange.strip(),
                asset_type=asset_type.strip(),
                price_currency=currency.strip(),
                start_date=_listing_date(start),
                end_date=_listing_date(end),
            )
        )
    if not rows:
        raise StockDataRefused("listing_empty", "no rows")
    return tuple(rows)


class StockAdapter:
    """
    Tiingo requests for the Stock Lab. `authorized` must be True for any
    request; transport, key, clock and sleep are injectable (tests use fakes).
    """

    def __init__(
        self,
        *,
        authorized: bool = False,
        token_provider: TokenProvider = env_token,
        transport: Transport = urllib_transport,
        clock: Callable[[], pd.Timestamp] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> None:
        if authorized is not True:
            authorized = False
        self._authorized = authorized
        self._token_provider = token_provider
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._monotonic = monotonic
        self._max_requests = max_requests
        self._min_interval = min_interval
        self._last_request: float | None = None
        self.requests_made = 0

    def _require_authorization(self) -> None:
        if not self._authorized:
            raise NotAuthorized(
                "live Tiingo requests need the owner's explicit authorization (authorized=True)"
            )

    def _get(
        self, url: str, headers: Mapping[str, str], ticker: str | None, *, keyed: bool = True
    ) -> bytes:
        self._require_authorization()
        if self.requests_made >= self._max_requests:
            raise TiingoRequestFailed(
                TiingoFailure.RATE_LIMITED, ticker=ticker, detail="this run's request budget"
            )
        if self._last_request is not None:
            wait = self._min_interval - (self._monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self.requests_made += 1
        self._last_request = self._monotonic()
        failure = None
        try:
            status, body = self._transport(url, headers)
        except TRANSPORT_FAILURES as error:
            failure = type(error).__name__
        if failure is not None:  # raised outside the except block: no chained cause
            raise TiingoRequestFailed(TiingoFailure.TRANSPORT_ERROR, ticker=ticker, detail=failure)
        if status == 200:
            return body
        if status in (401, 403) and keyed:
            reason = TiingoFailure.INVALID_TOKEN
        elif status == 429:
            reason = TiingoFailure.RATE_LIMITED
        elif status >= 500:
            reason = TiingoFailure.SERVER_ERROR
        else:
            reason = TiingoFailure.HTTP_ERROR
        raise TiingoRequestFailed(reason, ticker=ticker, status=status)

    def check_key(self) -> None:
        """Refuse a missing or malformed key now, without a request (the key is not kept)."""

        self._require_authorization()  # before the key is even read
        _Token(self._token_provider())

    def listing(self) -> ListingResponse:
        """The supported-tickers listing (no key is sent to its host)."""

        body = self._get(LISTING_URL, {}, None, keyed=False)
        return ListingResponse(parse_listing(body), self._clock())

    def prices(self, ticker: str, start: date, end: date) -> PriceResponse:
        """One ticker's raw daily bars from `start` through `end` (inclusive)."""

        url = prices_url(ticker, start, end)
        self._require_authorization()  # before the key is even read
        token = _Token(self._token_provider())
        body = self._get(url, token.headers(), ticker)
        bars = parse_prices(ticker, body)
        outside = bars["session"].dt.date.lt(start) | bars["session"].dt.date.gt(end)
        if outside.any():
            raise StockDataRefused("outside_request", f"{ticker}: a session outside the request")
        return PriceResponse(ticker, bars, self._clock(), start, end)
