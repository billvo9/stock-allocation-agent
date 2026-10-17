"""
Tickers: parsing what a user types or pastes, offline normalization, and the
classification of a ticker against a provider's listing.

Scope: US-listed stocks only. The lab never substitutes another listing:
it never maps a foreign ticker to its US listing or ADR, a share class to
another class, or an unknown ticker to a similar one. A ticker is studied
exactly as entered (after the formatting rules below) or not at all.

Formatting rules (offline, no provider needed)
    - surrounding spaces are removed and letters upper-cased;
    - a share-class separator written as "." or "/" becomes "-", the form
      Tiingo uses (BRK.B, BRK/B and BRK-B are one ticker, BRK-B). The entered
      text is kept and shown, so the rewrite is never hidden;
    - a known non-US exchange suffix (VOD.L, SHOP.TO, 7203.T) is unsupported:
      it names a listing outside the US;
    - index, currency and futures notation (^GSPC, EURUSD=X, ES=F) is
      unsupported: not a stock;
    - anything else must be 1-5 letters, optionally followed by "-" and a
      1-2 letter class or series (BRK-B, BF-A).

Classification against a listing (`classify`)
    A listing row says, for one ticker, its exchange, asset type, price
    currency and the first and last session the provider has prices for.
    Rows are the provider's view at retrieval, not a point-in-time security
    master: tickers are reused, and the provider does not say whether a
    listing is an ADR. Rules, in order:

    1. no row for the ticker: unsupported (not listed by the provider);
    2. rows whose asset type is not a stock, whose exchange is not a US
       exchange (or is over the counter), or whose currency is not USD are
       not eligible; if no eligible row remains, the first reason found is
       reported;
    3. several eligible rows whose coverage reaches the recent cutoff:
       ambiguous (never guessed);
    4. one eligible row reaching the cutoff: supported;
    5. otherwise the most recent eligible row decides: its last price within
       `delisted_after` sessions of the cutoff is inactive (no recent
       prices), older is "delisted or renamed" (the provider's coverage
       ended; it cannot say which).

    A row with no last date is treated as having no price history.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from stock_agent.stock_lab.contracts import SymbolReason, SymbolStatus

MAX_TICKERS_PER_PASTE = 200

# Suffixes that name a listing outside the US (exchange codes used by common
# data sites). A suffix is checked only after a "." or ":" separator.
NON_US_SUFFIXES = frozenset(
    {
        "L", "TO", "V", "CN", "NE", "HK", "T", "DE", "F", "BE", "DU", "HM", "MU", "SG",
        "PA", "AS", "BR", "LS", "MI", "MC", "SW", "VI", "ST", "OL", "CO", "HE", "IR",
        "WA", "PR", "AT", "IS", "ME", "AX", "NZ", "NS", "BO", "SS", "SZ", "KS", "KQ",
        "TW", "TWO", "SI", "BK", "JK", "KL", "SA", "MX", "BA", "SN", "JO", "TA", "LN",
        "GR", "FP", "NA", "SM", "IM", "CH", "JP", "AU",
    }
)  # fmt: skip

# Exchange names in Tiingo's supported-tickers listing.
US_EXCHANGES = frozenset(
    {"NYSE", "NASDAQ", "NYSE ARCA", "NYSE MKT", "NYSE AMERICAN", "AMEX", "BATS", "NYSE NAT"}
)
OTC_EXCHANGES = frozenset({"OTC", "OTCBB", "OTCMKTS", "OTCQX", "OTCQB", "PINK", "OTCPINK"})
NON_US_EXCHANGES = frozenset({"SHE", "SHG", "LSE", "TSX", "HKSE", "ASX", "XETRA", "EURONEXT"})
STOCK_ASSET_TYPES = frozenset({"STOCK"})
DEFAULT_DELISTED_AFTER = 20  # sessions after the cutoff before "delisted or renamed"

_SEPARATORS = re.compile(r"[\s,;]+")
_PLAIN = re.compile(r"[A-Z]{1,5}")
_CLASS = re.compile(r"([A-Z]{1,5})[-./]([A-Z]{1,2})")
_EXCHANGE_SUFFIX = re.compile(r"([A-Z0-9]{1,10})[.:]([A-Z]{1,3})")
_DERIVED = re.compile(r"\^.*|.*=[A-Z]$|.*=X$")


@dataclass(frozen=True)
class ParsedTicker:
    """One entry as typed and what it became: a ticker, or a refusal with a reason."""

    entered: str
    ticker: str | None
    reason: SymbolReason | None  # None when `ticker` is set
    detail: str

    @property
    def ok(self) -> bool:
        return self.ticker is not None

    @property
    def rewritten(self) -> bool:
        return self.ok and self.ticker != self.entered.strip().upper()


def parse_ticker(entered: str) -> ParsedTicker:
    """Normalize one entry by the formatting rules (module docstring); never guesses."""

    text = entered.strip().upper()
    if not text:
        return ParsedTicker(entered, None, SymbolReason.EMPTY, "Nothing was entered.")
    if _DERIVED.fullmatch(text):
        return ParsedTicker(
            entered,
            None,
            SymbolReason.INDEX_OR_DERIVED,
            "This looks like an index, currency or futures symbol, not a stock.",
        )
    if _PLAIN.fullmatch(text):
        return ParsedTicker(entered, text, None, "")
    exchange = _EXCHANGE_SUFFIX.fullmatch(text)
    if exchange and exchange.group(2) in NON_US_SUFFIXES:
        return ParsedTicker(
            entered,
            None,
            SymbolReason.NON_US_SUFFIX,
            f"The suffix .{exchange.group(2)} names a listing outside the US. This lab studies "
            "US-listed stocks only and never substitutes another listing.",
        )
    share_class = _CLASS.fullmatch(text)
    if share_class:
        ticker = f"{share_class.group(1)}-{share_class.group(2)}"
        detail = "" if ticker == text else f"Written as {ticker}, the provider's form."
        return ParsedTicker(entered, ticker, None, detail)
    return ParsedTicker(
        entered,
        None,
        SymbolReason.INVALID_FORMAT,
        "A US ticker is 1-5 letters, optionally with a share class such as BRK.B.",
    )


def parse_tickers(text: str) -> tuple[ParsedTicker, ...]:
    """
    Every entry of pasted text (separated by spaces, commas, semicolons or new
    lines), in order, without duplicates by normalized ticker (the first
    entry wins). Refuses more than MAX_TICKERS_PER_PASTE entries.
    """

    entries = [part for part in _SEPARATORS.split(text) if part]
    if len(entries) > MAX_TICKERS_PER_PASTE:
        raise ValueError(f"paste at most {MAX_TICKERS_PER_PASTE} tickers at a time")
    parsed, seen = [], set()
    for entry in entries:
        result = parse_ticker(entry)
        key = result.ticker or result.entered.strip().upper()
        if key in seen:
            continue
        seen.add(key)
        parsed.append(result)
    return tuple(parsed)


# --- classification against a provider listing ---


@dataclass(frozen=True)
class ListingRow:
    """One row of a provider's listing, as served (Tiingo's supported tickers)."""

    ticker: str
    exchange: str
    asset_type: str
    price_currency: str
    start_date: date | None
    end_date: date | None


@dataclass(frozen=True)
class Classification:
    ticker: str
    status: SymbolStatus
    reason: SymbolReason
    detail: str
    row: ListingRow | None = None  # the row that decided, when one did


def _ineligible(row: ListingRow) -> tuple[SymbolReason, str] | None:
    exchange = row.exchange.strip().upper()
    if row.asset_type.strip().upper() not in STOCK_ASSET_TYPES:
        return SymbolReason.NOT_A_STOCK, f"The provider lists it as {row.asset_type or 'unknown'}."
    if exchange in OTC_EXCHANGES:
        return SymbolReason.OVER_THE_COUNTER, f"It trades over the counter ({row.exchange})."
    if exchange in NON_US_EXCHANGES:
        return SymbolReason.NON_US_EXCHANGE, f"It is listed outside the US ({row.exchange})."
    if exchange not in US_EXCHANGES:
        return (
            SymbolReason.UNKNOWN_EXCHANGE,
            f"Its exchange ({row.exchange or 'none'}) is not a recognized US exchange.",
        )
    if row.price_currency.strip().upper() != "USD":
        return (
            SymbolReason.NON_USD,
            f"It is priced in {row.price_currency or 'an unknown currency'}.",
        )
    if row.end_date is None:
        return SymbolReason.NO_PRICE_HISTORY, "The provider has no price history for it."
    return None


def classify(
    ticker: str,
    rows: Iterable[ListingRow],
    *,
    recent_cutoff: date,
    sessions_since: Callable[[date], int],
    delisted_after: int = DEFAULT_DELISTED_AFTER,
) -> Classification:
    """
    The status of `ticker` against listing `rows` (rules in the module
    docstring). `recent_cutoff` is the last session a current listing must
    reach; `sessions_since(day)` counts sessions after `day` up to the cutoff.
    """

    matching = [row for row in rows if row.ticker.strip().upper() == ticker]
    if not matching:
        return Classification(
            ticker,
            SymbolStatus.UNSUPPORTED,
            SymbolReason.NOT_LISTED,
            "The data provider does not list this ticker.",
        )
    eligible, refusals = [], []
    for row in matching:
        refusal = _ineligible(row)
        if refusal is None:
            eligible.append(row)
        else:
            refusals.append((refusal, row))
    if not eligible:
        (reason, detail), row = refusals[0]
        return Classification(ticker, SymbolStatus.UNSUPPORTED, reason, detail, row)
    current = [row for row in eligible if row.end_date >= recent_cutoff]
    if len(current) > 1:
        exchanges = ", ".join(sorted({row.exchange for row in current}))
        return Classification(
            ticker,
            SymbolStatus.AMBIGUOUS,
            SymbolReason.SEVERAL_LISTINGS,
            f"The provider lists several current securities under this ticker ({exchanges}); "
            "the lab does not guess which one is meant.",
        )
    if current:
        row = current[0]
        return Classification(
            ticker,
            SymbolStatus.SUPPORTED,
            SymbolReason.LISTED_US_STOCK,
            f"A stock listed on {row.exchange}, priced in USD.",
            row,
        )
    latest = max(eligible, key=lambda row: row.end_date)
    behind = sessions_since(latest.end_date)
    if behind <= delisted_after:
        return Classification(
            ticker,
            SymbolStatus.INACTIVE,
            SymbolReason.NO_RECENT_PRICES,
            f"No prices since {latest.end_date.isoformat()} ({behind} sessions).",
            latest,
        )
    return Classification(
        ticker,
        SymbolStatus.DELISTED,
        SymbolReason.COVERAGE_ENDED,
        f"The provider's prices end on {latest.end_date.isoformat()}. The ticker may have been "
        "delisted or renamed; the provider does not say which.",
        latest,
    )


def unverified(ticker: str) -> Classification:
    """A ticker with a valid format that has not been checked against a listing."""

    return Classification(
        ticker,
        SymbolStatus.UNVERIFIED,
        SymbolReason.NOT_CHECKED,
        "Not yet checked against the data provider's listing.",
    )


def refused(parsed: ParsedTicker) -> Classification:
    """The classification of an entry that failed the formatting rules."""

    if parsed.ok:
        raise ValueError("refused() takes an entry that failed the formatting rules")
    return Classification(
        parsed.entered.strip().upper(), SymbolStatus.UNSUPPORTED, parsed.reason, parsed.detail
    )


def statuses_in_order(classifications: Sequence[Classification]) -> tuple[Classification, ...]:
    """Classifications sorted by ticker, a fixed order (nothing is ranked)."""

    return tuple(sorted(classifications, key=lambda c: c.ticker))
