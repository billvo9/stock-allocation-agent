"""
The Stock Lab's shared vocabulary (standard library only).

Target (the owner's brief, 2026-10-16)
    The daily price return of session t is the split-consistent close-to-close
    return

        r_t = close_t * split_factor_t / close_(t-1) - 1

    where `close` is the raw (as-traded, unadjusted) close and
    `split_factor_t` is the number of new shares per old share for a split
    whose ex-date is t (1 otherwise). It is a PRICE return: cash dividends
    are excluded, so an ex-dividend session shows the mechanical price drop.
    It is not the dividend-inclusive total return.

Categories (the stable band tau, configurable, 0.5% by default)
    Increase   r_t >  +tau
    Stable     -tau <= r_t <= +tau
    Decrease   r_t <  -tau

    The band is the same number for every stock, so it does not mean the
    same thing for all of them: for a stock whose daily moves are typically
    0.6% a ±0.5% band is wide (many Stable sessions); for one whose moves are
    typically 3% it is narrow (few). The Lab shows each stock's own share of
    Stable sessions beside the band for this reason.

Statuses
    A symbol's status says whether it can be studied at all; a data status
    says whether its stored prices are usable now. Both carry a
    machine-readable reason and plain words.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

DEFAULT_STABLE_BAND = 0.005

# The last session of development data: the owner's lockbox starts on
# 2025-01-01 (`model_validation.folds.MODEL_HOLDOUT_START`; equality tested).
# Real prices after it are not shown or scored until the owner decides how
# the sealed period may be used (docs/research/stock_probability_lab.md).
DEVELOPMENT_END = date(2024, 12, 31)
MIN_STABLE_BAND = 0.0001
MAX_STABLE_BAND = 0.05


class Category(StrEnum):
    """A session's direction against the stable band, in display order."""

    INCREASE = "increase"
    STABLE = "stable"
    DECREASE = "decrease"


CATEGORIES = (Category.INCREASE, Category.STABLE, Category.DECREASE)
CATEGORY_WORDS = {
    Category.INCREASE: "Increase",
    Category.STABLE: "Stable",
    Category.DECREASE: "Decrease",
}
# Integer codes for arrays: ordered so that Decrease < Stable < Increase
# (the order a ranked probability score needs).
CATEGORY_CODES = {Category.DECREASE: 0, Category.STABLE: 1, Category.INCREASE: 2}
CODE_CATEGORIES = {code: category for category, code in CATEGORY_CODES.items()}


@dataclass(frozen=True)
class StableBand:
    """The stable band tau, as a fraction (0.005 is ±0.5%)."""

    tau: float = DEFAULT_STABLE_BAND

    def __post_init__(self) -> None:
        if isinstance(self.tau, bool) or not isinstance(self.tau, int | float):
            raise TypeError("the stable band must be a number")
        if not MIN_STABLE_BAND <= float(self.tau) <= MAX_STABLE_BAND:
            raise ValueError(
                f"the stable band must be between {MIN_STABLE_BAND:.2%} and {MAX_STABLE_BAND:.0%}"
            )

    def categorize(self, r: float) -> Category | None:
        """The category of one return; None for a missing or non-finite return."""

        if r is None or not math.isfinite(r):
            return None
        if r > self.tau:
            return Category.INCREASE
        if r < -self.tau:
            return Category.DECREASE
        return Category.STABLE

    def words(self) -> str:
        """The band in words: ±0.5%, ±0.25%, ±1%."""

        return "±" + f"{self.tau * 100:.2f}".rstrip("0").rstrip(".") + "%"


DEFAULT_BAND = StableBand()


class SymbolStatus(StrEnum):
    """Whether a ticker can be studied, in words on the page."""

    UNVERIFIED = "unverified"  # valid format; no provider listing checked yet
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INACTIVE = "inactive"
    DELISTED = "delisted"
    AMBIGUOUS = "ambiguous"


SYMBOL_STATUS_WORDS = {
    SymbolStatus.UNVERIFIED: "Not yet checked",
    SymbolStatus.SUPPORTED: "Supported",
    SymbolStatus.UNSUPPORTED: "Unsupported",
    SymbolStatus.INACTIVE: "Inactive",
    SymbolStatus.DELISTED: "Delisted or renamed",
    SymbolStatus.AMBIGUOUS: "Ambiguous",
}


class SymbolReason(StrEnum):
    """Why a symbol has its status (machine-readable)."""

    NOT_CHECKED = "not_checked"
    LISTED_US_STOCK = "listed_us_stock"
    EMPTY = "empty"
    INVALID_FORMAT = "invalid_format"
    NON_US_SUFFIX = "non_us_suffix"
    INDEX_OR_DERIVED = "index_or_derived"
    NOT_LISTED = "not_listed_by_provider"
    NOT_A_STOCK = "not_a_stock"
    NON_US_EXCHANGE = "non_us_exchange"
    OVER_THE_COUNTER = "over_the_counter"
    UNKNOWN_EXCHANGE = "unknown_exchange"
    NON_USD = "non_usd_price_currency"
    NO_RECENT_PRICES = "no_recent_prices"
    COVERAGE_ENDED = "provider_coverage_ended"
    SEVERAL_LISTINGS = "several_active_listings"
    NO_PRICE_HISTORY = "no_price_history"


class DataStatus(StrEnum):
    """Whether a symbol's stored prices are usable as of now."""

    CURRENT = "current"
    STALE = "stale"
    UNCHECKED = "unchecked"  # stored, but no update job has checked freshness
    REVISED = "revised"  # a later download changed earlier sessions
    NO_DATA = "no_data"
    SYNTHETIC = "synthetic"


DATA_STATUS_WORDS = {
    DataStatus.CURRENT: "Current",
    DataStatus.STALE: "Stale",
    DataStatus.UNCHECKED: "Freshness not checked",
    DataStatus.REVISED: "Revised by a later download",
    DataStatus.NO_DATA: "Missing data",
    DataStatus.SYNTHETIC: "Synthetic demo",
}


class ReturnReason(StrEnum):
    """Why a session's return is unavailable (or flagged)."""

    OK = "ok"
    FIRST_CLOSE = "first_close"
    MISSING_CLOSE = "missing_close"
    PREVIOUS_CLOSE_MISSING = "previous_close_missing"
    NON_POSITIVE_CLOSE = "non_positive_close"
    INVALID_SPLIT_FACTOR = "invalid_split_factor"
    SPLIT_PRICE_MISMATCH = "split_price_mismatch"


class DataClass(StrEnum):
    """Where a price history came from; synthetic data is always labelled."""

    REAL = "real"
    SYNTHETIC = "synthetic"


class EvidenceStatus(StrEnum):
    """
    What a number on the page is worth as evidence. Nothing in this lab is
    validated: the best available status is a development evaluation on
    pre-2025 history of a hindsight-selected watchlist.
    """

    DESCRIPTIVE = "descriptive"  # counts and frequencies of the past only
    SYNTHETIC_DEMO = "synthetic_demo"
    EXPERIMENTAL_UNVALIDATED = "experimental_unvalidated"


EVIDENCE_WORDS = {
    EvidenceStatus.DESCRIPTIVE: "Describes the past only",
    EvidenceStatus.SYNTHETIC_DEMO: "Synthetic demo: not a real stock",
    EvidenceStatus.EXPERIMENTAL_UNVALIDATED: "Experimental, unvalidated",
}
