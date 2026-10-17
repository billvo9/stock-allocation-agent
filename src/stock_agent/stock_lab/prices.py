"""
Daily bars and the split-consistent returns the lab studies.

Bars
    One row per session: `session` (00:00 UTC of the session date, the
    project's convention), the raw as-traded `open`, `high`, `low`, `close`
    and `volume`, the cash dividend per share `div_cash` whose ex-date is the
    session, and `split_factor` (new shares per old share for a split whose
    ex-date is the session, else 1). Raw means unadjusted: a vendor's
    adjusted prices are never used, because they are recomputed whenever a
    later dividend or split arrives, so today's adjusted history is not what
    anyone could have seen on an earlier date (not a point-in-time vintage).
    A session without a bar has a missing close.

The split-consistent level (the price index the features use)

        L_t = close_t * prod(split_factor_s for sessions s <= t)

    L uses only splits up to t, so it is point-in-time safe, and ratios
    L_t / L_s equal the split-consistent price change between s and t. The
    session return is

        r_t = L_t / L_(t-1) - 1 = close_t * split_factor_t / close_(t-1) - 1

    and is unavailable (with a reason) when either close is missing or not
    positive, when the split factor is not a positive finite number, or when
    a large split is not visible in the raw closes (`split_price_mismatch`:
    the closes look already adjusted, or the factor is wrong). A mismatch
    also breaks the level (it is missing at that session), so no window of
    the level spans it: the bases before and after cannot be reconciled.

    Price return excludes dividends. `total_return_reference` adds the
    dividend, (close_t + div_cash_t) * split_factor_t / close_(t-1) - 1, for
    display only: the lab's target is the price return.

Flags (data is flagged, never changed)
    `ex_dividend`        a cash dividend goes ex on the session: its price
                         return includes the mechanical drop;
    `split_applied`      a split factor other than 1 was applied;
    `split_unverified`   a split too small to check against the closes
                         (|log factor| < 3 * log 1.25);
    `large_move`         |log(1 + r)| >= log 1.9 without a split: a real move
                         or an unrecorded split, never silently corrected.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import (
    CATEGORY_CODES,
    Category,
    ReturnReason,
    StableBand,
)

BAR_COLUMNS = ("session", "open", "high", "low", "close", "volume", "div_cash", "split_factor")
PRICE_COLUMNS = ("open", "high", "low", "close")

# A 25% band around 1 on |log ratio|, as in the ETF adapter's split checks.
SPLIT_BAND = math.log(1.25)
LARGE_MOVE = math.log(1.9)


class BarsRefused(ValueError):
    """Bars that break the schema; nothing is computed from them."""


def _sessions(values: pd.Series) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if index.hasnans:
        raise BarsRefused("a session is missing")
    if not (index == index.normalize()).all():
        raise BarsRefused("sessions must be 00:00 UTC session dates")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise BarsRefused("sessions must be unique and increasing")
    return index


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """
    A checked copy of `bars` with exactly BAR_COLUMNS, float64 values and a
    default index. Prices must be positive where present; volume and dividends
    non-negative; split factors positive and finite (missing means 1).
    """

    missing = [column for column in BAR_COLUMNS if column not in bars.columns]
    if missing:
        raise BarsRefused(f"missing columns: {missing}")
    checked = pd.DataFrame({"session": _sessions(bars["session"])})
    for column in BAR_COLUMNS[1:]:
        values = pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype="float64")
        checked[column] = values
    for column in PRICE_COLUMNS:
        present = checked[column].dropna()
        if not (np.isfinite(present) & (present > 0)).all():
            raise BarsRefused(f"{column} must be positive and finite where present")
    for column in ("volume", "div_cash"):
        present = checked[column].dropna()
        if not (np.isfinite(present) & (present >= 0)).all():
            raise BarsRefused(f"{column} must be non-negative and finite where present")
    factors = checked["split_factor"].fillna(1.0)
    if not (np.isfinite(factors) & (factors > 0)).all():
        raise BarsRefused("split_factor must be positive and finite")
    checked["split_factor"] = factors
    checked["div_cash"] = checked["div_cash"].fillna(0.0)
    return checked


def on_calendar(bars: pd.DataFrame, sessions: Sequence[pd.Timestamp] | None) -> pd.DataFrame:
    """
    `bars` reindexed to `sessions` (every session in the bars' span), so a
    session without a bar becomes a row with missing prices; without a
    calendar the bars' own sessions are used. Refuses bars on a non-session.
    """

    checked = validate_bars(bars)
    if sessions is None or checked.empty:
        return checked
    calendar = pd.DatetimeIndex(sessions)
    outside = ~checked["session"].isin(calendar)
    if outside.any():
        first = checked.loc[outside, "session"].iloc[0]
        raise BarsRefused(f"a bar is dated on a non-session: {first.date()}")
    first, last = checked["session"].iloc[0], checked["session"].iloc[-1]
    span = calendar[(calendar >= first) & (calendar <= last)]
    reindexed = checked.set_index("session").reindex(span)
    reindexed["split_factor"] = reindexed["split_factor"].fillna(1.0)
    reindexed["div_cash"] = reindexed["div_cash"].fillna(0.0)
    return reindexed.rename_axis("session").reset_index()


def split_level(bars: pd.DataFrame) -> pd.Series:
    """L_t = close_t * cumulative split factor through t (module docstring)."""

    cumulative = bars["split_factor"].to_numpy(dtype="float64").cumprod()
    return pd.Series(bars["close"].to_numpy(dtype="float64") * cumulative, index=bars.index)


def session_returns(bars: pd.DataFrame, band: StableBand | None = None) -> pd.DataFrame:
    """
    One row per bar: `session`, `close`, `level`, `price_return`, `reason`,
    `category` (when `band` is given; a category code from CATEGORY_CODES as a
    nullable integer in `category_code`), `total_return_reference` and the
    flags in the module docstring. `bars` must already be on its calendar.
    """

    close = bars["close"].to_numpy(dtype="float64")
    factor = bars["split_factor"].to_numpy(dtype="float64")
    dividend = bars["div_cash"].to_numpy(dtype="float64")
    previous = np.concatenate([[np.nan], close[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close * factor / previous
        raw_ratio = close / previous
        total = (close + dividend) * factor / previous - 1.0

    reasons = np.full(len(close), ReturnReason.OK.value, dtype=object)
    reasons[np.isnan(previous)] = ReturnReason.PREVIOUS_CLOSE_MISSING.value
    if len(reasons):
        reasons[0] = ReturnReason.FIRST_CLOSE.value
    reasons[np.isnan(close)] = ReturnReason.MISSING_CLOSE.value

    applied = factor != 1.0
    discriminable = np.abs(np.log(factor)) >= 3 * SPLIT_BAND
    with np.errstate(divide="ignore", invalid="ignore"):
        mismatch = applied & discriminable & ~(np.abs(np.log(ratio)) < SPLIT_BAND)
    available = reasons == ReturnReason.OK.value
    reasons[available & mismatch] = ReturnReason.SPLIT_PRICE_MISMATCH.value
    available = reasons == ReturnReason.OK.value

    price_return = np.where(available, ratio - 1.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        large = available & ~applied & (np.abs(np.log(raw_ratio)) >= LARGE_MOVE)
    result = pd.DataFrame(
        {
            "session": bars["session"].to_numpy(),
            "close": close,
            # A mismatched split breaks the level: the closes before and after it
            # cannot be put on one share basis, so no window may span that session.
            "level": np.where(mismatch, np.nan, split_level(bars).to_numpy()),
            "price_return": price_return,
            "reason": reasons.astype(str),
            "total_return_reference": np.where(available, total, np.nan),
            "ex_dividend": dividend > 0,
            "split_applied": applied,
            "split_unverified": applied & ~discriminable,
            "large_move": large,
        }
    )
    result["session"] = pd.DatetimeIndex(result["session"])
    if band is not None:
        result["category"] = categorize(result["price_return"], band)
        result["category_code"] = category_codes(result["category"])
    return result


def categorize(returns: pd.Series, band: StableBand) -> pd.Series:
    """Each return's category value ("increase", "stable", "decrease"), None where missing."""

    values = returns.to_numpy(dtype="float64")
    labels = np.full(len(values), None, dtype=object)
    labels[values > band.tau] = Category.INCREASE.value
    labels[values < -band.tau] = Category.DECREASE.value
    labels[(values >= -band.tau) & (values <= band.tau)] = Category.STABLE.value
    return pd.Series(labels, index=returns.index, dtype=object)


def category_codes(categories: pd.Series) -> pd.Series:
    """Integer codes (Decrease 0, Stable 1, Increase 2) as a nullable Int64 series."""

    codes = {category.value: code for category, code in CATEGORY_CODES.items()}
    return categories.map(codes).astype("Int64")


def stable_share(categories: pd.Series) -> float | None:
    """The fraction of sessions with a category that were Stable; None without any."""

    known = categories.dropna()
    if known.empty:
        return None
    return float((known == Category.STABLE.value).mean())
