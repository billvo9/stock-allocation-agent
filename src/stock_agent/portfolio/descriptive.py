"""
Descriptive fund-history calculations: OD8 group A of
`docs/research/t5_portfolio_formulas.md`, approved by the owner on
2026-10-14 (A1-A8, R1, R2, R4, R5a, R5d; R3 wording; R5b and R5c handled
conservatively as the owner directed).

Pure functions on pandas inputs: no I/O, no clock, no app or UI import.

- `total_returns` (A1, R1, R5a, R5b): total returns from raw closes, cash
  distributions and share-count splits, on one consistent share basis. A
  return that spans sessions without a close is a labelled multi-session
  return, never a one-session return; a distribution or an ambiguous
  share basis inside such a span makes it unavailable.
- `common_start` (R4): the shared start t0 of the funds compared.
- `growth_index` (A4, R5a): hypothetical growth of $10,000, distributions
  reinvested on the ex-date (a modelling convention, not actual cash
  flows), before taxes and transaction costs. Fund expenses are already in
  the prices and are not deducted again (A5).
- `drawdown_summary` (A6): the worst decline at the closes that exist, and
  whether that is the full-period maximum drawdown or only an observed
  decline over an incomplete history.
- `calendar_year_returns` (A7, R5d): year-end close to year-end close;
  partial years labelled, never annualized; a missing year-end close makes
  both affected years unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

GROWTH_START_VALUE = 10_000.0
CALENDAR_COLUMNS = ("year", "total_return", "partial", "losing", "status", "reason")
RETURN_COLUMNS = (
    "date",
    "close_observed",
    "total_return",
    "sessions_spanned",
    "status",
    "reason",
)

OK = "ok"
UNAVAILABLE = "unavailable"

# Machine reasons; the page translates them into words.
NO_CLOSE = "missing_input"
NO_PRIOR_CLOSE = "no_prior_close"
DISTRIBUTION_IN_GAP = "distribution_in_gap"
SPLIT_IN_GAP_WITH_DISTRIBUTION = "split_and_distribution_in_gap"
SHARE_BASIS_UNKNOWN = "share_basis_unknown_with_split"
YEAR_END_MISSING = "year_end_value_unavailable"
PERIOD_END_MISSING = "period_edge_value_unavailable"
INSUFFICIENT_HISTORY = "insufficient_history"

UNADJUSTED = "unadjusted"
PER_SHARE_AS_PAID = "per_share_as_paid"
PRE_SPLIT = "pre_split"
POST_SPLIT = "post_split"


class MetricUnavailable(Exception):
    """
    A metric cannot be given. `reason` is a machine-readable code; `detail`
    says why in plain words a reader can understand.
    """

    def __init__(self, metric: str, reason: str, detail: str) -> None:
        super().__init__(f"{metric} unavailable ({reason}): {detail}")
        self.metric = metric
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class DrawdownSummary:
    """
    The worst decline of one growth series (A6, F6.5), measured close to close.

    - `max_drawdown`: the largest percentage decline from a previous high, as
      a positive fraction (0.25 is a 25% decline).
    - `peak_date`, `trough_date`: the sessions of that episode's high and low.
    - `recovery_date`: the first session back at or above the peak, or None.
    - `duration_sessions`, `duration_days`: from the peak to the recovery when
      recovered; otherwise from the peak to the last session with a value.
    - `dollar_decline`: that episode's peak value minus its trough value.
    - `history_complete`: True only when every session from the start to the
      end of the series has a value. Then `max_drawdown` is the full-period
      maximum drawdown. Otherwise it is only the largest decline observed at
      the closes that exist; the full-period maximum is unavailable and may
      be deeper.
    - `missing_sessions`: sessions after the start without a value.
    """

    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    duration_sessions: int
    duration_days: int
    dollar_decline: float
    history_complete: bool
    missing_sessions: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown is a fraction in [0, 1]")
        if self.trough_date < self.peak_date:
            raise ValueError("the trough cannot precede the peak")
        if self.recovery_date is not None and self.recovery_date < self.trough_date:
            raise ValueError("the recovery cannot precede the trough")
        if min(self.duration_sessions, self.duration_days) < 0 or self.dollar_decline < 0:
            raise ValueError("durations and the dollar decline are non-negative")
        if self.missing_sessions < 0 or self.history_complete != (self.missing_sessions == 0):
            raise ValueError("history is complete exactly when no session is missing")

    @property
    def recovered(self) -> bool:
        """True exactly when a recovery session is recorded."""

        return self.recovery_date is not None


# --- A1, R1, R5a, R5b: total returns from raw records ---


def _events(frame: pd.DataFrame | None, date_column: str, columns: tuple[str, ...]) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=[date_column, *columns])
    missing = {date_column, *columns} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    return frame


def total_returns(
    sessions: pd.DatetimeIndex,
    closes: pd.Series,
    distributions: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    *,
    close_basis: str,
    amount_basis: str,
) -> pd.DataFrame:
    """
    One row per session in `sessions` (A1, R1, R5a, R5b), with the columns
    `RETURN_COLUMNS`.

    Inputs (R1: one consistent share basis, nothing adjusted twice):
    - `closes`: raw, unadjusted closes indexed by the sessions that have one.
      Vendor-adjusted closes are refused (`close_basis` must be
      "unadjusted"); distributions are never added to adjusted prices.
    - `distributions`: `ex_date`, `amount` per share as paid
      (`amount_basis` "per_share_as_paid"), and optionally `share_basis`
      ("pre_split" or "post_split"), which is needed whenever a split takes
      effect within the same return's span.
    - `close_basis` and `amount_basis` must be stated by every caller; there
      is no default, so an unstated basis can never pass.
    - `splits`: `effective_date`, `ratio` (new shares per old share).

    For a session t with a close, let s be the previous session with a
    close, f the product of the split ratios effective in (s, t] and D the
    distribution received per share held at s:

        r_t = (f * P_t + D) / P_s - 1

    - `sessions_spanned` is the number of sessions in (s, t]. When it is
      above 1, the return is a multi-session return over sessions without
      a close (R5a): it is labelled as such and is never a one-session
      return; the sessions between carry no value.
    - A distribution whose ex-date falls on a session without a close is
      not reinvested through the gap (R5b not approved): that return is
      unavailable (`distribution_in_gap`), and so is everything chained to
      it.
    - A distribution in a span where a split takes effect needs its share
      basis; without it the return is unavailable (`share_basis_unknown_with_split`).
    - The first session with a close has no prior close (`no_prior_close`).
    """

    if close_basis != UNADJUSTED:
        raise ValueError("total returns are computed only from unadjusted closes (R1)")
    if amount_basis != PER_SHARE_AS_PAID:
        raise ValueError("distribution amounts must be per share as paid (R1)")
    sessions = pd.DatetimeIndex(sessions)
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("sessions must be strictly increasing")
    closes = closes.astype("float64")
    if not closes.index.isin(sessions).all():
        raise ValueError("every close must be on a session")
    if not (np.isfinite(closes.to_numpy()) & (closes.to_numpy() > 0)).all():
        raise ValueError("closes must be finite and positive")
    distributions = _events(distributions, "ex_date", ("amount",))
    splits = _events(splits, "effective_date", ("ratio",))
    if not pd.DatetimeIndex(distributions["ex_date"]).isin(sessions).all():
        raise ValueError("every ex-date must be a session")
    if not pd.DatetimeIndex(splits["effective_date"]).isin(sessions).all():
        raise ValueError("every split must take effect on a session")
    if "share_basis" in distributions.columns:
        bases = distributions["share_basis"].dropna()
        if not bases.isin([PRE_SPLIT, POST_SPLIT]).all():
            raise ValueError("share_basis must be 'pre_split' or 'post_split'")
    amounts = distributions["amount"].astype("float64").to_numpy()
    if not (np.isfinite(amounts) & (amounts > 0)).all():
        raise ValueError("distribution amounts must be finite and positive")
    ratios = splits["ratio"].astype("float64").to_numpy()
    if not (np.isfinite(ratios) & (ratios > 0)).all():
        raise ValueError("split ratios must be finite and positive")

    observed = sessions.isin(closes.index)
    close_values = closes.reindex(sessions).to_numpy(dtype="float64")
    dist_at = sessions.get_indexer(pd.DatetimeIndex(distributions["ex_date"]))
    dist_basis = (
        distributions["share_basis"].to_numpy(dtype=object)
        if "share_basis" in distributions.columns
        else np.full(len(distributions), None, dtype=object)
    )
    split_at = sessions.get_indexer(pd.DatetimeIndex(splits["effective_date"]))
    rows: list[tuple[object, ...]] = []
    previous: int | None = None
    for position, session in enumerate(sessions):
        if not observed[position]:
            rows.append((session, False, np.nan, 0, UNAVAILABLE, NO_CLOSE))
            continue
        if previous is None:
            rows.append((session, True, np.nan, 0, UNAVAILABLE, NO_PRIOR_CLOSE))
            previous = position
            continue
        in_span = (split_at > previous) & (split_at <= position)
        ratio = float(ratios[in_span].prod()) if in_span.any() else 1.0
        paid = (dist_at > previous) & (dist_at <= position)
        reason = None
        if (paid & (dist_at != position)).any():
            reason = SPLIT_IN_GAP_WITH_DISTRIBUTION if in_span.any() else DISTRIBUTION_IN_GAP
        cash = 0.0
        if reason is None and paid.any():
            if not in_span.any():
                cash = float(amounts[paid].sum())
            else:
                for amount, basis in zip(amounts[paid], dist_basis[paid], strict=True):
                    if basis == PRE_SPLIT:
                        cash += float(amount)
                    elif basis == POST_SPLIT:
                        cash += float(amount) * ratio
                    elif basis is None or (isinstance(basis, float) and np.isnan(basis)):
                        reason = SHARE_BASIS_UNKNOWN
                        break
                    else:
                        raise ValueError("share_basis must be 'pre_split' or 'post_split'")
        if reason is not None:
            rows.append((session, True, np.nan, 0, UNAVAILABLE, reason))
        else:
            value = (ratio * close_values[position] + cash) / close_values[previous] - 1.0
            rows.append((session, True, value, position - previous, OK, None))
        previous = position
    frame = pd.DataFrame(rows, columns=list(RETURN_COLUMNS))
    frame["date"] = pd.DatetimeIndex(frame["date"])
    frame["sessions_spanned"] = frame["sessions_spanned"].astype("int64")
    return frame


# --- R4: the shared start ---


def common_start(
    available_sessions: Mapping[str, pd.DatetimeIndex], window_start: pd.Timestamp
) -> pd.Timestamp | None:
    """
    The shared start t0 (R4): the first session, on or after the latest of
    the funds' first closes on or after `window_start`, at which every fund
    has a close. None when there is none. `available_sessions` maps each
    fund to the sorted sessions at which it has a close.
    """

    if not available_sessions:
        return None
    inside = {
        fund: pd.DatetimeIndex(sessions)[pd.DatetimeIndex(sessions) >= window_start]
        for fund, sessions in available_sessions.items()
    }
    if any(sessions.empty for sessions in inside.values()):
        return None
    latest_first = max(sessions[0] for sessions in inside.values())
    shared = None
    for sessions in inside.values():
        candidates = set(sessions[sessions >= latest_first])
        shared = candidates if shared is None else shared & candidates
    return min(shared) if shared else None


# --- A4, R5a: hypothetical growth of $10,000 ---


def growth_index(
    returns: pd.Series, sessions_spanned: pd.Series, start_value: float = GROWTH_START_VALUE
) -> pd.Series:
    """
    Hypothetical growth of `start_value` (A4), distributions reinvested on the
    ex-date, indexed like `returns`, whose first session is t0:

        G(t0) = start_value;  G(t) = G(s) * (1 + r_t)

    where s is the session `sessions_spanned[t]` positions before t. A
    multi-session return chains only from the last session with a value,
    across sessions that have none (R5a). A session whose return is missing
    has no value, and nothing chains through an unavailable return.
    """

    if not returns.index.equals(sessions_spanned.index):
        raise ValueError("returns and sessions_spanned must share their index")
    if len(returns) == 0:
        raise MetricUnavailable("growth_index", INSUFFICIENT_HISTORY, "no session in the period")
    values = np.full(len(returns), np.nan)
    values[0] = start_value
    rates = returns.to_numpy(dtype="float64")
    spans = sessions_spanned.to_numpy()
    for position in range(1, len(returns)):
        rate, span = rates[position], int(spans[position])
        if not np.isfinite(rate) or span < 1:
            continue
        origin = position - span
        if origin < 0 or not np.isfinite(values[origin]):
            continue
        if origin + 1 < position and np.isfinite(values[origin + 1 : position]).any():
            raise ValueError("a multi-session return may only span sessions without a value")
        values[position] = values[origin] * (1.0 + rate)
    return pd.Series(values, index=returns.index, name=returns.name)


# --- A6: drawdown and recovery ---


def drawdown_summary(growth: pd.Series) -> DrawdownSummary:
    """
    The worst decline of `growth` at the sessions that have a value (A6):
    DD_t = 1 - G_t / max_{s<=t} G_s, its peak, trough, dollar size and
    recovery. `history_complete` says whether every session from the start
    to the last session has a value; if not, the result is only the largest
    observed decline, never presented as the full-period maximum drawdown.
    """

    values = growth.dropna()
    if len(values) < 2:
        raise MetricUnavailable(
            "drawdown_summary", INSUFFICIENT_HISTORY, "fewer than two sessions with a value"
        )
    missing = int(growth.loc[growth.index[0] : growth.index[-1]].isna().sum())
    peaks = values.cummax()
    declines = 1.0 - values / peaks
    trough_date = declines.idxmax()
    max_drawdown = float(declines.loc[trough_date])
    peak_value = float(peaks.loc[trough_date])
    peak_date = values.loc[:trough_date][values.loc[:trough_date] == peak_value].index[-1]
    after = values.loc[trough_date:]
    back = after[after >= peak_value]
    if max_drawdown == 0.0:  # never below a previous high: nothing to recover from
        recovery_date = peak_date
    else:
        recovery_date = None if back.empty else back.index[0]
    end_date = recovery_date if recovery_date is not None else values.index[-1]
    position = growth.index.get_loc
    return DrawdownSummary(
        max_drawdown=max_drawdown,
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        duration_sessions=int(position(end_date) - position(peak_date)),
        duration_days=int((end_date - peak_date).days),
        dollar_decline=peak_value - float(values.loc[trough_date]),
        history_complete=missing == 0,
        missing_sessions=missing,
    )


# --- A7, R5d: calendar-year returns ---


def calendar_year_returns(growth: pd.Series, year_end_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """
    One row per calendar year that `growth` reaches past its start (A7):

        R_Y = G(last session of Y) / G(last session of Y - 1) - 1

    - A year whose start is t0 rather than the previous year's last
      session, or whose end is the series' last session rather than the
      year's last session, is `partial`, and is never annualized.
    - If a year-end session lies in the series but has no value (no close
      there, or a break in the growth chain), the years on both sides of it
      are unavailable (`year_end_value_unavailable`, R5d). A partial year
      whose period start or end has no value is unavailable too
      (`period_edge_value_unavailable`).
    - `losing` is decided on the unrounded return (it is False when the
      return is unavailable).

    `year_end_sessions` lists the last session of each calendar year of the
    data's calendar.
    """

    if growth.empty:
        return pd.DataFrame(columns=list(CALENDAR_COLUMNS))
    start, last = growth.index[0], growth.index[-1]
    ends = {int(session.year): session for session in pd.DatetimeIndex(year_end_sessions)}
    rows = []
    for year in range(start.year, last.year + 1):
        previous_end = ends.get(year - 1)
        if previous_end is not None and previous_end >= start:
            base_date, partial = previous_end, False
        else:
            base_date, partial = start, True
        year_end = ends.get(year)
        if year_end is not None and year_end <= last:
            end_date = year_end
        else:
            end_date, partial = last, True
        if end_date <= base_date:
            continue
        base, end = growth.get(base_date, np.nan), growth.get(end_date, np.nan)
        if np.isfinite(base) and np.isfinite(end):
            value = end / base - 1.0
            rows.append((year, value, partial, bool(value < 0.0), OK, None))
        else:
            missing_at = base_date if not np.isfinite(base) else end_date
            reason = YEAR_END_MISSING if missing_at in ends.values() else PERIOD_END_MISSING
            rows.append((year, np.nan, partial, False, UNAVAILABLE, reason))
    return pd.DataFrame(rows, columns=list(CALENDAR_COLUMNS))


class DescriptiveCalculator:
    """The approved group A calculations behind the Explore page's metrics interface."""

    def common_start(
        self, available_sessions: Mapping[str, pd.DatetimeIndex], window_start: pd.Timestamp
    ) -> pd.Timestamp | None:
        return common_start(available_sessions, window_start)

    def growth_index(
        self,
        returns: pd.Series,
        sessions_spanned: pd.Series,
        start_value: float = GROWTH_START_VALUE,
    ) -> pd.Series:
        return growth_index(returns, sessions_spanned, start_value)

    def drawdown_summary(self, growth: pd.Series) -> DrawdownSummary:
        return drawdown_summary(growth)

    def calendar_year_returns(
        self, growth: pd.Series, year_end_sessions: pd.DatetimeIndex
    ) -> pd.DataFrame:
        return calendar_year_returns(growth, year_end_sessions)
