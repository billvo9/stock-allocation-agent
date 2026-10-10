"""
Market-regime labels for descriptive filtering, in two strictly separate kinds.

Point-in-time regimes use only information available at the close of each
date: stored features of that date (trailing windows that end at its close)
and thresholds estimated from EARLIER dates only. A regime known at date t
never changes when later data arrive (tested by prefix stability).

    universe volatility   cross-sectional median of volatility_20d on date t,
                          "high" if above the median of that series over the
                          THRESHOLD_WINDOW earlier dates (at least MIN_HISTORY)
    universe trend        sign of the cross-sectional mean of momentum_20d on t

Both describe the run's own universe, not the market: with a handful of
hindsight-selected names they are universe states.

Retrospective (ex post) episodes are calendar windows whose boundaries were
identified only after they ended (peak and trough are known in hindsight).
They may slice results descriptively; they are never a feature, signal or
decision input, and they are labelled as retrospective wherever shown.

Filtering by either kind is descriptive: it never produces new inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_HISTORY = 63
THRESHOLD_WINDOW = 252  # trailing year: an all-history median drifts with the volatility level
UNAVAILABLE = "unavailable"
INSUFFICIENT_HISTORY = "insufficient history"
OUTSIDE_EPISODES = "outside listed episodes"


@dataclass(frozen=True)
class RegimeDefinition:
    key: str
    label: str
    point_in_time: bool
    description: str
    source_column: str | None = None


UNIVERSE_VOLATILITY = RegimeDefinition(
    "universe_volatility_pit",
    "Universe volatility (point-in-time)",
    True,
    "Cross-sectional median of the stored volatility_20d feature on each date, 'high' when "
    "above the median of that series over the previous 252 dates (dates before it only). "
    "Known at each date's close.",
    "volatility_20d",
)
UNIVERSE_TREND = RegimeDefinition(
    "universe_trend_pit",
    "Universe trend (point-in-time)",
    True,
    "Sign of the cross-sectional mean of the stored momentum_20d feature on each date. "
    "Known at each date's close; no estimated threshold.",
    "momentum_20d",
)
DRAWDOWN_EPISODES = RegimeDefinition(
    "drawdown_episode_ex_post",
    "Drawdown episodes (retrospective, ex post)",
    False,
    "S&P 500 closing peak-to-trough windows identified after they ended. For descriptive "
    "slicing only; never a feature, signal or decision input.",
)

# (label, first session, last session): S&P 500 closing peak to closing trough.
EX_POST_EPISODES = (
    ("COVID-19 crash, ex post (2020-02-19 to 2020-03-23)", "2020-02-19", "2020-03-23"),
    ("2022 bear market, ex post (2022-01-03 to 2022-10-12)", "2022-01-03", "2022-10-12"),
)

DEFINITIONS = (UNIVERSE_VOLATILITY, UNIVERSE_TREND, DRAWDOWN_EPISODES)


def _date_series(inputs: pd.DataFrame, column: str, how: str) -> pd.Series:
    values = inputs[["date", column]].dropna()
    return values.groupby("date")[column].agg(how).sort_index()


def volatility_regime(
    inputs: pd.DataFrame, *, window: int = THRESHOLD_WINDOW, min_history: int = MIN_HISTORY
) -> pd.Series:
    """'high' / 'normal' / 'insufficient history' per date, from earlier dates only."""

    level = _date_series(inputs, "volatility_20d", "median")
    # shift(1): the threshold for date t uses dates strictly before t.
    threshold = level.shift(1).rolling(window, min_periods=min_history).median()
    labels = np.where(
        threshold.isna(), INSUFFICIENT_HISTORY, np.where(level > threshold, "high", "normal")
    )
    return pd.Series(labels, index=level.index, name=UNIVERSE_VOLATILITY.key)


def trend_regime(inputs: pd.DataFrame) -> pd.Series:
    """'up' / 'down' / 'flat' per date: the sign of that date's mean momentum_20d."""

    level = _date_series(inputs, "momentum_20d", "mean")
    labels = np.select([level > 0, level < 0], ["up", "down"], default="flat")
    return pd.Series(labels, index=level.index, name=UNIVERSE_TREND.key)


def ex_post_episode(dates: pd.Series) -> pd.Series:
    """The retrospective episode containing each date (or 'outside listed episodes')."""

    days = pd.to_datetime(dates, utc=True).dt.normalize()
    labels = pd.Series(OUTSIDE_EPISODES, index=dates.index, name=DRAWDOWN_EPISODES.key)
    for label, first, last in EX_POST_EPISODES:
        inside = days.between(pd.Timestamp(first, tz="UTC"), pd.Timestamp(last, tz="UTC"))
        labels[inside] = label
    return labels


def regime_table(inputs: pd.DataFrame, dates: pd.Series) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    One row per evaluation date with every regime label, and the reason each
    unavailable point-in-time regime could not be formed from stored inputs.
    """

    table = pd.DataFrame({"date": pd.Series(sorted(pd.unique(dates)))})
    reasons = {}
    for definition, builder in (
        (UNIVERSE_VOLATILITY, volatility_regime),
        (UNIVERSE_TREND, trend_regime),
    ):
        if definition.source_column not in inputs.columns:
            reasons[definition.key] = f"stored inputs have no {definition.source_column} column"
            table[definition.key] = UNAVAILABLE
            continue
        labels = builder(inputs)
        table[definition.key] = table["date"].map(labels).fillna(UNAVAILABLE)
    table[DRAWDOWN_EPISODES.key] = ex_post_episode(table["date"])
    return table, reasons
