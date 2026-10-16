"""
Bull-trap and bear-trap setups: an explicit, registered rule (numpy and
pandas only). Chart-pattern names are not evidence; this module counts how
often a precisely defined setup failed in a stock's own past, beside the
base rate of failure for all comparable breakouts, and says whether the
setup adds anything.

The rule (TrapRule; every parameter is shown with the results), on the
split-consistent level L, at session d:

    breakout   L_d > H_d, where H_d = max(L over the `breakout_window`
               sessions before d): a close above the prior closing high
    decline    min(L over the `low_window` sessions before d) <=
               (1 - decline) * max(L over the `high_window` sessions before d),
               and that high came before that low: the price fell from the
               longer high to the recent low
    rebound    L_d / min(L over the `low_window` sessions before d) - 1 >= rebound
    volume     optional: relative dollar volume at d >= min_relative_volume

    Breakouts are declustered: one within `cooldown` sessions after the
    previous counted breakout is skipped. A bull-trap SETUP is a counted
    breakout that also has the decline and the rebound (and the volume
    condition when set), so every setup is one of the breakouts it is
    compared with.

    FAILURE (the breakout failed): within the `failure_window` sessions after
    d, some close falls below H_d * (1 - failure_margin), back under the
    level it broke out of.

Status, never upgraded without the full window:
    unresolved        the failure window has not fully passed (or passes the
                      outcome cutoff, or has a gap): a present-day setup is
                      always this, never "confirmed"
    confirmed trap    resolved and failed
    held              resolved and did not fail

Bear traps mirror the rule on 1 / L (a breakdown below the prior closing
low after a rally that then recovers above it).

Evaluation (resolved events only):
    base rate    P(failure | breakout), over all declustered breakouts, also
                 within the stock's volatility terciles
    matched base the base rate weighted by the setups' own mix of volatility
                 terciles: a fixed failure margin is easier to breach in a
                 volatile spell, and setups follow declines, which come with
                 higher volatility
    setup rate   P(failure | setup)
    lift         setup rate / matched base rate (1 means the setup adds
                 nothing beyond its volatility); the unmatched ratio is kept
    precision    TP / (TP + FP), recall TP / (TP + FN), false-alarm rate
                 FP / (FP + TN), where a breakout is flagged when it is a
                 setup and positive when it failed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stock_agent.stock_lab.reversal import Proportion, proportion, volatility_regime

UNRESOLVED = "unresolved"
CONFIRMED = "confirmed_trap"
HELD = "held"


@dataclass(frozen=True)
class TrapRule:
    decline: float = 0.10
    high_window: int = 60
    low_window: int = 20
    rebound: float = 0.05
    breakout_window: int = 20
    failure_window: int = 10
    failure_margin: float = 0.02
    min_relative_volume: float | None = None
    cooldown: int = 10

    def __post_init__(self) -> None:
        for name in ("decline", "rebound", "failure_margin"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("high_window", "low_window", "breakout_window", "failure_window", "cooldown"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

    def as_dict(self) -> dict:
        return asdict(self)


def _prior(values: pd.Series, window: int, how: str) -> pd.Series:
    rolled = values.shift(1).rolling(window, min_periods=window)
    return rolled.max() if how == "max" else rolled.min()


def _decluster(flags: np.ndarray, cooldown: int) -> np.ndarray:
    kept = np.zeros(len(flags), dtype=bool)
    last = -(10**9)
    for i in np.flatnonzero(flags):
        if i - last > cooldown:
            kept[i] = True
            last = i
    return kept


def trap_events(
    frame: pd.DataFrame,
    rule: TrapRule | None = None,
    *,
    kind: str = "bull",
    outcome_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    One row per declustered breakout: session, level, the broken level,
    is_setup, status (unresolved / confirmed_trap / held) and failed (NaN
    while unresolved), plus the volatility regime at the breakout.
    """

    rule = rule or TrapRule()
    if kind not in ("bull", "bear"):
        raise ValueError("kind must be bull or bear")
    level = frame["level"].astype("float64")
    series = level if kind == "bull" else 1.0 / level
    broken = _prior(series, rule.breakout_window, "max")
    low = _prior(series, rule.low_window, "min")
    high = _prior(series, rule.high_window, "max")
    breakout = (series > broken).to_numpy() & broken.notna().to_numpy()
    decline = (low <= (1.0 - rule.decline) * high).to_numpy()
    rebound = (series / low - 1.0 >= rule.rebound).to_numpy()
    breakouts = _decluster(breakout, rule.cooldown)
    setups = breakouts & decline & rebound
    if rule.min_relative_volume is not None:
        volume = frame["rel_dollar_volume"].to_numpy(dtype="float64")
        setups &= volume >= rule.min_relative_volume
    values_all = series.to_numpy()
    for i in np.flatnonzero(setups):  # the high must come before the low
        high_at = i - rule.high_window + int(np.nanargmax(values_all[i - rule.high_window : i]))
        low_at = i - rule.low_window + int(np.nanargmin(values_all[i - rule.low_window : i]))
        if not high_at < low_at:
            setups[i] = False

    sessions = pd.DatetimeIndex(frame["session"])
    values = series.to_numpy()
    usable = frame["price_return"].notna().to_numpy()
    regimes = volatility_regime(frame).to_numpy()
    rows = []
    for i in np.flatnonzero(breakouts):
        end = i + rule.failure_window
        threshold = broken.iloc[i] * (1.0 - rule.failure_margin)
        window = values[i + 1 : end + 1]
        complete = (
            end < len(frame)
            and usable[i + 1 : end + 1].all()
            and (outcome_end is None or sessions[end] <= outcome_end)
        )
        failed_early = bool((window < threshold).any())
        if complete:
            status = CONFIRMED if failed_early else HELD
            failed = float(failed_early)
        else:
            status, failed = UNRESOLVED, np.nan
        rows.append(
            {
                "session": sessions[i],
                "kind": kind,
                "is_breakout": bool(breakouts[i]),
                "is_setup": bool(setups[i]),
                "broken_level": float(broken.iloc[i]),
                "status": status,
                "failed": failed,
                "regime": regimes[i],
            }
        )
    columns = [
        "session",
        "kind",
        "is_breakout",
        "is_setup",
        "broken_level",
        "status",
        "failed",
        "regime",
    ]
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True, eq=False)
class TrapEvaluation:
    rule: TrapRule
    kind: str
    setup_rate: Proportion
    base_rate: Proportion
    matched_base_rate: float
    lift: float  # against the matched base rate
    unmatched_lift: float
    precision: float
    recall: float
    false_alarm_rate: float
    by_regime: pd.DataFrame
    unresolved_setups: int
    latest_setup: pd.Series | None  # the most recent setup, any status


def evaluate_traps(events: pd.DataFrame, rule: TrapRule, kind: str = "bull") -> TrapEvaluation:
    resolved = events[events["status"] != UNRESOLVED]
    breakouts = resolved[resolved["is_breakout"]]
    setups = resolved[resolved["is_setup"]]
    setup_rate = proportion(setups["failed"].to_numpy())
    base_rate = proportion(breakouts["failed"].to_numpy())
    weights, rates = [], []
    for regime in ("calm", "middle", "volatile"):
        share = float((setups["regime"] == regime).mean()) if len(setups) else 0.0
        in_regime = breakouts.loc[breakouts["regime"] == regime, "failed"]
        if share and len(in_regime):
            weights.append(share)
            rates.append(float(in_regime.mean()))
    matched = (
        float(np.dot(weights, rates) / sum(weights)) if weights and sum(weights) else float("nan")
    )

    def ratio_of(a: float, b: float) -> float:
        return a / b if b and np.isfinite(a) and np.isfinite(b) else float("nan")

    lift = ratio_of(setup_rate.estimate, matched)
    unmatched = ratio_of(setup_rate.estimate, base_rate.estimate)
    flagged = breakouts["is_setup"].to_numpy()
    positive = breakouts["failed"].to_numpy() == 1.0
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    tn = int((~flagged & ~positive).sum())

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    regime_rows = []
    for regime in ("calm", "middle", "volatile"):
        subset = breakouts[breakouts["regime"] == regime]
        base = proportion(subset["failed"].to_numpy())
        setup = proportion(subset.loc[subset["is_setup"], "failed"].to_numpy())
        regime_rows.append(
            {
                "regime": regime,
                "breakouts": base.n,
                "base_rate": base.estimate,
                "setups": setup.n,
                "setup_rate": setup.estimate,
                "status": setup.status,
            }
        )
    all_setups = events[events["is_setup"]]
    latest = None if all_setups.empty else all_setups.iloc[-1]
    return TrapEvaluation(
        rule=rule,
        kind=kind,
        setup_rate=setup_rate,
        base_rate=base_rate,
        matched_base_rate=matched,
        lift=lift,
        unmatched_lift=unmatched,
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
        false_alarm_rate=ratio(fp, fp + tn),
        by_regime=pd.DataFrame(regime_rows),
        unresolved_setups=int((events["is_setup"] & (events["status"] == UNRESOLVED)).sum()),
        latest_setup=latest,
    )
