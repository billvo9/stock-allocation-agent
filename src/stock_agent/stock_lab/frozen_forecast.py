"""
The forecast record of a frozen, registered model (research side; the Lab's
page reads stored records only): what the real evening job records for one
stock from its inference window alone (stock_lab.evening).

    forecast_record(panel, symbol, model, ...) -> the checked artifact
        (forecast.py, schema 1.1)

What it records
    - the frozen estimator's probabilities (registry.probabilities) for the
      five sessions and the five-session total, at the model's band;
    - the B1 comparator: B1's forecast on the same panel, sessions and bands,
      inside the same record, never a forecast of its own (forecast.py);
    - the state the forecast was made in, from the window through the as-of
      close: the run, momentum over 5 and 20 sessions, volatility over 20
      sessions and its EWMA, and the volatility regime against the window's
      own terciles (descriptive, never a signal);
    - what followed the current run in development history, frozen with the
      model (registry.freeze); on a 2025+ window every such event would be
      censored, so none is computed from protected prices;
    - no trap signal: real trap flags wait for the owner's confirmation of
      the failure margin f (decision A5).

Point in time: the panel is built from the window only, so nothing after the
as-of close is read. The record states that an end-of-day forecast reflects
no overnight news and no premarket move.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab.contracts import EvidenceStatus
from stock_agent.stock_lab.forecast import PROBABILITY_KEYS
from stock_agent.stock_lab.frozen_model import BENCHMARK, RegistryRefused
from stock_agent.stock_lab.registry import FrozenModel, Probabilities, probabilities
from stock_agent.stock_lab.reversal import volatility_regime
from stock_agent.stock_lab.store import stamp
from stock_agent.stock_lab.targets import cumulative_band

MARKET_STATE = ("ret_5", "ret_20", "vol_20", "ewma_vol")
# First among a reconstruction's limitations (owner decision of 2026-10-30).
RECONSTRUCTED_LIMITATION = (
    "Reconstructed after the fact for a missed evening: made after its evening closed, from "
    "prices dated up to its as-of close only, as the provider served them when it was made, "
    "which may differ from what that evening could have read (a session corrected or first "
    "served later)."
)
LIMITATIONS = (
    (
        "Experimental and unvalidated: the evaluation is a development test of a "
        "hindsight-selected list, not evidence of skill."
    ),
    (
        "Each session's probabilities are for that day's own move, not the move from the "
        "as-of close to that day; the five-session total is shown separately."
    ),
    (
        "Made after the as-of close from end-of-day prices only: it reflects no overnight "
        "news and no premarket move."
    ),
    (
        "The development test scored this model with its running state taken from the full "
        "history; here it comes from the 252-session window (the frozen model's state check "
        "measures the difference)."
    ),
)
STREAK_CAP = 7  # the frozen reversal tables cover runs of 1 to 7 sessions


def _finite(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _mix(values) -> dict:
    return dict(zip(PROBABILITY_KEYS, (float(v) for v in values), strict=True))


def _normalized(probabilities: Probabilities) -> tuple[np.ndarray, np.ndarray]:
    daily = probabilities.daily / probabilities.daily.sum(axis=1, keepdims=True)
    cumulative = probabilities.cumulative / probabilities.cumulative.sum()
    return daily, cumulative


def _signals(panel: pd.DataFrame, row: int, model: FrozenModel, usual) -> dict:
    last = panel.iloc[row]
    symbol = last["symbol"]
    direction = last["run_direction"]
    direction = direction if isinstance(direction, str) else None
    length = int(last["run_length"])
    stock = panel[panel["symbol"] == symbol].reset_index(drop=True)
    regime = volatility_regime(stock).iloc[-1]
    signals = {
        "current_run": {"direction": direction, "length": length},
        "usual_mix": None if usual is None else _mix(usual),
        "usual_mix_basis": (
            f"B0 frozen at the end of development ({model.record['outcome_end']})"
            if model.state_rule == "frozen_long_run"
            else "B0 from the inference window"
        ),
        "market_state": {
            **{name: _finite(last[name]) for name in MARKET_STATE},
            "regime": regime if isinstance(regime, str) else None,
            "basis": "the inference window the forecast read, through its as-of close",
            "units": (
                "ret_5 and ret_20: price returns over 5 and 20 sessions (dividends excluded; "
                "missing across a gap); vol_20 and ewma_vol: the standard deviation of daily "
                "price returns, not annualized"
            ),
            "regime_basis": (
                "the stock's EWMA volatility against its own terciles in that window, not "
                "comparable with the development tables' regimes"
            ),
        },
    }
    if direction in ("increase", "decrease") and length >= 1:
        shown = min(length, STREAK_CAP)
        table = model.reversal.get(symbol, {}).get(direction, {}).get(str(shown))
        if table is not None:
            signals["after_this_streak"] = {
                **table,
                "source": f"development history through {model.record['outcome_end']}",
                "run_length": length,
                "table_length": shown,  # runs longer than 7 sessions read the 7-session table
            }
    return signals


def forecast_record(
    panel: pd.DataFrame,
    symbol: str,
    model: FrozenModel,
    *,
    target_sessions: Sequence[str],
    created_at: pd.Timestamp,
    data: Mapping[str, object],
    data_class: str,
    reconstructed: bool = False,
) -> dict:
    """
    The checked forecast artifact (schema 1.1, or 1.2 `reconstructed` for a missed evening,
    stated first in its limitations) of a frozen model at `symbol`'s last session
    with a close in `panel`, built from the inference window alone: the frozen estimator's
    probabilities, B1's on the same inputs, band and sessions (the comparator), and the state
    the forecast was made in (the run, momentum, volatility, regime). The reversal statistics
    come from development history, frozen with the model. No trap signal is recorded: real
    trap flags wait for the owner's confirmation of the failure margin f (decision A5).
    """

    if data_class != model.data_class:
        raise RegistryRefused(
            "data_class_mismatch", f"a {model.data_class} model, {data_class} data"
        )
    estimate = probabilities(panel, symbol, model)
    last = panel.iloc[estimate.row]
    refusal = daily = cumulative = comparator = None
    usual = estimate.usual
    if not np.isfinite(last["ewma_vol"]):
        refusal = {"reason": "insufficient_history", "detail": "Too little history for volatility."}
        usual = None
    else:
        daily, cumulative = _normalized(estimate)
        b1_daily, b1_cumulative = _normalized(probabilities(panel, symbol, model, BENCHMARK))
        values = (daily, cumulative, b1_daily, b1_cumulative)
        if not all(np.isfinite(v).all() for v in values):
            refusal = {"reason": "estimator_unavailable", "detail": "An input was missing."}
            daily = cumulative = None
        else:
            comparator = {
                "half_life": model.final.half_life,
                "daily": b1_daily.tolist(),
                "cumulative": b1_cumulative.tolist(),
            }
    limitations = list(LIMITATIONS)
    if data_class == "synthetic":
        limitations.insert(0, "Synthetic demo stock: not a real company.")
    if reconstructed:
        limitations.insert(0, RECONSTRUCTED_LIMITATION)
    return artifact.build(
        symbol=symbol,
        data_class=data_class,
        as_of_session=str(pd.Timestamp(last["session"]).date()),
        created_at=stamp(created_at),
        target_sessions=list(target_sessions),
        daily=None if daily is None else daily.tolist(),
        cumulative=None if cumulative is None else cumulative.tolist(),
        band=model.final.config.band.tau,
        cumulative_band=cumulative_band(model.final.config.band).tau,
        estimator=model.estimator,
        model_version=model.version,
        feature_version=model.record["feature_version"],
        evaluation_run_id=model.record["evaluation_run_id"],
        data=data,
        evidence_status=(
            EvidenceStatus.SYNTHETIC_DEMO.value
            if data_class == "synthetic"
            else EvidenceStatus.EXPERIMENTAL_UNVALIDATED.value
        ),
        calibration_status=(
            "development_tested_synthetic" if data_class == "synthetic" else "development_tested"
        ),
        signals=_signals(panel, estimate.row, model, usual),
        limitations=limitations,
        refusal=refusal,
        comparator=comparator,
        reconstructed=reconstructed,
    )
