"""
Run the rule baselines on the approved T5a engine over a seeded synthetic panel.

SYNTHETIC ENGINEERING ONLY. The panel is three random-walk assets standing
for single stocks; nothing here is evidence about any real strategy, and the
baselines' numbers are not a strategy comparison. Uninvested cash earns 0%
(D4 (a), synthetic engineering only; group C pending), and the 10 bp cost is
D2's provisional single-stock rate.

Two outputs:

- the Model Health Center's input (agreed format v1.0, ignored by Git), one
  per baseline: <health-inputs-root>/backtest_summary/<run_id>.json;
- with --attribution, the one-change-at-a-time attribution of each baseline's
  numbers from the pre-T5a engine to the approved one:
      legacy     same-close execution (lag 0), the cash leg charged on two-way
                 turnover and netted additively, a free initial portfolio, and
                 each strategy's own rebalance phase
      lag        + next-close execution (D1)
      cost_base  + costs on risky legs only, multiplicative (D2)
      build      + a charged build from 100% cash (D3)
      schedule   + one registered rebalance offset for every baseline (D7)
  then the approved engine at every rebalance offset (F7.2: the registered
  offset is the result, the others its sensitivity family) and at 0x, 1x and
  2x the cost rate (D2). The legacy
  step reproduces the pre-T5a `run_rebalanced_backtest` exactly; the legacy
  transition below exists only for this attribution.

Usage (from the repository root):
    .venv/bin/python scripts/run_t5a_baselines.py [--attribution]
        [--health-inputs-root reports/health_inputs]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.evaluation.baseline import (
    BaselineResult,
    add_cash_returns,
    backtest_inputs,
    baseline_result,
    prepare_equal_weight_targets,
    prepare_feature_matrix,
    prepare_inverse_volatility_targets,
    prepare_momentum_targets,
    run_equal_weight_baseline,
    run_inverse_volatility_baseline,
    run_momentum_baseline,
)
from stock_agent.portfolio import backtest_summary
from stock_agent.portfolio.engine import simulate
from stock_agent.portfolio.schedule import execution_lag
from stock_agent.portfolio.transition import (
    NO_TARGET,
    PROVISIONAL_RATES,
    SENSITIVITY_MULTIPLIERS,
    SINGLE_STOCK,
    TRADED,
    StepResult,
    Transition,
    portfolio_step,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEALTH_INPUTS_ROOT = PROJECT_ROOT / "reports" / "health_inputs"

SYMBOLS = ("AAA", "BBB", "CCC")
SEED = 20261015
SESSIONS = 300
INITIAL_WEALTH = 100_000.0
REBALANCE_EVERY = 5
REBALANCE_OFFSET = 0  # the one registered offset (D7)
LOOKBACK = 20
COST_RATE = PROVISIONAL_RATES[SINGLE_STOCK]
EVIDENCE_STATUS = "synthetic_engineering_only"
POLICIES = {
    "equal_weight": f"equal_weight_every_{REBALANCE_EVERY}",
    "inverse_volatility": f"inverse_volatility_every_{REBALANCE_EVERY}",
    "momentum": f"momentum_{LOOKBACK}_every_{REBALANCE_EVERY}",
}
STEPS = ("legacy", "lag", "cost_base", "build", "schedule")


def synthetic_feature_frame(seed: int = SEED, sessions: int = SESSIONS) -> pd.DataFrame:
    """
    A long (date, symbol) frame with `daily_return` and a trailing 20-session
    `volatility_20d` (including the date's own return), from independent
    Gaussian log returns; the 19 warm-up dates are dropped.
    """

    rng = np.random.RandomState(seed)  # legacy stream: stable across numpy versions
    volatility = np.array([0.012, 0.018, 0.025])
    drift = np.array([0.0004, 0.0002, 0.0003])
    returns = np.expm1(rng.normal(drift, volatility, size=(sessions, len(SYMBOLS))))
    wide = pd.DataFrame(
        returns, index=pd.bdate_range("2024-01-02", periods=sessions), columns=SYMBOLS
    )
    trailing = wide.rolling(20).std()
    long = wide.stack().rename("daily_return").to_frame()
    long["volatility_20d"] = trailing.stack()
    long = long.dropna().reset_index()
    return long.rename(columns={"level_0": "date", "level_1": "symbol"})


def baseline_frames(frame: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Each baseline's (returns, targets) on the common dates, as `run_*_baseline` builds them."""

    symbols = list(SYMBOLS)
    momentum = prepare_momentum_targets(frame, symbols, lookback=LOOKBACK)
    common = frame[frame["date"] >= momentum.index[0]]
    returns = prepare_feature_matrix(common, symbols, "daily_return")
    return {
        "equal_weight": (returns, prepare_equal_weight_targets(returns, symbols)),
        "inverse_volatility": (returns, prepare_inverse_volatility_targets(common, symbols)),
        "momentum": (add_cash_returns(returns.loc[momentum.index]), momentum),
    }


def run_baselines(frame: pd.DataFrame, rate: float = COST_RATE) -> dict[str, BaselineResult]:
    """The production baselines, run exactly as callers run them."""

    symbols = list(SYMBOLS)
    momentum = prepare_momentum_targets(frame, symbols, lookback=LOOKBACK)
    common = frame[frame["date"] >= momentum.index[0]].copy()
    shared = {
        "initial_wealth": INITIAL_WEALTH,
        "rebalance_every": REBALANCE_EVERY,
        "transaction_cost_rate": rate,
        "rebalance_offset": REBALANCE_OFFSET,
    }
    return {
        "equal_weight": run_equal_weight_baseline(common, symbols, **shared),
        "inverse_volatility": run_inverse_volatility_baseline(common, symbols, **shared),
        "momentum": run_momentum_baseline(frame, symbols, lookback=LOOKBACK, **shared),
    }


# --- the attribution (pre-T5a behaviour, for explanation only) ---


def legacy_transition(w_pre, target, r_next, cost_rate, cash_index) -> StepResult:
    """
    The pre-T5a step, for the attribution only: one rate on the two-way
    turnover of every column (the cash leg charged), netted additively.
    """

    pre = np.asarray(w_pre, dtype=float)
    post = pre if target is None else np.asarray(target, dtype=float)
    two_way = float(np.abs(post - pre).sum())
    cost = float(np.max(cost_rate)) * two_way
    earned = np.where(post > 0, r_next, 0.0)
    gross = float(post @ earned)
    return StepResult(
        w_post_trade=post,
        w_next=post * (1.0 + earned) / (1.0 + gross),
        gross_return=gross,
        cost=cost,
        net_return=gross - cost,
        turnover_one_way=0.5 * two_way,
        traded_notional=float(np.abs(np.delete(post - pre, cash_index)).sum()),
        reason=NO_TARGET if target is None else TRADED,
    )


def free_first_trade(transition: Transition) -> Transition:
    """`transition` with the first call's trade free (the pre-T5a initial portfolio)."""

    calls = []

    def step(w_pre, target, r_next, cost_rate, cash_index):
        rates = np.zeros_like(cost_rate) if not calls else cost_rate
        calls.append(1)
        return transition(w_pre, target, r_next, rates, cash_index)

    return step


# Each strategy's own pre-T5a rebalance phase: equal weight decided on
# dates k, 2k, ..., the others on k - 1, 2k - 1, ... (spec 7 (a)).
LEGACY_OFFSETS = {
    "equal_weight": 0,
    "inverse_volatility": REBALANCE_EVERY - 1,
    "momentum": REBALANCE_EVERY - 1,
}


def variant(
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    lag: int,
    transition: Transition,
    offset: int,
    rate: float = COST_RATE,
) -> BaselineResult:
    inputs = backtest_inputs(
        returns,
        targets,
        rebalance_every=REBALANCE_EVERY,
        transaction_cost_rate=rate,
        rebalance_offset=offset,
    )
    simulation = simulate(
        inputs.returns,
        inputs.targets,
        lag=lag,
        cost_rate=inputs.cost_rate,
        constraints=inputs.constraints,
        initial_wealth=INITIAL_WEALTH,
        transition=transition,
    )
    return baseline_result(simulation, inputs.dates, inputs.constraints)


def attribution(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Each baseline at each step of STEPS, then at each cost multiplier."""

    lag = execution_lag()
    rows = []
    for name, (returns, targets) in baseline_frames(frame).items():
        legacy_offset = LEGACY_OFFSETS[name]
        settings = {
            "legacy": (0, free_first_trade(legacy_transition), legacy_offset),
            "lag": (lag, free_first_trade(legacy_transition), legacy_offset),
            "cost_base": (lag, free_first_trade(portfolio_step), legacy_offset),
            "build": (lag, portfolio_step, legacy_offset),
            "schedule": (lag, portfolio_step, REBALANCE_OFFSET),
        }
        previous = None
        for step in STEPS:
            step_lag, transition, offset = settings[step]
            result = variant(returns, targets, lag=step_lag, transition=transition, offset=offset)
            rows.append(_row(name, step, result, previous))
            previous = result
        for offset in range(REBALANCE_EVERY):  # F7.2: one variant, a k-member family
            result = variant(returns, targets, lag=lag, transition=portfolio_step, offset=offset)
            rows.append(_row(name, f"offset_{offset}", result, None))
        for multiplier in SENSITIVITY_MULTIPLIERS:
            result = variant(
                returns,
                targets,
                lag=lag,
                transition=portfolio_step,
                offset=REBALANCE_OFFSET,
                rate=COST_RATE * multiplier,
            )
            rows.append(_row(name, f"cost_{multiplier:g}x", result, None))
    return rows


def _row(name, step, result: BaselineResult, previous: BaselineResult | None) -> dict:
    growth = result.final_wealth / result.initial_wealth
    delta = None
    if previous is not None:
        delta = 1e4 * (growth - previous.final_wealth / previous.initial_wealth)
    return {
        "baseline": name,
        "step": step,
        "first_valued_date": str(result.history.wealth.index[0].date()),
        "terminal_wealth_ratio": growth,
        "delta_bp": delta,
        "total_turnover_one_way": result.total_turnover,
        "costs_bp": 1e4 * sum(step.cost for step in result.simulation.steps),
    }


# --- health inputs ---


def git_code_version(root: Path = PROJECT_ROOT) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_sha": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}


def write_summaries(
    results: dict[str, BaselineResult],
    root: Path,
    *,
    generated_at: datetime,
    code: dict[str, object],
) -> list[Path]:
    paths = []
    for name, result in results.items():
        policy = POLICIES[name]
        record = backtest_summary.summary_record(
            result.simulation,
            constraints=result.constraints,
            run_id=f"{generated_at:%Y%m%dT%H%M%S%fZ}-{policy}",
            policy=policy,
            evidence_status=EVIDENCE_STATUS,
            synthetic=True,
            generated_at=generated_at,
            code_sha=code.get("git_sha"),
            assumptions={
                "data": f"seeded synthetic panel (seed {SEED}, {SESSIONS} sessions, {SYMBOLS})",
                "cost_rate_bps": {symbol: COST_RATE * 1e4 for symbol in SYMBOLS},
                "cost_rate_basis": "D2 provisional single-stock all-in rate",
                "cost_multiplier": 1.0,
                "execution_lag": execution_lag(),
                "rebalance_every": REBALANCE_EVERY,
                "rebalance_offset": REBALANCE_OFFSET,
                "build": "from 100% cash at the first execution close, charged (D3)",
                "constraint_set": "plain baselines: w_max 1.0, no turnover limit (D5)",
                "git_dirty": code.get("git_dirty"),
            },
        )
        paths.append(backtest_summary.write_summary(record, root))
    return paths


def main(
    argv: list[str] | None = None,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_version: Callable[[], dict] = git_code_version,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--health-inputs-root", type=Path, default=DEFAULT_HEALTH_INPUTS_ROOT)
    parser.add_argument("--attribution", action="store_true")
    args = parser.parse_args(argv)

    frame = synthetic_feature_frame()
    results = run_baselines(frame)
    paths = write_summaries(
        results, args.health_inputs_root, generated_at=clock(), code=code_version()
    )
    print("Synthetic engineering only: not a strategy comparison.")
    for path in paths:
        print(f"wrote {path}")
    if args.attribution:
        print(pd.DataFrame(attribution(frame)).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
