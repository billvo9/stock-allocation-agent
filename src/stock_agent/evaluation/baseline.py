"""
Rule baselines (equal weight, inverse volatility, momentum) on the approved
T5a engine (D1-D8, owner approval 2026-10-15).

Every baseline runs through `portfolio.engine.simulate`, so `portfolio_step`
is the single source of its costs, returns and drift:

- D1: the target decided at the close of date d trades at the close of
  d + 1 (the label's `entry_lag`) and first earns the return dated d + 2.
- D2: the cost is `transaction_cost_rate` per unit of traded notional on
  every risky asset, from the drifted weights, applied multiplicatively;
  the cash account is the free funding leg. 10 bp is the provisional
  single-stock rate.
- D3: every run builds from 100% cash at the first execution close and pays
  for it; there is no exit charge. The build decision is `build_date` (by
  default the first date); every portfolio and benchmark in a comparison
  shares one (`require_common_schedule`), and rebalance offsets count from
  it (D7).
- D5: the plain-baseline constraint set, w_max = 1 and no turnover limit.
- D7: rebalance decisions are the dates d after the build decision b with
  (d - b) mod k == offset, one registered offset (default 0) shared by every
  compared baseline.
- D4 (a): a cash account earning 0% is added when the inputs have none;
  uninvested cash earns 0% for synthetic engineering only, and group C cash
  returns (D4 (b)) are not approved, so a nonzero cash return is refused.

`total_turnover` is the summed one-way turnover (F2.4) of every execution,
the build included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.agents.equal_weight import equal_weight
from stock_agent.agents.inverse_volatility import inverse_volatility
from stock_agent.agents.momentum import momentum_weights
from stock_agent.evaluation.results import BacktestHistory
from stock_agent.portfolio.backtest_summary import max_drawdown
from stock_agent.portfolio.engine import UNINVESTED_CASH_RETURN, SimulationResult, simulate
from stock_agent.portfolio.schedule import ScheduleSpec, decision_sessions, execution_lag
from stock_agent.portfolio.transition import Constraints, Transition, portfolio_step

CASH_SYMBOL = "CASH"
CASH_RETURN_REFUSAL = "Uninvested cash earns 0% (D4 (a)): group C cash returns are not approved."


class ScheduleMismatchError(ValueError):
    """Compared runs do not share one build date and one rebalance schedule (D3, D7)."""


def prepare_feature_matrix(
    frame: pd.DataFrame,
    symbols: list[str],
    column: str,
) -> pd.DataFrame:
    """Convert a long-format feature column into a date x symbol matrix."""

    filtered = frame[frame["symbol"].isin(symbols)][["date", "symbol", column]]

    matrix = filtered.pivot(
        index="date",
        columns="symbol",
        values=column,
    ).sort_index()

    missing_symbols = set(symbols) - set(matrix.columns)

    if missing_symbols:
        raise ValueError(f"missing required symbols: {sorted(missing_symbols)}")

    matrix = matrix[symbols]

    if matrix.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    return matrix


def drift_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    """
    Update portfolio weights after asset returns occur: the drift identity
    w * (1 + r) / (1 + w . r) (F1.2), computed by `portfolio_step` with no
    trade, so drift has one definition.
    """
    if weights.shape != asset_returns.shape:
        raise ValueError("Weights and asset returns must have the same shape.")

    step = portfolio_step(weights, None, asset_returns, np.zeros_like(weights), 0)

    return np.array(step.w_next)


@dataclass(frozen=True)
class BaselineResult:
    """
    A baseline run. When the result came from the engine, `simulation` is
    its full record (executions, steps, feasibility), `constraints` the
    registered D5 set, `schedule` the registered schedule and `build_date`
    the build decision's date.
    """

    initial_wealth: float
    final_wealth: float
    cumulative_return: float
    max_drawdown: float
    total_turnover: float
    history: BacktestHistory | None = None
    simulation: SimulationResult | None = None
    constraints: Constraints | None = None
    schedule: ScheduleSpec | None = None
    build_date: pd.Timestamp | None = None


def update_peak_and_drawdown(
    wealth: float,
    peak_wealth: float,
) -> tuple[float, float]:
    """
    Update running peak wealth and calculate current drawdown
    (F6.5: 1 - W / peak, a positive fraction).
    """
    if wealth <= 0 or peak_wealth <= 0:
        raise ValueError("Wealth and peak wealth must be positive.")

    new_peak = max(peak_wealth, wealth)
    drawdown = 0.0

    if peak_wealth > wealth:
        drawdown = 1.0 - wealth / peak_wealth

    return (new_peak, drawdown)


def weights_to_array(
    weights: dict[str, float],
    symbols: list[str],
) -> np.ndarray:
    missing_symbols = set(symbols) - set(weights)

    if missing_symbols:
        raise ValueError(f"Missing weights for symbols: {sorted(missing_symbols)}")

    return np.array(
        [weights[symbol] for symbol in symbols],
        dtype=float,
    )


def inverse_volatility_weights(
    volatility_row: pd.Series,
    symbols: list[str],
) -> np.ndarray:
    missing_symbols = set(symbols) - set(volatility_row.index)

    if missing_symbols:
        raise ValueError(f"Missing volatility values for symbols: {sorted(missing_symbols)}")

    volatility_map = {symbol: float(volatility_row[symbol]) for symbol in symbols}

    inverse_volatility_dict = inverse_volatility(volatility_map)

    return weights_to_array(
        inverse_volatility_dict,
        symbols,
    )


def prepare_inverse_volatility_targets(
    frame: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    volatility_matrix = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="volatility_20d",
    )

    target_rows = []

    for _, volatility_row in volatility_matrix.iterrows():
        weights = inverse_volatility_weights(volatility_row, symbols)
        target_rows.append(weights)

    targets = pd.DataFrame(
        data=target_rows,
        index=volatility_matrix.index,
        columns=symbols,
    )

    return targets


@dataclass(frozen=True, eq=False)
class BacktestInputs:
    """
    A backtest in the engine's layout: `returns` (date x asset, cash
    included, row 0 unused), the targets decided on the schedule's dates,
    the cost-rate vector (cash 0), and the plain-baseline constraint set.
    """

    dates: pd.Index
    columns: tuple[str, ...]
    returns: np.ndarray
    targets: dict[int, np.ndarray]
    cost_rate: np.ndarray
    constraints: Constraints
    schedule: ScheduleSpec


def backtest_inputs(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    rebalance_every: int,
    transaction_cost_rate: float,
    rebalance_offset: int = 0,
    cash_symbol: str = CASH_SYMBOL,
    build_date: pd.Timestamp | None = None,
) -> BacktestInputs:
    """
    Validate a date x symbol backtest and lay it out for
    `portfolio.engine.simulate`. The build is decided at `build_date` (the
    first date by default); earlier rows are never traded or earned.
    """

    if rebalance_every <= 0:
        raise ValueError("Rebalance interval must be positive.")

    if transaction_cost_rate < 0:
        raise ValueError("Transaction cost rate cannot be negative.")

    if not returns.index.equals(target_weights.index):
        raise ValueError("Returns and target weights must have matching dates.")

    if list(returns.columns) != list(target_weights.columns):
        raise ValueError("Returns and target weights must have matching symbols.")

    if build_date is None:
        first = 0
    elif build_date in returns.index:
        first = int(returns.index.get_loc(build_date))
    else:
        raise ValueError(f"The build date {build_date} is not a date of these inputs.")

    if len(returns) - first < 3:
        raise ValueError(
            "Backtest requires at least three dates from the build: a decision, its "
            "execution close and one earned return."
        )

    returns = returns.astype(float)
    target_weights = target_weights.astype(float)

    if not np.isfinite(returns.to_numpy()[1:]).all():
        raise ValueError("Returns contain missing values; the engine handles gaps directly.")

    if cash_symbol in returns.columns:
        if not (returns[cash_symbol].iloc[1:] == UNINVESTED_CASH_RETURN).all():
            raise ValueError(CASH_RETURN_REFUSAL)
    else:
        returns = returns.assign(**{cash_symbol: UNINVESTED_CASH_RETURN})
        target_weights = target_weights.assign(**{cash_symbol: 0.0})

    columns = tuple(returns.columns)
    cash = columns.index(cash_symbol)
    schedule = ScheduleSpec(
        every=rebalance_every,
        offset=rebalance_offset,
        lag=execution_lag(),
        first_decision=first,
    )
    weights = target_weights.to_numpy(dtype=float)
    cost_rate = np.full(len(columns), float(transaction_cost_rate))
    cost_rate[cash] = 0.0

    return BacktestInputs(
        dates=returns.index,
        columns=columns,
        returns=returns.to_numpy(dtype=float),
        targets={d: weights[d].copy() for d in decision_sessions(schedule, len(returns))},
        cost_rate=cost_rate,
        constraints=Constraints(cash),
        schedule=schedule,
    )


def baseline_result(
    simulation: SimulationResult,
    dates: pd.Index,
    constraints: Constraints | None = None,
    schedule: ScheduleSpec | None = None,
) -> BaselineResult:
    """A `BaselineResult` from a complete engine run over `dates`."""

    if simulation.status != "complete" or any(value is None for value in simulation.wealth):
        raise ValueError("A baseline needs a complete run with wealth at every date.")

    wealth = pd.Series(
        data=list(simulation.wealth),
        index=dates[list(simulation.sessions)],
        name="wealth",
        dtype=float,
    )

    portfolio_returns = pd.Series(
        data=[step.net_return for step in simulation.steps],
        index=dates[[end for _, end in simulation.spans]],
        name="portfolio_return",
        dtype=float,
    )

    initial_wealth = float(wealth.iloc[0])
    final_wealth = float(wealth.iloc[-1])

    return BaselineResult(
        initial_wealth=initial_wealth,
        final_wealth=final_wealth,
        cumulative_return=final_wealth / initial_wealth - 1.0,
        max_drawdown=max_drawdown(list(wealth)),
        total_turnover=float(sum(step.turnover_one_way for step in simulation.steps)),
        history=BacktestHistory(portfolio_returns=portfolio_returns, wealth=wealth),
        simulation=simulation,
        constraints=constraints,
        schedule=schedule,
        build_date=None if schedule is None else dates[schedule.first_decision],
    )


def require_common_schedule(
    portfolios: dict[str, BaselineResult],
    benchmarks: dict[str, BaselineResult] | None = None,
) -> None:
    """
    D3 and D7: every compared portfolio and benchmark builds at one common
    build date and first execution close, and every portfolio rebalances on
    one registered schedule (the same k and offset). Buy-and-hold benchmarks
    share the build only. Raises `ScheduleMismatchError` otherwise.
    """

    runs = {**portfolios, **(benchmarks or {})}
    if not portfolios:
        raise ScheduleMismatchError("A comparison needs at least one portfolio.")
    missing = sorted(name for name, run in runs.items() if run.schedule is None)
    if missing:
        raise ScheduleMismatchError(f"No engine schedule recorded for {missing}.")

    def first_close(run: BaselineResult) -> pd.Timestamp:
        return run.history.wealth.index[0]

    builds = {(run.build_date, first_close(run)) for run in runs.values()}
    if len(builds) > 1:
        detail = {name: (str(run.build_date), str(first_close(run))) for name, run in runs.items()}
        raise ScheduleMismatchError(f"Runs build on different dates: {detail}")
    phases = {(run.schedule.every, run.schedule.offset) for run in portfolios.values()}
    if len(phases) > 1:
        detail = {
            name: (run.schedule.every, run.schedule.offset) for name, run in portfolios.items()
        }
        raise ScheduleMismatchError(f"Portfolios rebalance on different schedules: {detail}")


def run_rebalanced_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
    rebalance_offset: int = 0,
    *,
    cash_symbol: str = CASH_SYMBOL,
    build_date: pd.Timestamp | None = None,
    transition: Transition = portfolio_step,
) -> BaselineResult:
    """
    Simulate `target_weights` (row d: the target decided at the close of
    date d) on `returns` (row t: each asset's return dated t) through the
    approved engine. See the module docstring for the rules.
    """

    if initial_wealth <= 0:
        raise ValueError("Initial wealth must be positive.")

    inputs = backtest_inputs(
        returns,
        target_weights,
        rebalance_every=rebalance_every,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_offset=rebalance_offset,
        cash_symbol=cash_symbol,
        build_date=build_date,
    )

    simulation = simulate(
        inputs.returns,
        inputs.targets,
        lag=inputs.schedule.lag,
        cost_rate=inputs.cost_rate,
        constraints=inputs.constraints,
        initial_wealth=initial_wealth,
        transition=transition,
    )

    return baseline_result(simulation, inputs.dates, inputs.constraints, inputs.schedule)


def run_inverse_volatility_baseline(
    frame: pd.DataFrame,
    symbols: list[str],
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
    rebalance_offset: int = 0,
    build_date: pd.Timestamp | None = None,
) -> BaselineResult:
    returns = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="daily_return",
    )

    target_weights = prepare_inverse_volatility_targets(
        frame=frame,
        symbols=symbols,
    )

    return run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=initial_wealth,
        rebalance_every=rebalance_every,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_offset=rebalance_offset,
        build_date=build_date,
    )


def prepare_equal_weight_targets(
    returns: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    target_map = equal_weight(symbols)

    target_array = weights_to_array(
        target_map,
        symbols,
    )

    # 1. repeat target_array once for every date in returns
    # 2. build a DataFrame
    # 3. use returns.index as the index
    # 4. use symbols as the columns
    # 5. return the DataFrame
    return pd.DataFrame(
        data=np.tile(target_array, (len(returns), 1)), index=returns.index, columns=symbols
    )


def run_equal_weight_baseline(
    frame: pd.DataFrame,
    symbols: list[str],
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
    rebalance_offset: int = 0,
    build_date: pd.Timestamp | None = None,
) -> BaselineResult:
    returns = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="daily_return",
    )

    target_weights = prepare_equal_weight_targets(
        returns=returns,
        symbols=symbols,
    )

    return run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=initial_wealth,
        rebalance_every=rebalance_every,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_offset=rebalance_offset,
        build_date=build_date,
    )


def add_cash_returns(
    returns: pd.DataFrame,
    cash_symbol: str = "CASH",
    cash_return: float = 0.0,
) -> pd.DataFrame:
    """Add a synthetic cash return column to a return matrix."""

    if cash_symbol in returns.columns:
        raise ValueError(f"{cash_symbol} already exists in returns.")

    result = returns.copy()
    result[cash_symbol] = cash_return

    return result


def prepare_momentum_targets(
    frame: pd.DataFrame,
    symbols: list[str],
    lookback: int = 20,
    cash_symbol: str = "CASH",
) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("Momentum lookback must be positive.")

    if cash_symbol in symbols:
        raise ValueError(f"{cash_symbol} must not be included in market symbols.")

    returns = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="daily_return",
    )

    momentum = (1.0 + returns).rolling(window=lookback).apply(np.prod, raw=True) - 1.0

    momentum = momentum.dropna()

    portfolio_symbols = [*symbols, cash_symbol]

    target_rows = []

    for _, momentum_row in momentum.iterrows():
        scores = {symbol: float(momentum_row[symbol]) for symbol in symbols}

        target_map = momentum_weights(
            scores,
            cash_symbol=cash_symbol,
        )

        target_array = weights_to_array(
            target_map,
            portfolio_symbols,
        )

        target_rows.append(target_array)

    return pd.DataFrame(
        data=target_rows,
        index=momentum.index,
        columns=portfolio_symbols,
    )


def run_momentum_baseline(
    frame: pd.DataFrame,
    symbols: list[str],
    lookback: int = 20,
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
    cash_symbol: str = CASH_SYMBOL,
    cash_return: float = UNINVESTED_CASH_RETURN,
    rebalance_offset: int = 0,
    build_date: pd.Timestamp | None = None,
) -> BaselineResult:
    if cash_return != UNINVESTED_CASH_RETURN:
        raise ValueError(CASH_RETURN_REFUSAL)

    returns = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="daily_return",
    )

    target_weights = prepare_momentum_targets(
        frame=frame,
        symbols=symbols,
        lookback=lookback,
        cash_symbol=cash_symbol,
    )

    # Align stock returns to the dates where momentum targets exist.
    returns = returns.loc[target_weights.index]

    # Add synthetic CASH return so returns and targets have matching columns.
    portfolio_returns = add_cash_returns(
        returns=returns,
        cash_symbol=cash_symbol,
        cash_return=cash_return,
    )

    return run_rebalanced_backtest(
        returns=portfolio_returns,
        target_weights=target_weights,
        initial_wealth=initial_wealth,
        rebalance_every=rebalance_every,
        transaction_cost_rate=transaction_cost_rate,
        rebalance_offset=rebalance_offset,
        cash_symbol=cash_symbol,
        build_date=build_date,
    )
