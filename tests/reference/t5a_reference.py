"""
Independent reference oracle for the T5a engine.

A deliberately simple, loop-based ledger of share counts and cash dollars
implementing the RECOMMENDED options of `docs/research/t5a_open_decisions.md`
(D1-D8, v2, 2026-10-14). Those rules are PENDING the owner's approval; this
module is test-only and approves nothing.

Independence: it imports nothing from `stock_agent` and none of the
engine's weight algebra (no drift identity, no (1 - C)(1 + w.r) product).
It values holdings as shares x price plus cash, trades by moving dollars
between the cash account and share counts, and pays fees out of cash. The
weights-based engine and this ledger agreeing within 1e-12 on random paths
is the conservation test.

Recommended rules implemented:

- D1: a decision at session d executes at the close of d + lag. Holdings
  bought there first earn the price change to the close of d + lag + 1.
- D2: fee = sum over non-cash i of rate_i * |target_i * W- - value_i|,
  dollars measured on pre-trade wealth W-. Then W+ = W- - fee and each
  holding becomes target_i * W+ (the O(c^2) fixed point is ignored, as F2.3
  states). The cash leg is free.
- D3: a run starts as 100% cash at its first decision's execution close and
  pays D2 to build. No exit charge.
- D4 (a): the cash account earns the cash column of `returns` (zero in the
  recommended interim).
- D5: long only and summing to 1, else `InvalidTarget`; a per-asset cap
  whose excess goes to cash; then a turnover limit trading part way.
- D6: a rebalance waits for the next session at which every involved asset
  (held, or in the target) has a close. Share counts are held through gaps;
  wealth is unavailable (None) at a close where a held asset has none. A
  held asset with a close but no available return (a distribution inside
  the gap) stops the run.
- D7: one common build decision; rebalances where
  (d - first) mod every == offset.
- D8 (a): reward = log of the wealth ratio between consecutive valued closes.

Input layout (shared with the engine): `returns[t][i]` is asset i's return
from its previous close to close t, with the cash account at `cash_index`;
row 0 is unused; NaN where the asset has no close at t, and after a gap the
multi-session return from the last close.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

SUM_TOLERANCE = 1e-9

UNEXECUTABLE_MISSING_CLOSE = "unexecutable_missing_close"
NO_EXECUTION_SESSION = "no_execution_session"
DISTRIBUTION_IN_GAP = "distribution_in_gap"
COMPLETE = "complete"
STOPPED = "stopped"


class InvalidTarget(ValueError):
    """A target that is not long-only or does not sum to 1 (D5: never repaired)."""


class ScenarioNotCovered(Exception):
    """The decision packet does not specify this case, so the oracle refuses it."""


@dataclass
class Fill:
    """One executed trade at a close, in ledger terms."""

    decision: int
    session: int
    wealth_before: float
    fee: float
    wealth_after: float
    drifted: list[float]
    executed: list[float]

    @property
    def cost(self) -> float:
        return self.fee / self.wealth_before

    @property
    def turnover_one_way(self) -> float:
        return 0.5 * sum(abs(a - b) for a, b in zip(self.executed, self.drifted, strict=True))

    def traded_notional(self, cash_index: int) -> float:
        return sum(
            abs(self.executed[i] - self.drifted[i])
            for i in range(len(self.executed))
            if i != cash_index
        )


@dataclass
class Event:
    """Where a decision executed, or why it did not."""

    decision: int
    planned: int
    executed: int | None
    reason: str | None


@dataclass
class OracleRun:
    sessions: list[int] = field(default_factory=list)
    wealth: list[float | None] = field(default_factory=list)
    weights: list[list[float] | None] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    status: str = COMPLETE
    reason: str | None = None

    def wealth_at(self, session: int) -> float | None:
        return self.wealth[self.sessions.index(session)]


class Ledger:
    """Share counts per non-cash asset plus a cash balance in dollars."""

    def __init__(self, n_assets: int, cash_index: int, cash: float) -> None:
        self.shares = [0.0] * n_assets
        self.cash = float(cash)
        self.cash_index = cash_index

    def held(self) -> list[int]:
        return [i for i, count in enumerate(self.shares) if count != 0.0]

    def value(self, prices: Sequence[float | None]) -> float | None:
        total = self.cash
        for i in self.held():
            if prices[i] is None:
                return None
            total += self.shares[i] * prices[i]
        return total

    def weights(self, prices: Sequence[float | None]) -> list[float] | None:
        total = self.value(prices)
        if total is None:
            return None
        weights = []
        for i in range(len(self.shares)):
            if i == self.cash_index:
                weights.append(self.cash / total)
            elif self.shares[i] == 0.0:
                weights.append(0.0)
            else:
                weights.append(self.shares[i] * prices[i] / total)
        return weights

    def trade_to(
        self, target: Sequence[float], prices: Sequence[float | None], rates: Sequence[float]
    ) -> tuple[float, float, float]:
        """Rebalance to `target` at `prices`; returns (W-, fee, W+)."""

        before = self.value(prices)
        values = [
            self.shares[i] * prices[i] if self.shares[i] != 0.0 else 0.0
            for i in range(len(self.shares))
        ]
        fee = 0.0
        for i in range(len(target)):
            if i != self.cash_index:
                fee += rates[i] * abs(target[i] * before - values[i])
        after = before - fee
        for i in range(len(target)):
            if i == self.cash_index:
                continue
            if target[i] == 0.0:
                self.cash += values[i]
                self.shares[i] = 0.0
            else:
                new_value = target[i] * after
                self.cash -= new_value - values[i]
                self.shares[i] = new_value / prices[i]
        self.cash -= fee
        if abs(self.cash - target[self.cash_index] * after) > 1e-9 * max(1.0, abs(after)):
            raise AssertionError("ledger cash does not match the target cash weight")
        return before, fee, after


def feasible(
    requested: Sequence[float],
    drifted: Sequence[float],
    cash_index: int,
    *,
    w_max: float = 1.0,
    tau_max: float | None = None,
) -> list[float]:
    """D5: validate, cap with the excess to cash, then apply the turnover limit."""

    if len(requested) != len(drifted):
        raise InvalidTarget("target and drifted weights differ in length")
    for weight in requested:
        if not math.isfinite(weight) or weight < 0:
            raise InvalidTarget(f"weight {weight!r} is not a finite non-negative number")
    if abs(sum(requested) - 1.0) > SUM_TOLERANCE:
        raise InvalidTarget(f"weights sum to {sum(requested)!r}, not 1")
    capped = [float(weight) for weight in requested]
    for i in range(len(capped)):
        if i != cash_index and capped[i] > w_max:
            capped[cash_index] += capped[i] - w_max
            capped[i] = w_max
    if tau_max is None:
        return capped
    turnover = 0.5 * sum(abs(capped[i] - drifted[i]) for i in range(len(capped)))
    if turnover <= tau_max:
        return capped
    alpha = tau_max / turnover
    return [drifted[i] + alpha * (capped[i] - drifted[i]) for i in range(len(capped))]


def schedule_decisions(first: int, every: int, offset: int, n_sessions: int) -> list[int]:
    """D7: the build decision, then (d - first) mod every == offset for d > first."""

    decisions = [first]
    for d in range(first + 1, n_sessions):
        if (d - first) % every == offset:
            decisions.append(d)
    return decisions


def decide(
    policy: Callable[[list[list[float]]], Sequence[float] | None],
    returns: Sequence[Sequence[float]],
    decisions: Sequence[int],
) -> dict[int, list[float] | None]:
    """The honest caller: the policy at d sees a copy of rows 0..d only."""

    targets = {}
    for d in decisions:
        history = [list(map(float, row)) for row in returns[: d + 1]]
        target = policy(history)
        targets[d] = None if target is None else [float(w) for w in target]
    return targets


def _rows(table: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in table]


def _closes(
    returns: list[list[float]], has_close: Sequence[Sequence[bool]] | None
) -> list[list[bool]]:
    if has_close is None:
        closes = [[math.isfinite(value) for value in row] for row in returns]
        closes[0] = [True] * len(returns[0])
        return closes
    closes = [[bool(value) for value in row] for row in has_close]
    for t, row in enumerate(returns):
        for i, value in enumerate(row):
            if t > 0 and math.isfinite(value) and not closes[t][i]:
                raise ValueError(f"asset {i} has a return but no close at session {t}")
    return closes


def _price_table(
    returns: list[list[float]], closes: list[list[bool]], cash_index: int
) -> tuple[list[list[float | None]], list[list[bool]]]:
    """
    Close prices rebuilt by chaining returns from each asset's previous close
    (1.0 at its first close). `broken[t][i]`: closed at t but its return is
    unavailable, so a holding cannot be valued through t.
    """

    n = len(returns[0])
    prices: list[list[float | None]] = [[None] * n for _ in returns]
    broken = [[False] * n for _ in returns]
    last: list[float | None] = [None] * n
    for t, row in enumerate(returns):
        for i in range(n):
            if i == cash_index or not closes[t][i]:
                continue
            if last[i] is None:
                price = 1.0
            elif math.isfinite(row[i]):
                price = last[i] * (1.0 + row[i])
            else:
                broken[t][i] = True
                price = 1.0
            prices[t][i] = price
            last[i] = price
    return prices, broken


def run(
    returns: Sequence[Sequence[float]],
    targets: Mapping[int, Sequence[float] | None],
    *,
    lag: int,
    cost_rate: Sequence[float],
    cash_index: int,
    initial_wealth: float = 1.0,
    has_close: Sequence[Sequence[bool]] | None = None,
    w_max: float = 1.0,
    tau_max: float | None = None,
) -> OracleRun:
    """Run the ledger over every session from the build's execution close."""

    table = _rows(returns)
    rates = [float(rate) for rate in cost_rate]
    if rates[cash_index] != 0.0:
        raise ValueError("the cash account is the funding leg: its cost rate must be 0")
    closes = _closes(table, has_close)
    prices, broken = _price_table(table, closes, cash_index)
    last_session = len(table) - 1
    decisions = sorted(targets)
    if not decisions:
        raise ValueError("no decisions")
    start = decisions[0] + lag
    if start >= last_session:
        raise ValueError("the build cannot execute inside the data")
    planned = {d + lag: d for d in decisions}
    ledger = Ledger(len(table[0]), cash_index, initial_wealth)
    result = OracleRun()
    waiting: tuple[int, int] | None = None  # (decision, planned execution)

    for t in range(start, last_session + 1):
        if t > start:
            cash_return = table[t][cash_index]
            if not math.isfinite(cash_return):
                raise ValueError(f"the cash return at session {t} is not finite")
            ledger.cash *= 1.0 + cash_return
        if any(broken[t][i] for i in ledger.held()):
            result.status, result.reason = STOPPED, DISTRIBUTION_IN_GAP
            break
        result.sessions.append(t)
        result.wealth.append(ledger.value(prices[t]))
        result.weights.append(ledger.weights(prices[t]))

        if t in planned:
            if waiting is not None:
                raise ScenarioNotCovered("a new decision is due while an earlier one waits (D6)")
            waiting = (planned[t], t)
        if waiting is None:
            continue
        decision, due = waiting
        target = targets[decision]
        if target is None:
            result.events.append(Event(decision, due, t, None))
            waiting = None
            continue
        if t == last_session:
            result.events.append(Event(decision, due, None, NO_EXECUTION_SESSION))
            waiting = None
            continue
        involved = [
            i
            for i in range(len(target))
            if i != cash_index and (ledger.shares[i] != 0.0 or target[i] > 0.0)
        ]
        if any(prices[t][i] is None for i in involved):
            continue  # deferred: drifted holdings kept, nothing filled
        drifted = ledger.weights(prices[t])
        executed = feasible(target, drifted, cash_index, w_max=w_max, tau_max=tau_max)
        before, fee, after = ledger.trade_to(executed, prices[t], rates)
        result.fills.append(Fill(decision, t, before, fee, after, drifted, executed))
        reason = None if t == due else UNEXECUTABLE_MISSING_CLOSE
        result.events.append(Event(decision, due, t, reason))
        waiting = None
    if result.status == COMPLETE:
        for due, decision in sorted(planned.items()):
            if due > last_session:
                result.events.append(Event(decision, due, None, NO_EXECUTION_SESSION))
    return result


def log_rewards(result: OracleRun) -> list[tuple[int, float]]:
    """
    D8 (a): log(W_t / W_prev) at each valued close after the start; across a
    gap the span's reward lands at the first close valued again.
    """

    rewards = []
    previous = None
    for session, wealth in zip(result.sessions, result.wealth, strict=True):
        if wealth is None:
            continue
        if previous is not None:
            rewards.append((session, math.log(wealth / previous)))
        previous = wealth
    return rewards


def ledger_step(
    w_pre: Sequence[float],
    target: Sequence[float] | None,
    r_next: Sequence[float],
    cost_rate: Sequence[float],
    cash_index: int,
) -> dict[str, object]:
    """
    One execution close and the session after it, on a ledger worth 1 at
    unit prices: the shares-and-cash counterpart of `portfolio_step`.
    """

    n = len(w_pre)
    rates = [float(rate) for rate in cost_rate]
    if rates[cash_index] != 0.0:
        raise ValueError("the cash account is the funding leg: its cost rate must be 0")
    unit = [None if i == cash_index else 1.0 for i in range(n)]
    ledger = Ledger(n, cash_index, w_pre[cash_index])
    for i in range(n):
        if i != cash_index:
            ledger.shares[i] = float(w_pre[i])
    before = ledger.value(unit)
    fee = 0.0
    after = before
    if target is not None:
        before, fee, after = ledger.trade_to([float(w) for w in target], unit, rates)
    w_post_trade = ledger.weights(unit)
    moved = []
    for i in range(n):
        if i == cash_index:
            moved.append(None)
            continue
        if ledger.shares[i] != 0.0 and not math.isfinite(r_next[i]):
            raise ValueError(f"held asset {i} has no finite return")
        moved.append(1.0 + float(r_next[i]) if math.isfinite(r_next[i]) else None)
    ledger.cash *= 1.0 + float(r_next[cash_index])
    wealth_next = ledger.value(moved)
    delta = [w_post_trade[i] - float(w_pre[i]) for i in range(n)]
    return {
        "w_post_trade": w_post_trade,
        "w_next": ledger.weights(moved),
        "gross_return": wealth_next / after - 1.0,
        "cost": fee / before,
        "net_return": wealth_next / before - 1.0,
        "turnover_one_way": 0.5 * sum(abs(d) for d in delta),
        "traded_notional": sum(abs(delta[i]) for i in range(n) if i != cash_index),
        "reward": math.log(wealth_next / before),
    }
