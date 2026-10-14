"""
Pure logic for the Portfolio Builder: the user's weights and their checks,
how the weights are spread (concentration), and discrete what-if weight
alternatives. No Streamlit, no Plotly, and no financial formula: every value
here is a restatement or a rearrangement of the weights themselves.

How a mix of funds would have grown is not computed. A fixed mix held
without rebalancing is a reference portfolio, whose formula is awaiting the
owner's approval; until then the page says so plainly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from stock_agent.app import labels, view
from stock_agent.app.history import FundHistory

TOTAL_POINTS = 100
WEIGHT_STEP = 5  # percentage points per step of an entered weight
SHIFT = 0.10  # what-if alternatives move ten percentage points between roles
ROLES = ("equity", "bonds", "cash proxy")
DEFENSIVE_ROLES = ("bonds", "cash proxy")


# --- weights ---


def starting_points(fund_ids: Sequence[str]) -> dict[str, int]:
    """
    Whole percentage points in steps of WEIGHT_STEP that add to 100, as even
    as the steps allow, the first funds listed taking any remainder: the
    starting values of entered weights (35, 35, 30 for three funds).
    """

    if not fund_ids:
        return {}
    steps = TOTAL_POINTS // WEIGHT_STEP
    base, extra = divmod(steps, len(fund_ids))
    return {
        fund_id: (base + (1 if index < extra else 0)) * WEIGHT_STEP
        for index, fund_id in enumerate(fund_ids)
    }


def equal_weights(fund_ids: Sequence[str]) -> dict[str, float]:
    return {fund_id: 1.0 / len(fund_ids) for fund_id in fund_ids}


def points_gap(points: Mapping[str, int]) -> int:
    """How many points the entered weights are short of 100 (negative when over)."""

    return TOTAL_POINTS - sum(points.values())


def from_points(points: Mapping[str, int]) -> dict[str, float]:
    """Entered whole-point weights as fractions, keeping only funds above zero."""

    if any(value < 0 for value in points.values()):
        raise ValueError("weights cannot be negative")
    if points_gap(points) != 0:
        raise ValueError("entered weights must add to 100 points")
    return {fund_id: value / TOTAL_POINTS for fund_id, value in points.items() if value > 0}


def held(weights: Mapping[str, float]) -> dict[str, float]:
    """The funds with a positive weight; refuses weights that do not form a mix."""

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("weights cannot be negative")
    positive = {fund_id: float(weight) for fund_id, weight in weights.items() if weight > 0}
    if not positive:
        raise ValueError("a mix needs at least one fund with a positive weight")
    if not np.isclose(sum(positive.values()), 1.0, rtol=0, atol=1e-9):
        raise ValueError("weights must add to 1")
    return positive


# --- concentration: a restatement of the weights ---


@dataclass(frozen=True)
class Concentration:
    holdings: int
    largest_fund: str
    largest_weight: float
    by_role: dict[str, float]  # each role's total weight, in the fixed role order


def concentration(histories: Sequence[FundHistory], weights: Mapping[str, float]) -> Concentration:
    """How the weights are spread: across funds and across their roles."""

    positive = held(weights)
    by_id = {history.fund_id: history for history in histories}
    missing = [fund_id for fund_id in positive if fund_id not in by_id]
    if missing:
        raise ValueError(f"weights name funds that were not given: {missing}")
    order = list(positive)
    largest = max(order, key=lambda fund_id: (positive[fund_id], -order.index(fund_id)))
    roles: dict[str, float] = {}
    for role in ROLES:
        total = sum(weight for fund_id, weight in positive.items() if by_id[fund_id].role == role)
        if total > 0:
            roles[role] = total
    return Concentration(len(positive), by_id[largest].name, positive[largest], roles)


# --- what-if alternatives: other weights for the same funds ---

EQUAL = "equal"
TOWARD_DEFENSIVE = "toward_defensive"
TOWARD_EQUITY = "toward_equity"


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    weights: dict[str, float]


def _shift(
    weights: dict[str, float], source: Sequence[str], target: Sequence[str]
) -> dict[str, float]:
    """Move SHIFT from `source` funds to `target` funds, in proportion to their weights."""

    source_total = sum(weights[f] for f in source)
    target_total = sum(weights[f] for f in target)
    moved = dict(weights)
    for fund_id in source:
        moved[fund_id] -= SHIFT * weights[fund_id] / source_total
    for fund_id in target:
        moved[fund_id] += SHIFT * weights[fund_id] / target_total
    return moved


def variants(histories: Sequence[FundHistory], weights: Mapping[str, float]) -> tuple[Variant, ...]:
    """
    The discrete alternatives offered for the current weights, each over the
    same funds: equal weights (unless already equal), and ten points moved
    from equity funds to bond and cash funds or back, where the mix holds
    both and the funds giving up weight hold at least ten points.
    """

    positive = held(weights)
    roles = {history.fund_id: history.role for history in histories}
    equity = [fund_id for fund_id in positive if roles[fund_id] == "equity"]
    defensive = [fund_id for fund_id in positive if roles[fund_id] in DEFENSIVE_ROLES]
    found = []
    equal = equal_weights(list(positive))
    if len(positive) > 1 and not all(np.isclose(positive[f], equal[f]) for f in positive):
        found.append(Variant(EQUAL, "Equal weights across the same funds", equal))
    if equity and defensive:
        if sum(positive[f] for f in equity) >= SHIFT - 1e-12:
            found.append(
                Variant(
                    TOWARD_DEFENSIVE,
                    "Ten points moved from equity funds to bond and cash funds",
                    _shift(positive, equity, defensive),
                )
            )
        if sum(positive[f] for f in defensive) >= SHIFT - 1e-12:
            found.append(
                Variant(
                    TOWARD_EQUITY,
                    "Ten points moved from bond and cash funds to equity funds",
                    _shift(positive, defensive, equity),
                )
            )
    return tuple(found)


# --- text for the page ---


def weights_text(histories: Sequence[FundHistory], weights: Mapping[str, float]) -> str:
    """'Synthetic equity fund 60.0% · Synthetic bond fund 40.0%', in listing order."""

    names = {history.fund_id: history.name for history in histories}
    order = [history.fund_id for history in histories if weights.get(history.fund_id, 0) > 0]
    return " · ".join(f"{names[f]} {view.percent(weights[f])}" for f in order)


def concentration_row(
    name: str, histories: Sequence[FundHistory], weights: Mapping[str, float]
) -> dict[str, str]:
    """One column of the what-if comparison, as text: weights, largest weight, roles."""

    spread = concentration(histories, weights)
    row = {
        "Mix": name,
        "Weights": weights_text(histories, weights),
        "Funds held": str(spread.holdings),
        "Largest weight": f"{spread.largest_fund}, {view.percent(spread.largest_weight)}",
    }
    for role in ROLES:
        row[labels.ROLE_TEXT[role]] = view.percent(spread.by_role.get(role, 0.0))
    return row
