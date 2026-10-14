"""
The Portfolio Builder's pure logic (`stock_agent.app.mix`): entered and
equal weights, concentration, the discrete what-if weight alternatives and
their text. The module restates weights only: it computes no growth, return
or decline (the mix history awaits approval), which is tested too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.app import mix
from stock_agent.app.history import RECORD_COLUMNS, FundHistory

MIX = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "app" / "mix.py"
DATES = ("2021-12-30", "2021-12-31", "2022-01-03")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def fund(fund_id: str, role: str = "equity") -> FundHistory:
    """A small hand-built history with a close on every session."""

    rows = [(utc(DATES[0]), True, np.nan, 0, "unavailable", "no_prior_close")]
    rows += [(utc(text), True, 0.01, 1, "ok", None) for text in DATES[1:]]
    records = pd.DataFrame(rows, columns=list(RECORD_COLUMNS))
    records["date"] = pd.DatetimeIndex(records["date"])
    return FundHistory(
        fund_id=fund_id,
        name=f"Fund {fund_id}",
        role=role,
        records=records,
        calendar=pd.DatetimeIndex([utc(text) for text in DATES]),
        source_label="test source",
        retrieved_label="test retrieval",
        vintage_note="test vintage",
        limitations=("synthetic_data",),
    )


A, B, C = fund("a"), fund("b", "bonds"), fund("c", "cash proxy")


# --- weights ---


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, [100]),
        (2, [50, 50]),
        (3, [35, 35, 30]),
        (4, [25, 25, 25, 25]),
        (6, [20, 20, 15, 15, 15, 15]),
    ],
)
def test_starting_points_are_even_steps_of_five_that_add_to_100(count, expected):
    ids = [f"f{i}" for i in range(count)]
    points = mix.starting_points(ids)
    assert list(points.values()) == expected
    assert sum(points.values()) == mix.TOTAL_POINTS
    assert all(value % mix.WEIGHT_STEP == 0 for value in points.values())
    assert mix.starting_points([]) == {}


def test_entered_points_become_fractions_and_a_fund_at_zero_is_left_out():
    assert mix.from_points({"a": 60, "b": 40, "c": 0}) == {"a": 0.6, "b": 0.4}
    assert mix.points_gap({"a": 60, "b": 35}) == 5
    assert mix.points_gap({"a": 60, "b": 45}) == -5
    with pytest.raises(ValueError, match="add to 100"):
        mix.from_points({"a": 60, "b": 35})
    with pytest.raises(ValueError, match="negative"):
        mix.from_points({"a": 110, "b": -10})


def test_equal_weights_split_evenly():
    weights = mix.equal_weights(["a", "b", "c"])
    assert weights == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"a": 0.6, "b": 0.3}, "add to 1"),
        ({"a": 0.0, "b": 0.0}, "positive weight"),
        ({"a": 1.2, "b": -0.2}, "negative"),
    ],
)
def test_weights_that_do_not_form_a_mix_are_refused(weights, message):
    with pytest.raises(ValueError, match=message):
        mix.concentration((A, B), weights)


# --- concentration ---


def test_concentration_restates_the_weights_by_fund_and_role():
    spread = mix.concentration((A, B, C), {"a": 0.5, "b": 0.3, "c": 0.2})
    assert spread.holdings == 3
    assert (spread.largest_fund, spread.largest_weight) == ("Fund a", 0.5)
    assert spread.by_role == pytest.approx({"equity": 0.5, "bonds": 0.3, "cash proxy": 0.2})
    assert list(spread.by_role) == ["equity", "bonds", "cash proxy"]  # fixed order, not ranked


def test_a_tie_goes_to_the_first_fund_listed_never_a_ranking():
    assert mix.concentration((A, B), {"a": 0.5, "b": 0.5}).largest_fund == "Fund a"
    assert mix.concentration((A, B), {"b": 0.5, "a": 0.5}).largest_fund == "Fund b"


def test_funds_at_zero_are_not_held_and_unknown_funds_are_refused():
    spread = mix.concentration((A, B), {"a": 1.0, "b": 0.0})
    assert spread.holdings == 1 and spread.by_role == {"equity": 1.0}
    with pytest.raises(ValueError, match="not given"):
        mix.concentration((A,), {"a": 0.5, "z": 0.5})


# --- what-if alternatives ---


def test_equal_weights_are_offered_only_when_the_weights_differ():
    assert mix.EQUAL in [v.key for v in mix.variants((A, B), {"a": 0.6, "b": 0.4})]
    assert mix.EQUAL not in [v.key for v in mix.variants((A, B), {"a": 0.5, "b": 0.5})]


def test_ten_points_move_between_roles_in_proportion_to_the_weights():
    found = {v.key: v for v in mix.variants((A, B, C), {"a": 0.6, "b": 0.3, "c": 0.1})}
    assert found[mix.TOWARD_DEFENSIVE].weights == pytest.approx({"a": 0.5, "b": 0.375, "c": 0.125})
    assert found[mix.TOWARD_EQUITY].weights == pytest.approx({"a": 0.7, "b": 0.225, "c": 0.075})
    for variant in found.values():
        assert sum(variant.weights.values()) == pytest.approx(1.0)
        assert min(variant.weights.values()) >= 0


def test_no_role_shift_without_both_roles_or_ten_points_to_give():
    assert [v.key for v in mix.variants((A,), {"a": 1.0})] == []
    keys = [v.key for v in mix.variants((A, B), {"a": 0.95, "b": 0.05})]
    assert mix.TOWARD_DEFENSIVE in keys and mix.TOWARD_EQUITY not in keys


def test_variant_labels_never_rank_or_advise():
    text = " ".join(v.label for v in mix.variants((A, B, C), {"a": 0.6, "b": 0.3, "c": 0.1}))
    for word in (" buy", " sell", "recommend", "better", "best", "optimal", "outperform"):
        assert word not in text.lower(), word


# --- text ---


def test_weights_text_lists_funds_in_listing_order():
    assert mix.weights_text((A, B), {"b": 0.4, "a": 0.6}) == "Fund a 60.0% · Fund b 40.0%"


def test_a_concentration_row_states_weights_largest_and_every_role():
    row = mix.concentration_row("This mix", (A, B, C), {"a": 0.6, "b": 0.4})
    assert row == {
        "Mix": "This mix",
        "Weights": "Fund a 60.0% · Fund b 40.0%",
        "Funds held": "2",
        "Largest weight": "Fund a, 60.0%",
        "Equity": "60.0%",
        "Bonds": "40.0%",
        "Cash proxy": "0.0%",
    }


# --- no performance until the mix formula is approved ---


def test_the_builder_logic_computes_no_growth_return_or_decline():
    tree = ast.parse(MIX.read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(module.startswith("stock_agent.portfolio") for module in imported)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for word in ("growth", "drawdown", "return", "calendar", "value"):
        assert not [name for name in names if word in name], word
