from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.evaluation.baseline import (
    BaselineResult,
)
from stock_agent.evaluation.reporting import (
    build_drawdown_frame,
    build_normalized_wealth_frame,
)
from stock_agent.evaluation.results import (
    BacktestHistory,
)


def _make_result(
    wealth_values: list[float],
    dates: list[str],
) -> BaselineResult:
    index = pd.to_datetime(dates)

    wealth = pd.Series(
        data=wealth_values,
        index=index,
        name="wealth",
        dtype=float,
    )

    portfolio_returns = wealth.pct_change().iloc[1:].rename("portfolio_return")

    drawdown = wealth / wealth.cummax() - 1.0

    return BaselineResult(
        initial_wealth=float(wealth.iloc[0]),
        final_wealth=float(wealth.iloc[-1]),
        cumulative_return=(float(wealth.iloc[-1] / wealth.iloc[0] - 1.0)),
        max_drawdown=abs(float(drawdown.min())),
        total_turnover=0.0,
        history=BacktestHistory(
            portfolio_returns=(portfolio_returns),
            wealth=wealth,
        ),
    )


def test_normalized_wealth_starts_at_100():
    results = {
        "Strategy A": _make_result(
            wealth_values=[
                100_000.0,
                110_000.0,
                120_000.0,
            ],
            dates=[
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
        ),
        "Strategy B": _make_result(
            wealth_values=[
                50_000.0,
                52_000.0,
                55_000.0,
            ],
            dates=[
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
        ),
    }

    frame = build_normalized_wealth_frame(results)

    assert (frame.iloc[0] == 100.0).all()


def test_normalized_wealth_matches_known_path():
    results = {
        "Strategy": _make_result(
            wealth_values=[
                100_000.0,
                110_000.0,
                121_000.0,
            ],
            dates=[
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
        )
    }

    frame = build_normalized_wealth_frame(results)

    assert list(frame["Strategy"]) == pytest.approx(
        [
            100.0,
            110.0,
            121.0,
        ]
    )


def test_drawdown_matches_known_path():
    results = {
        "Strategy": _make_result(
            wealth_values=[
                100.0,
                120.0,
                90.0,
                108.0,
            ],
            dates=[
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
        )
    }

    frame = build_drawdown_frame(results)

    assert list(frame["Strategy"]) == pytest.approx(
        [
            0.0,
            0.0,
            -0.25,
            -0.10,
        ]
    )


def test_reporting_rejects_mismatched_history_dates():
    results = {
        "Strategy A": _make_result(
            wealth_values=[
                100.0,
                101.0,
                102.0,
            ],
            dates=[
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
        ),
        "Strategy B": _make_result(
            wealth_values=[
                100.0,
                101.0,
                102.0,
            ],
            dates=[
                "2026-01-01",
                "2026-01-02",
                "2026-01-04",
            ],
        ),
    }

    with pytest.raises(
        ValueError,
        match="same dates",
    ):
        build_normalized_wealth_frame(results)

    with pytest.raises(
        ValueError,
        match="same dates",
    ):
        build_drawdown_frame(results)
