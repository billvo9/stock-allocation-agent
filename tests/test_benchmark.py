from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.evaluation.benchmark import (
    run_buy_and_hold_benchmark,
)


def test_buy_and_hold_benchmark_zero_returns():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "symbol": [
                "SP500",
                "SP500",
                "SP500",
            ],
            "daily_return": [
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    result = run_buy_and_hold_benchmark(
        frame=frame,
        symbol="SP500",
        initial_wealth=100_000.0,
    )

    # D3 (approved 2026-10-15): the benchmark starts in cash and pays the D2 build cost,
    # the provisional broad-ETF rate of 5 bp, on a full switch (one-way turnover 1.0).
    assert result.final_wealth == pytest.approx(99_950.0)

    assert result.cumulative_return == pytest.approx(-0.0005)

    assert result.total_turnover == pytest.approx(1.0)


def test_buy_and_hold_benchmark_compounds_returns():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                ]
            ),
            "symbol": [
                "SP500",
                "SP500",
                "SP500",
                "SP500",
            ],
            # OD8.14 rewrite: built at the Jan 2 close, the benchmark earns Jan 3 and Jan 4.
            "daily_return": [
                0.0,
                0.50,
                0.10,
                0.10,
            ],
        }
    )

    result = run_buy_and_hold_benchmark(
        frame=frame,
        symbol="SP500",
        initial_wealth=100_000.0,
    )

    assert result.final_wealth == pytest.approx(100_000.0 * 0.9995 * 1.21)

    assert result.cumulative_return == pytest.approx(0.9995 * 1.21 - 1)


def test_buy_and_hold_benchmark_builds_at_the_common_build_date():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    frame = pd.DataFrame(
        {"date": dates, "symbol": ["SP500"] * 5, "daily_return": [0.0, 0.5, 0.5, 0.10, 0.10]}
    )

    result = run_buy_and_hold_benchmark(
        frame=frame,
        symbol="SP500",
        initial_wealth=100_000.0,
        build_date=dates[2],
    )

    assert result.build_date == dates[2]
    assert result.history.wealth.index[0] == dates[3]  # executes at the next close (D1)
    assert result.final_wealth == pytest.approx(100_000.0 * 0.9995 * 1.10)
    assert result.simulation.steps[0].cost == pytest.approx(0.0005)
