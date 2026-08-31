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

    assert result.final_wealth == pytest.approx(100_000.0)

    assert result.cumulative_return == pytest.approx(0.0)

    assert result.total_turnover == pytest.approx(0.0)


def test_buy_and_hold_benchmark_compounds_returns():
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

    assert result.final_wealth == pytest.approx(121_000.0)

    assert result.cumulative_return == pytest.approx(0.21)
