import pandas as pd

from stock_agent.features.build import find_common_investable_dates


def test_find_common_investable_dates_requires_all_symbols():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
    )

    frame = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "MU"},
            {"date": dates[0], "symbol": "NVDA"},
            {"date": dates[1], "symbol": "MU"},
            {"date": dates[2], "symbol": "MU"},
            {"date": dates[2], "symbol": "NVDA"},
        ]
    )

    result = find_common_investable_dates(
        frame=frame,
        investable_symbols=["MU", "NVDA"],
    )

    assert list(result) == [
        dates[0],
        dates[2],
    ]


def test_find_common_investable_dates_respects_requested_universe():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    frame = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "MU"},
            {"date": dates[0], "symbol": "NVDA"},
            {"date": dates[1], "symbol": "MU"},
            {"date": dates[1], "symbol": "NVDA"},
            {"date": dates[1], "symbol": "INTC"},
        ]
    )

    two_asset_result = find_common_investable_dates(
        frame=frame,
        investable_symbols=["MU", "NVDA"],
    )

    three_asset_result = find_common_investable_dates(
        frame=frame,
        investable_symbols=["MU", "NVDA", "INTC"],
    )

    assert list(two_asset_result) == [
        dates[0],
        dates[1],
    ]

    assert list(three_asset_result) == [
        dates[1],
    ]
