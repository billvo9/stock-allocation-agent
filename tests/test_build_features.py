import pandas as pd

import stock_agent.features.build as build_module
from stock_agent.features.build import (
    build_model_dataset,
    find_common_investable_dates,
)


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


def _model_feature_row(
    date: str,
    symbol: str,
    adjusted_close: float = 100.0,
) -> dict:
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "adjusted_close": adjusted_close,
        "daily_return": 0.01,
        "momentum_20d": 0.10,
        "volatility_20d": 0.20,
    }


def test_build_model_dataset_preserves_longer_asset_history(
    monkeypatch,
):
    frame = pd.DataFrame(
        [
            _model_feature_row(
                "2020-01-02",
                "MU",
            ),
            _model_feature_row(
                "2021-01-04",
                "MU",
            ),
            _model_feature_row(
                "2025-02-24",
                "MU",
            ),
            _model_feature_row(
                "2025-02-24",
                "SNDK",
                adjusted_close=50.0,
            ),
        ]
    )

    monkeypatch.setattr(
        build_module,
        "run_feature_query",
        lambda: frame.copy(),
    )

    result = build_model_dataset()

    mu = result[result["symbol"].eq("MU")]

    sndk = result[result["symbol"].eq("SNDK")]

    assert len(mu) == 3
    assert len(sndk) == 1

    assert mu["date"].min() == pd.Timestamp("2020-01-02")


def test_adding_new_asset_does_not_change_existing_asset_history(
    monkeypatch,
):
    mu_only = pd.DataFrame(
        [
            _model_feature_row(
                "2020-01-02",
                "MU",
            ),
            _model_feature_row(
                "2021-01-04",
                "MU",
            ),
            _model_feature_row(
                "2022-01-03",
                "MU",
            ),
        ]
    )

    expanded = pd.concat(
        [
            mu_only,
            pd.DataFrame(
                [
                    _model_feature_row(
                        "2025-02-24",
                        "SNDK",
                        adjusted_close=50.0,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    monkeypatch.setattr(
        build_module,
        "run_feature_query",
        lambda: mu_only.copy(),
    )

    base_result = build_model_dataset()

    monkeypatch.setattr(
        build_module,
        "run_feature_query",
        lambda: expanded.copy(),
    )

    expanded_result = build_model_dataset()

    base_mu = base_result[base_result["symbol"].eq("MU")].reset_index(drop=True)

    expanded_mu = expanded_result[expanded_result["symbol"].eq("MU")].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        base_mu,
        expanded_mu,
    )
