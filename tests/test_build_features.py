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


def _unemployment_macro_frame(
    observation_date: str = "2020-01-01",
    available_at: str = "2020-02-07",
    value: float = 3.5,
) -> pd.DataFrame:
    """Create one canonical unemployment macro vintage."""

    frame = pd.DataFrame(
        [
            {
                "observation_date": observation_date,
                "available_at": available_at,
                "vintage_date": available_at,
                "series_id": "us_unemployment_rate",
                "provider_series_id": "UNRATE",
                "value": value,
                "frequency": "monthly",
                "units": "percent",
                "source": "FRED",
            }
        ]
    )

    for column in [
        "observation_date",
        "available_at",
        "vintage_date",
    ]:
        frame[column] = pd.to_datetime(
            frame[column],
            utc=True,
        )

    return frame


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

    assert mu["date"].min() == pd.Timestamp(
        "2020-01-02",
        tz="UTC",
    )

    assert sndk["date"].min() == pd.Timestamp(
        "2025-02-24",
        tz="UTC",
    )


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


def test_build_model_dataset_attaches_macro_features(
    monkeypatch,
):
    frame = pd.DataFrame(
        [
            _model_feature_row(
                "2020-02-10",
                "MU",
            ),
            _model_feature_row(
                "2020-02-10",
                "NVDA",
                adjusted_close=200.0,
            ),
        ]
    )

    macro_vintages = _unemployment_macro_frame(
        observation_date="2020-01-01",
        available_at="2020-02-07",
        value=3.5,
    )

    monkeypatch.setattr(
        build_module,
        "run_feature_query",
        lambda: frame.copy(),
    )

    result = build_model_dataset(
        macro_vintages=macro_vintages,
    )

    assert "us_unemployment_rate" in result.columns
    assert "us_unemployment_available" in result.columns
    assert "macro_coverage_ratio" in result.columns

    assert result["us_unemployment_rate"].tolist() == [
        3.5,
        3.5,
    ]

    assert result["us_unemployment_available"].all()


def test_build_model_dataset_macro_does_not_change_asset_keys(
    monkeypatch,
):
    frame = pd.DataFrame(
        [
            _model_feature_row(
                "2020-02-10",
                "MU",
            ),
            _model_feature_row(
                "2020-02-10",
                "NVDA",
                adjusted_close=200.0,
            ),
            _model_feature_row(
                "2020-02-11",
                "MU",
                adjusted_close=101.0,
            ),
            _model_feature_row(
                "2020-02-11",
                "NVDA",
                adjusted_close=201.0,
            ),
        ]
    )

    macro_vintages = _unemployment_macro_frame()

    monkeypatch.setattr(
        build_module,
        "run_feature_query",
        lambda: frame.copy(),
    )

    without_macro = build_model_dataset()

    with_macro = build_model_dataset(
        macro_vintages=macro_vintages,
    )

    without_macro_keys = set(
        zip(
            without_macro["date"],
            without_macro["symbol"],
        )
    )

    with_macro_keys = set(
        zip(
            with_macro["date"],
            with_macro["symbol"],
        )
    )

    assert with_macro_keys == without_macro_keys
    assert len(with_macro) == len(without_macro)

    assert not with_macro.duplicated(["date", "symbol"]).any()


def test_build_model_dataset_macro_does_not_shorten_longer_asset_history(
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
                adjusted_close=110.0,
            ),
            _model_feature_row(
                "2025-02-24",
                "MU",
                adjusted_close=120.0,
            ),
            _model_feature_row(
                "2025-02-24",
                "SNDK",
                adjusted_close=50.0,
            ),
        ]
    )

    macro_vintages = _unemployment_macro_frame(
        observation_date="2019-11-01",
        available_at="2019-12-06",
        value=3.5,
    )

    monkeypatch.setattr(
        build_module,
        "run_feature_query",
        lambda: frame.copy(),
    )

    result = build_model_dataset(
        macro_vintages=macro_vintages,
    )

    mu = result[result["symbol"] == "MU"].reset_index(drop=True)

    sndk = result[result["symbol"] == "SNDK"].reset_index(drop=True)

    assert len(mu) == 3
    assert len(sndk) == 1

    assert mu["date"].min() == pd.Timestamp(
        "2020-01-02",
        tz="UTC",
    )

    assert sndk["date"].min() == pd.Timestamp(
        "2025-02-24",
        tz="UTC",
    )

    assert mu["date"].tolist() == [
        pd.Timestamp(
            "2020-01-02",
            tz="UTC",
        ),
        pd.Timestamp(
            "2021-01-04",
            tz="UTC",
        ),
        pd.Timestamp(
            "2025-02-24",
            tz="UTC",
        ),
    ]


def test_build_model_dataset_without_macro_preserves_backward_compatibility(
    monkeypatch,
):
    frame = pd.DataFrame(
        [
            _model_feature_row(
                "2020-01-02",
                "MU",
            ),
            _model_feature_row(
                "2020-01-03",
                "MU",
                adjusted_close=101.0,
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

    default_result = build_model_dataset()

    explicit_none_result = build_model_dataset(
        macro_vintages=None,
    )

    pd.testing.assert_frame_equal(
        default_result,
        explicit_none_result,
    )

    assert "macro_coverage_ratio" not in (default_result.columns)

    assert "us_unemployment_rate" not in (default_result.columns)
