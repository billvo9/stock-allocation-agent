import pandas as pd
import pytest

from stock_agent.features.panel import (
    build_dynamic_feature_panel,
)


def _market_frame(
    rows: list[dict],
) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_old_asset_retains_full_history():
    market = _market_frame(
        [
            {
                "date": "2020-01-02",
                "symbol": "MU",
                "momentum": 0.10,
            },
            {
                "date": "2021-01-04",
                "symbol": "MU",
                "momentum": 0.20,
            },
            {
                "date": "2025-02-24",
                "symbol": "MU",
                "momentum": 0.30,
            },
            {
                "date": "2025-02-24",
                "symbol": "SNDK",
                "momentum": 0.05,
            },
        ]
    )

    result = build_dynamic_feature_panel(market)

    mu = result[result["symbol"].eq("MU")]

    assert len(mu) == 3

    assert mu["date"].min() == pd.Timestamp(
        "2020-01-02",
        tz="UTC",
    )


def test_adding_new_asset_does_not_truncate_old_asset():
    mu_only = _market_frame(
        [
            {
                "date": "2020-01-02",
                "symbol": "MU",
                "momentum": 0.10,
            },
            {
                "date": "2021-01-04",
                "symbol": "MU",
                "momentum": 0.20,
            },
            {
                "date": "2022-01-03",
                "symbol": "MU",
                "momentum": 0.30,
            },
        ]
    )

    with_new_asset = pd.concat(
        [
            mu_only,
            _market_frame(
                [
                    {
                        "date": "2025-02-24",
                        "symbol": "SNDK",
                        "momentum": 0.05,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    base_result = build_dynamic_feature_panel(mu_only)

    expanded_result = build_dynamic_feature_panel(with_new_asset)

    base_mu = base_result[base_result["symbol"].eq("MU")]

    expanded_mu = expanded_result[expanded_result["symbol"].eq("MU")]

    pd.testing.assert_frame_equal(
        base_mu.reset_index(drop=True),
        expanded_mu.reset_index(drop=True),
    )


def test_duplicate_asset_date_raises():
    market = _market_frame(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
            },
            {
                "date": "2026-01-02",
                "symbol": "MU",
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        build_dynamic_feature_panel(market)


def test_panel_is_sorted_by_symbol_and_date():
    market = _market_frame(
        [
            {
                "date": "2026-01-03",
                "symbol": "SNDK",
            },
            {
                "date": "2026-01-03",
                "symbol": "MU",
            },
            {
                "date": "2026-01-02",
                "symbol": "MU",
            },
        ]
    )

    result = build_dynamic_feature_panel(market)

    assert result[["symbol", "date"]].to_records(index=False).tolist() == [
        (
            "MU",
            pd.Timestamp(
                "2026-01-02",
                tz="UTC",
            ),
        ),
        (
            "MU",
            pd.Timestamp(
                "2026-01-03",
                tz="UTC",
            ),
        ),
        (
            "SNDK",
            pd.Timestamp(
                "2026-01-03",
                tz="UTC",
            ),
        ),
    ]


def test_missing_fundamentals_do_not_drop_asset():
    market = _market_frame(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "momentum": 0.10,
            },
            {
                "date": "2026-01-02",
                "symbol": "SNDK",
                "momentum": 0.20,
            },
        ]
    )

    fundamentals = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "gross_margin": 0.40,
            },
        ]
    )

    result = build_dynamic_feature_panel(
        market_features=market,
        feature_frames={
            "fundamentals": fundamentals,
        },
    )

    assert len(result) == 2

    sndk = result.loc[result["symbol"].eq("SNDK")].iloc[0]

    assert pd.isna(sndk["gross_margin"])


def test_point_in_time_information_is_preserved():
    market = _market_frame(
        [
            {
                "date": "2026-06-30",
                "symbol": "MU",
            },
        ]
    )

    fundamentals = pd.DataFrame(
        [
            {
                "date": "2026-06-30",
                "symbol": "MU",
                "available_at": ("2026-06-27"),
                "gross_margin": 0.40,
            },
        ]
    )

    result = build_dynamic_feature_panel(
        market_features=market,
        feature_frames={
            "fundamentals": fundamentals,
        },
    )

    assert (result["available_at"].isna() | (result["available_at"] <= result["date"])).all()


def test_future_information_raises():
    market = _market_frame(
        [
            {
                "date": "2026-06-26",
                "symbol": "MU",
            },
        ]
    )

    fundamentals = pd.DataFrame(
        [
            {
                "date": "2026-06-26",
                "symbol": "MU",
                "available_at": ("2026-06-27"),
                "gross_margin": 0.40,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="future information",
    ):
        build_dynamic_feature_panel(
            market_features=market,
            feature_frames={
                "fundamentals": (fundamentals),
            },
        )


def test_supplemental_frame_cannot_overwrite_features():
    market = _market_frame(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "momentum": 0.10,
            },
        ]
    )

    fundamentals = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "momentum": 0.90,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="overwrite",
    ):
        build_dynamic_feature_panel(
            market_features=market,
            feature_frames={
                "fundamentals": (fundamentals),
            },
        )
