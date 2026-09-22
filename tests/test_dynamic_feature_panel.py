import pandas as pd
import pytest

from stock_agent.data.macro.schema import MACRO_COLUMNS
from stock_agent.features.panel import (
    attach_macro_features,
    build_dynamic_feature_panel,
)


def _market_frame(
    rows: list[dict],
) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _macro_vintage_frame(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Build a canonical macro-vintage frame for panel tests."""

    frame = pd.DataFrame(
        rows,
        columns=MACRO_COLUMNS,
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


def _unemployment_vintage(
    *,
    observation_date: str,
    available_at: str,
    value: float,
    vintage_date: str | None = None,
) -> dict[str, object]:
    """Create one unemployment vintage."""

    return {
        "observation_date": observation_date,
        "available_at": available_at,
        "vintage_date": (vintage_date if vintage_date is not None else available_at),
        "series_id": "us_unemployment_rate",
        "provider_series_id": "UNRATE",
        "value": value,
        "frequency": "monthly",
        "units": "percent",
        "source": "FRED",
    }


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


def test_macro_features_are_shared_across_symbols_on_same_date():
    asset_panel = pd.DataFrame(
        [
            {
                "date": "2020-02-10",
                "symbol": "MU",
                "adjusted_close": 55.0,
            },
            {
                "date": "2020-02-10",
                "symbol": "NVDA",
                "adjusted_close": 60.0,
            },
        ]
    )

    macro_vintages = _macro_vintage_frame(
        [
            _unemployment_vintage(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = attach_macro_features(
        asset_panel=asset_panel,
        macro_vintages=macro_vintages,
    )

    same_date = result[
        result["date"]
        == pd.Timestamp(
            "2020-02-10",
            tz="UTC",
        )
    ]

    assert set(same_date["symbol"]) == {
        "MU",
        "NVDA",
    }

    assert same_date["us_unemployment_rate"].nunique() == 1

    assert same_date["us_unemployment_rate"].iloc[0] == pytest.approx(3.5)


def test_macro_integration_preserves_asset_rows():
    asset_panel = pd.DataFrame(
        [
            {
                "date": "2020-02-10",
                "symbol": "MU",
                "adjusted_close": 55.0,
            },
            {
                "date": "2020-02-10",
                "symbol": "NVDA",
                "adjusted_close": 60.0,
            },
            {
                "date": "2020-02-11",
                "symbol": "MU",
                "adjusted_close": 56.0,
            },
        ]
    )

    macro_vintages = _macro_vintage_frame(
        [
            _unemployment_vintage(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = attach_macro_features(
        asset_panel=asset_panel,
        macro_vintages=macro_vintages,
    )

    assert len(result) == len(asset_panel)

    assert result["symbol"].tolist() == (asset_panel["symbol"].tolist())


def test_macro_integration_preserves_asset_date_keys():
    asset_panel = pd.DataFrame(
        [
            {
                "date": "2020-02-10",
                "symbol": "MU",
                "adjusted_close": 55.0,
            },
            {
                "date": "2020-02-10",
                "symbol": "NVDA",
                "adjusted_close": 60.0,
            },
            {
                "date": "2020-02-11",
                "symbol": "MU",
                "adjusted_close": 56.0,
            },
            {
                "date": "2020-02-11",
                "symbol": "NVDA",
                "adjusted_close": 61.0,
            },
        ]
    )

    macro_vintages = _macro_vintage_frame(
        [
            _unemployment_vintage(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = attach_macro_features(
        asset_panel=asset_panel,
        macro_vintages=macro_vintages,
    )

    expected = asset_panel.copy()

    expected["date"] = pd.to_datetime(
        expected["date"],
        utc=True,
    ).dt.normalize()

    expected_keys = set(
        zip(
            expected["date"],
            expected["symbol"],
        )
    )

    result_keys = set(
        zip(
            result["date"],
            result["symbol"],
        )
    )

    assert result_keys == expected_keys

    assert not result.duplicated(["date", "symbol"]).any()


def test_macro_features_change_over_time_not_by_symbol():
    asset_panel = pd.DataFrame(
        [
            {
                "date": "2020-02-10",
                "symbol": "MU",
                "adjusted_close": 55.0,
            },
            {
                "date": "2020-02-10",
                "symbol": "NVDA",
                "adjusted_close": 60.0,
            },
            {
                "date": "2020-03-10",
                "symbol": "MU",
                "adjusted_close": 57.0,
            },
            {
                "date": "2020-03-10",
                "symbol": "NVDA",
                "adjusted_close": 62.0,
            },
        ]
    )

    macro_vintages = _macro_vintage_frame(
        [
            _unemployment_vintage(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
            _unemployment_vintage(
                observation_date="2020-02-01",
                available_at="2020-03-06",
                value=3.8,
            ),
        ]
    )

    result = attach_macro_features(
        asset_panel=asset_panel,
        macro_vintages=macro_vintages,
    )

    february = result[
        result["date"]
        == pd.Timestamp(
            "2020-02-10",
            tz="UTC",
        )
    ]

    march = result[
        result["date"]
        == pd.Timestamp(
            "2020-03-10",
            tz="UTC",
        )
    ]

    # Same macro environment for every asset
    # evaluated on the same date.
    assert february["us_unemployment_rate"].nunique() == 1

    assert march["us_unemployment_rate"].nunique() == 1

    # But the macro environment may evolve
    # from one decision date to another.
    assert february["us_unemployment_rate"].iloc[0] == pytest.approx(3.5)

    assert march["us_unemployment_rate"].iloc[0] == pytest.approx(3.8)


def test_asset_date_before_macro_release_is_preserved():
    asset_panel = pd.DataFrame(
        [
            {
                "date": "2020-02-06",
                "symbol": "MU",
                "adjusted_close": 55.0,
            },
            {
                "date": "2020-02-06",
                "symbol": "NVDA",
                "adjusted_close": 60.0,
            },
        ]
    )

    macro_vintages = _macro_vintage_frame(
        [
            _unemployment_vintage(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = attach_macro_features(
        asset_panel=asset_panel,
        macro_vintages=macro_vintages,
    )

    # No asset rows are deleted simply because the
    # macro release was not available yet.
    assert len(result) == 2

    assert set(result["symbol"]) == {
        "MU",
        "NVDA",
    }

    assert result["us_unemployment_rate"].isna().all()

    assert not result["us_unemployment_available"].any()

    assert result["macro_coverage_ratio"].tolist() == pytest.approx([0.0, 0.0])


def test_macro_revision_only_changes_panel_after_available_date():
    asset_panel = pd.DataFrame(
        [
            {
                "date": "2020-03-13",
                "symbol": "MU",
                "adjusted_close": 55.0,
            },
            {
                "date": "2020-03-13",
                "symbol": "NVDA",
                "adjusted_close": 60.0,
            },
            {
                "date": "2020-03-16",
                "symbol": "MU",
                "adjusted_close": 56.0,
            },
            {
                "date": "2020-03-16",
                "symbol": "NVDA",
                "adjusted_close": 61.0,
            },
        ]
    )

    macro_vintages = _macro_vintage_frame(
        [
            # Original January release.
            _unemployment_vintage(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
            # Revision to the exact same economic
            # observation becomes available later.
            _unemployment_vintage(
                observation_date="2020-01-01",
                available_at="2020-03-15",
                vintage_date="2020-03-15",
                value=3.7,
            ),
        ]
    )

    result = attach_macro_features(
        asset_panel=asset_panel,
        macro_vintages=macro_vintages,
    )

    before_revision = result[
        result["date"]
        == pd.Timestamp(
            "2020-03-13",
            tz="UTC",
        )
    ]

    after_revision = result[
        result["date"]
        == pd.Timestamp(
            "2020-03-16",
            tz="UTC",
        )
    ]

    # March 14 cannot see the March 15 revision.
    assert before_revision["us_unemployment_rate"].tolist() == pytest.approx([3.5, 3.5])

    # March 16 may use it.
    assert after_revision["us_unemployment_rate"].tolist() == pytest.approx([3.7, 3.7])

    # Revision timing is global, not ticker-specific.
    assert before_revision["us_unemployment_rate"].nunique() == 1

    assert after_revision["us_unemployment_rate"].nunique() == 1
