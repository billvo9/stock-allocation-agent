from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.features.training import (
    add_forward_return_target,
    select_training_rows_asof,
    split_temporal_dataset,
)


def _price_row(
    date: str,
    symbol: str,
    price: float,
) -> dict:
    """Create one simple asset-date price row."""

    return {
        "date": date,
        "symbol": symbol,
        "adjusted_close": price,
    }


def _price_series(
    symbol: str,
    start: str,
    prices: list[float],
) -> list[dict]:
    dates = pd.bdate_range(
        start=start,
        periods=len(prices),
    )

    return [
        _price_row(
            str(date.date()),
            symbol,
            price,
        )
        for date, price in zip(
            dates,
            prices,
        )
    ]


def test_forward_returns_are_independent_by_symbol():
    """
    Contract:

        target_return(symbol, t)
        =
        price(symbol, t+h)
        / price(symbol, t)
        - 1

    No target may cross from one symbol into another.
    """

    # Arrange
    frame = pd.DataFrame(
        [
            _price_row(
                "2026-01-02",
                "MU",
                100.0,
            ),
            _price_row(
                "2026-01-02",
                "SNDK",
                50.0,
            ),
            _price_row(
                "2026-01-05",
                "MU",
                110.0,
            ),
            _price_row(
                "2026-01-05",
                "SNDK",
                40.0,
            ),
            _price_row(
                "2026-01-06",
                "MU",
                121.0,
            ),
            _price_row(
                "2026-01-06",
                "SNDK",
                60.0,
            ),
        ]
    )

    result = add_forward_return_target(
        frame,
        horizon=1,
    )

    mu = result[result["symbol"].eq("MU")].reset_index(drop=True)

    sndk = result[result["symbol"].eq("SNDK")].reset_index(drop=True)

    assert mu.loc[
        0,
        "target_return",
    ] == pytest.approx(0.10)

    assert mu.loc[
        1,
        "target_return",
    ] == pytest.approx(0.10)

    assert pd.isna(
        mu.loc[
            2,
            "target_return",
        ]
    )

    assert sndk.loc[
        0,
        "target_return",
    ] == pytest.approx(-0.20)

    assert sndk.loc[
        1,
        "target_return",
    ] == pytest.approx(0.50)

    assert pd.isna(
        sndk.loc[
            2,
            "target_return",
        ]
    )


def test_twenty_period_target_uses_same_symbol_future_observation():
    """
    Contract:

        target_return_t
        =
        P_(t+20) / P_t - 1

    exactly 20 observations ahead,
    within the same symbol.
    """

    dates = pd.bdate_range(
        start="2026-01-02",
        periods=21,
    )

    frame = pd.DataFrame(
        _price_series(
            "MU",
            "2026-01-02",
            [100.0] * 20 + [120.0],
        )
        + _price_series(
            "SNDK",
            "2026-01-02",
            [50.0] * 20 + [40.0],
        )
    )

    result = add_forward_return_target(
        frame,
        horizon=20,
    )

    mu = result[result["symbol"].eq("MU")].reset_index(drop=True)

    sndk = result[result["symbol"].eq("SNDK")].reset_index(drop=True)

    assert mu.loc[
        0,
        "target_return",
    ] == pytest.approx(0.20)

    assert sndk.loc[
        0,
        "target_return",
    ] == pytest.approx(-0.20)

    assert mu["target_return"].iloc[1:].isna().all()

    assert sndk["target_return"].iloc[1:].isna().all()

    expected_end_date = pd.Timestamp(
        dates[20].date(),
        tz="UTC",
    )

    assert (
        mu.loc[
            0,
            "target_end_date",
        ]
        == expected_end_date
    )

    assert (
        sndk.loc[
            0,
            "target_end_date",
        ]
        == expected_end_date
    )


def test_asof_filter_rejects_unmatured_labels():
    """
    Contract:

        Keep row only if:

        target_return is known
        AND
        target_end_date < cutoff
    """
    frame = pd.DataFrame(
        [
            {
                "date": "2024-12-01",
                "target_return": 0.10,
                "target_end_date": ("2024-12-20"),
            },
            {
                "date": "2024-12-05",
                "target_return": 0.08,
                "target_end_date": ("2025-01-01"),
            },
            {
                "date": "2024-12-10",
                "target_return": 0.05,
                "target_end_date": ("2025-01-10"),
            },
        ]
    )

    result = select_training_rows_asof(
        frame,
        as_of="2025-01-01",
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["date"] == "2024-12-01"

    assert row["target_return"] == pytest.approx(0.10)

    assert row["target_end_date"] == pd.Timestamp(
        "2024-12-20",
        tz="UTC",
    )


def test_rows_go_into_correct_temporal_partitions():
    """
    Contracts:

        Train:
        2016-01-01 <= date <= 2024-12-31

        Validation:
        2025-01-01 <= date <= 2025-12-31

        Test:
        date >= 2026-01-01
    """
    frame = pd.DataFrame(
        [
            {
                "date": "2016-01-01",
                "symbol": "MU",
            },
            {
                "date": "2020-06-15",
                "symbol": "MU",
            },
            {
                "date": "2024-12-31",
                "symbol": "MU",
            },
            {
                "date": "2025-01-01",
                "symbol": "MU",
            },
            {
                "date": "2025-06-15",
                "symbol": "MU",
            },
            {
                "date": "2025-12-31",
                "symbol": "MU",
            },
            {
                "date": "2026-01-01",
                "symbol": "MU",
            },
            {
                "date": "2026-06-15",
                "symbol": "MU",
            },
        ]
    )
    split = split_temporal_dataset(frame)

    assert len(split.train) == 3
    assert len(split.validation) == 3
    assert len(split.test) == 2

    assert split.train["date"].min() == pd.Timestamp(
        "2016-01-01",
        tz="UTC",
    )

    assert split.train["date"].max() == pd.Timestamp(
        "2024-12-31",
        tz="UTC",
    )

    assert split.validation["date"].min() == pd.Timestamp(
        "2025-01-01",
        tz="UTC",
    )

    assert split.validation["date"].max() == pd.Timestamp(
        "2025-12-31",
        tz="UTC",
    )

    assert split.test["date"].min() == pd.Timestamp(
        "2026-01-01",
        tz="UTC",
    )


def test_adding_new_ticker_does_not_change_existing_asset_targets():
    """
    Architectural invariant:

        History(MU)
        =
        History(MU | MU + SNDK)

    Adding a newly listed asset must not
    truncate an older asset's history.
    """
    mu_only = pd.DataFrame(
        _price_series(
            "MU",
            "2024-12-20",
            [
                100.0,
                105.0,
                110.0,
                120.0,
            ],
        )
    )

    sndk = pd.DataFrame(
        _price_series(
            "SNDK",
            "2025-01-02",
            [
                50.0,
                52.0,
                55.0,
            ],
        )
    )

    expanded = pd.concat(
        [
            mu_only,
            sndk,
        ],
        ignore_index=True,
    )

    base_result = add_forward_return_target(
        mu_only,
        horizon=1,
    )

    expanded_result = add_forward_return_target(
        expanded,
        horizon=1,
    )

    base_mu = base_result[base_result["symbol"].eq("MU")].reset_index(drop=True)

    expanded_mu = expanded_result[expanded_result["symbol"].eq("MU")].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        base_mu,
        expanded_mu,
    )


def test_asof_accepts_timezone_aware_cutoff():
    frame = pd.DataFrame(
        [
            {
                "date": "2024-12-01",
                "target_return": 0.10,
                "target_end_date": ("2025-01-01 13:00:00+00:00"),
            }
        ]
    )

    cutoff = pd.Timestamp(
        "2025-01-01 09:00",
        tz="America/New_York",
    )

    result = select_training_rows_asof(
        frame,
        as_of=cutoff,
    )

    # Jan 1 09:00 New York
    # =
    # Jan 1 14:00 UTC
    #
    # 13:00 UTC < 14:00 UTC
    #
    # Therefore the target has matured.

    assert len(result) == 1


@pytest.mark.parametrize(
    "horizon",
    [
        0,
        -1,
    ],
)
def test_invalid_horizon_raises(
    horizon,
):
    frame = pd.DataFrame(
        [
            _price_row(
                "2026-01-02",
                "MU",
                100.0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        add_forward_return_target(
            frame,
            horizon=horizon,
        )


def test_duplicate_asset_date_raises():
    frame = pd.DataFrame(
        [
            _price_row(
                "2026-01-02",
                "MU",
                100.0,
            ),
            _price_row(
                "2026-01-02",
                "MU",
                101.0,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        add_forward_return_target(frame)


def test_invalid_training_date_raises():
    frame = pd.DataFrame(
        [
            _price_row(
                "not-a-date",
                "MU",
                100.0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="invalid dates",
    ):
        add_forward_return_target(frame)


def test_training_pipeline_excludes_labels_that_cross_cutoff():
    frame = pd.DataFrame(
        [
            _price_row(
                "2024-12-27",
                "MU",
                100.0,
            ),
            _price_row(
                "2024-12-30",
                "MU",
                102.0,
            ),
            _price_row(
                "2024-12-31",
                "MU",
                104.0,
            ),
            _price_row(
                "2025-01-02",
                "MU",
                106.0,
            ),
        ]
    )

    targeted = add_forward_return_target(
        frame,
        horizon=1,
    )

    split = split_temporal_dataset(targeted)

    safe_train = select_training_rows_asof(
        split.train,
        as_of="2025-01-01",
    )

    assert len(safe_train) == 2

    assert (
        safe_train["date"]
        <= pd.Timestamp(
            "2024-12-31",
            tz="UTC",
        )
    ).all()

    assert (
        safe_train["target_end_date"]
        < pd.Timestamp(
            "2025-01-01",
            tz="UTC",
        )
    ).all()
