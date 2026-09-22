from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.macro.schema import (
    MACRO_COLUMNS,
)
from stock_agent.features.macro import (
    MACRO_FEATURE_COLUMNS,
    build_macro_features,
)


def _macro_row(
    *,
    series_id: str,
    provider_series_id: str,
    observation_date: str,
    available_at: str,
    value: float,
    frequency: str,
    units: str,
    vintage_date: str | None = None,
) -> dict[str, object]:
    return {
        "observation_date": observation_date,
        "available_at": available_at,
        "vintage_date": (vintage_date if vintage_date is not None else available_at),
        "series_id": series_id,
        "provider_series_id": (provider_series_id),
        "value": value,
        "frequency": frequency,
        "units": units,
        "source": "FRED",
    }


def _macro_frame(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
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


def test_unemployment_level_and_monthly_changes():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=4.0,
                frequency="monthly",
                units="percent",
            ),
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-02-01",
                available_at="2020-03-06",
                value=4.2,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-03-10",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert row["us_unemployment_rate"] == pytest.approx(4.2)

    assert row["us_unemployment_change_1m_pp"] == pytest.approx(0.2)

    assert pd.isna(row["us_unemployment_change_3m_pp"])

    assert bool(row["us_unemployment_available"]) is True


def test_cpi_yoy_and_inflation_acceleration():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_consumer_price_index",
                provider_series_id="CPIAUCSL",
                observation_date="2018-12-01",
                available_at="2019-01-15",
                value=250.0,
                frequency="monthly",
                units="index",
            ),
            _macro_row(
                series_id="us_consumer_price_index",
                provider_series_id="CPIAUCSL",
                observation_date="2019-01-01",
                available_at="2019-02-15",
                value=251.0,
                frequency="monthly",
                units="index",
            ),
            _macro_row(
                series_id="us_consumer_price_index",
                provider_series_id="CPIAUCSL",
                observation_date="2019-12-01",
                available_at="2020-01-14",
                value=255.0,
                frequency="monthly",
                units="index",
            ),
            _macro_row(
                series_id="us_consumer_price_index",
                provider_series_id="CPIAUCSL",
                observation_date="2020-01-01",
                available_at="2020-02-13",
                value=257.0,
                frequency="monthly",
                units="index",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-02-20",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    current_yoy = 257.0 / 251.0 - 1.0

    prior_yoy = 255.0 / 250.0 - 1.0

    assert row["us_cpi_yoy"] == pytest.approx(current_yoy)

    assert row["us_cpi_yoy_change_1m"] == pytest.approx(current_yoy - prior_yoy)


def test_real_gdp_yoy_and_qoq_annualized():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_real_gdp",
                provider_series_id="GDPC1",
                observation_date="2019-01-01",
                available_at="2019-04-29",
                value=100.0,
                frequency="quarterly",
                units=("billions_chained_2017_dollars"),
            ),
            _macro_row(
                series_id="us_real_gdp",
                provider_series_id="GDPC1",
                observation_date="2019-10-01",
                available_at="2020-01-30",
                value=110.0,
                frequency="quarterly",
                units=("billions_chained_2017_dollars"),
            ),
            _macro_row(
                series_id="us_real_gdp",
                provider_series_id="GDPC1",
                observation_date="2020-01-01",
                available_at="2020-04-29",
                value=112.0,
                frequency="quarterly",
                units=("billions_chained_2017_dollars"),
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-04-30",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert row["us_real_gdp_yoy"] == pytest.approx(112.0 / 100.0 - 1.0)

    assert row["us_real_gdp_qoq_annualized"] == pytest.approx((112.0 / 110.0) ** 4 - 1.0)


def test_fed_funds_changes_are_percentage_points():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_fed_funds_rate",
                provider_series_id="FEDFUNDS",
                observation_date="2020-01-01",
                available_at="2020-02-01",
                value=2.00,
                frequency="monthly",
                units="percent",
            ),
            _macro_row(
                series_id="us_fed_funds_rate",
                provider_series_id="FEDFUNDS",
                observation_date="2020-02-01",
                available_at="2020-03-01",
                value=1.75,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-03-05",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert row["us_fed_funds_rate"] == pytest.approx(1.75)

    assert row["us_fed_funds_change_1m_pp"] == pytest.approx(-0.25)


def test_m2_yoy_and_growth_acceleration():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_m2_money_supply",
                provider_series_id="M2SL",
                observation_date="2018-12-01",
                available_at="2019-01-15",
                value=100.0,
                frequency="monthly",
                units="billions_usd",
            ),
            _macro_row(
                series_id="us_m2_money_supply",
                provider_series_id="M2SL",
                observation_date="2019-01-01",
                available_at="2019-02-15",
                value=101.0,
                frequency="monthly",
                units="billions_usd",
            ),
            _macro_row(
                series_id="us_m2_money_supply",
                provider_series_id="M2SL",
                observation_date="2019-12-01",
                available_at="2020-01-15",
                value=108.0,
                frequency="monthly",
                units="billions_usd",
            ),
            _macro_row(
                series_id="us_m2_money_supply",
                provider_series_id="M2SL",
                observation_date="2020-01-01",
                available_at="2020-02-15",
                value=112.0,
                frequency="monthly",
                units="billions_usd",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-02-20",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    current_yoy = 112.0 / 101.0 - 1.0

    previous_yoy = 108.0 / 100.0 - 1.0

    assert row["us_m2_yoy"] == pytest.approx(current_yoy)

    assert row["us_m2_yoy_change_1m"] == pytest.approx(current_yoy - previous_yoy)


def test_future_revision_does_not_leak_into_yoy():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_consumer_price_index",
                provider_series_id="CPIAUCSL",
                observation_date="2020-01-01",
                available_at="2020-02-15",
                value=100.0,
                frequency="monthly",
                units="index",
            ),
            _macro_row(
                series_id="us_consumer_price_index",
                provider_series_id="CPIAUCSL",
                observation_date="2021-01-01",
                available_at="2021-02-15",
                value=120.0,
                frequency="monthly",
                units="index",
            ),
            _macro_row(
                series_id="us_consumer_price_index",
                provider_series_id="CPIAUCSL",
                observation_date="2020-01-01",
                available_at="2021-03-15",
                value=80.0,
                frequency="monthly",
                units="index",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2021-02-20",
            "2021-03-20",
        ],
        macro_vintages=frame,
    )

    before_revision = result.iloc[0]
    after_revision = result.iloc[1]

    assert before_revision["us_cpi_yoy"] == pytest.approx(120.0 / 100.0 - 1.0)

    assert after_revision["us_cpi_yoy"] == pytest.approx(120.0 / 80.0 - 1.0)


def test_missing_expected_period_does_not_use_neighbor():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=4.0,
                frequency="monthly",
                units="percent",
            ),
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-03-01",
                available_at="2020-04-03",
                value=4.5,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-04-05",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert pd.isna(row["us_unemployment_change_1m_pp"])

    assert pd.isna(row["us_unemployment_change_3m_pp"])


def test_future_release_is_not_available():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=4.0,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-02-06",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert bool(row["us_unemployment_available"]) is False

    assert pd.isna(row["us_unemployment_rate"])


def test_macro_coverage_ratio_reflects_available_series():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=4.0,
                frequency="monthly",
                units="percent",
            ),
            _macro_row(
                series_id="us_fed_funds_rate",
                provider_series_id="FEDFUNDS",
                observation_date="2020-01-01",
                available_at="2020-02-01",
                value=1.5,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-02-10",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    # 2 of the 5 core macro series are available.
    assert row["macro_coverage_ratio"] == pytest.approx(2 / 5)


def test_observation_age_uses_calendar_days():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=4.0,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-02-10",
        ],
        macro_vintages=frame,
    )

    assert result.iloc[0]["us_unemployment_age_days"] == pytest.approx(40.0)


def test_unsorted_input_produces_deterministic_output():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_fed_funds_rate",
                provider_series_id="FEDFUNDS",
                observation_date="2020-02-01",
                available_at="2020-03-01",
                value=1.75,
                frequency="monthly",
                units="percent",
            ),
            _macro_row(
                series_id="us_fed_funds_rate",
                provider_series_id="FEDFUNDS",
                observation_date="2020-01-01",
                available_at="2020-02-01",
                value=2.00,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-03-05",
            "2020-02-05",
        ],
        macro_vintages=frame,
    )

    assert list(result.columns) == MACRO_FEATURE_COLUMNS

    assert result["date"].is_monotonic_increasing

    march = result[
        result["date"]
        == pd.Timestamp(
            "2020-03-05",
            tz="UTC",
        )
    ].iloc[0]

    assert march["us_fed_funds_rate"] == pytest.approx(1.75)


def test_duplicate_decision_dates_are_collapsed():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=4.0,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    result = build_macro_features(
        decision_dates=[
            "2020-02-10",
            "2020-02-10",
        ],
        macro_vintages=frame,
    )

    assert len(result) == 1


def test_invalid_decision_date_raises():
    frame = _macro_frame(
        [
            _macro_row(
                series_id="us_unemployment_rate",
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=4.0,
                frequency="monthly",
                units="percent",
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="invalid",
    ):
        build_macro_features(
            decision_dates=[
                "not-a-date",
            ],
            macro_vintages=frame,
        )
