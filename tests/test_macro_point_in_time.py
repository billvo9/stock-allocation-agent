from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.macro.point_in_time import (
    MACRO_STATE_COLUMNS,
    build_macro_state_asof,
)
from stock_agent.data.macro.schema import (
    MACRO_COLUMNS,
)


def _macro_row(
    *,
    series_id: str = "us_unemployment_rate",
    provider_series_id: str = "UNRATE",
    observation_date: str,
    available_at: str,
    vintage_date: str | None = None,
    value: float,
    frequency: str = "monthly",
    units: str = "percent",
    source: str = "FRED",
) -> dict[str, object]:
    """
    Create one canonical macro-vintage observation.
    """

    return {
        "observation_date": observation_date,
        "available_at": available_at,
        "vintage_date": (vintage_date if vintage_date is not None else available_at),
        "series_id": series_id,
        "provider_series_id": (provider_series_id),
        "value": value,
        "frequency": frequency,
        "units": units,
        "source": source,
    }


def _macro_frame(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Create a canonical test macro frame."""

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


def test_release_is_invisible_before_available_at():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-02-06",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert bool(row["macro_available"]) is False

    assert pd.isna(row["value"])

    assert pd.isna(row["observation_date"])


def test_release_becomes_visible_on_available_date():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-02-07",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert bool(row["macro_available"]) is True

    assert row["value"] == pytest.approx(3.5)

    assert row["observation_date"] == pd.Timestamp(
        "2020-01-01",
        tz="UTC",
    )


def test_future_revision_does_not_rewrite_history():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-03-10",
                value=3.7,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-02-20",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert row["value"] == pytest.approx(3.5)

    assert row["available_at"] == pd.Timestamp(
        "2020-02-07",
        tz="UTC",
    )


def test_revision_replaces_previous_vintage_after_release():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-03-10",
                value=3.7,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-03-10",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert row["value"] == pytest.approx(3.7)

    assert row["available_at"] == pd.Timestamp(
        "2020-03-10",
        tz="UTC",
    )


def test_newer_observation_replaces_older_observation():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
            _macro_row(
                observation_date="2020-02-01",
                available_at="2020-03-06",
                value=3.6,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-02-20",
            "2020-03-06",
        ],
        macro_vintages=frame,
    )

    before = result.iloc[0]
    after = result.iloc[1]

    assert before["observation_date"] == pd.Timestamp(
        "2020-01-01",
        tz="UTC",
    )

    assert before["value"] == pytest.approx(3.5)

    assert after["observation_date"] == pd.Timestamp(
        "2020-02-01",
        tz="UTC",
    )

    assert after["value"] == pytest.approx(3.6)


def test_revision_to_old_observation_does_not_replace_newer_observation():
    """
    A late revision to January must not replace February as the
    latest economic observation.
    """

    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
            _macro_row(
                observation_date="2020-02-01",
                available_at="2020-03-06",
                value=3.6,
            ),
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-03-20",
                value=3.8,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-03-25",
        ],
        macro_vintages=frame,
    )

    row = result.iloc[0]

    assert row["observation_date"] == pd.Timestamp(
        "2020-02-01",
        tz="UTC",
    )

    assert row["value"] == pytest.approx(3.6)


def test_series_are_selected_independently():
    frame = _macro_frame(
        [
            _macro_row(
                series_id=("us_unemployment_rate"),
                provider_series_id="UNRATE",
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
                units="percent",
            ),
            _macro_row(
                series_id=("us_consumer_price_index"),
                provider_series_id="CPIAUCSL",
                observation_date="2020-01-01",
                available_at="2020-02-13",
                value=259.1,
                units="index",
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-02-10",
            "2020-02-15",
        ],
        macro_vintages=frame,
    )

    feb_10 = result[
        result["date"]
        == pd.Timestamp(
            "2020-02-10",
            tz="UTC",
        )
    ].set_index("series_id")

    assert (
        bool(
            feb_10.loc[
                "us_unemployment_rate",
                "macro_available",
            ]
        )
        is True
    )

    assert (
        bool(
            feb_10.loc[
                "us_consumer_price_index",
                "macro_available",
            ]
        )
        is False
    )

    feb_15 = result[
        result["date"]
        == pd.Timestamp(
            "2020-02-15",
            tz="UTC",
        )
    ].set_index("series_id")

    assert (
        bool(
            feb_15.loc[
                "us_unemployment_rate",
                "macro_available",
            ]
        )
        is True
    )

    assert (
        bool(
            feb_15.loc[
                "us_consumer_price_index",
                "macro_available",
            ]
        )
        is True
    )


def test_dates_before_first_release_are_preserved():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-01-15",
            "2020-02-10",
        ],
        macro_vintages=frame,
    )

    assert len(result) == 2

    assert bool(result.iloc[0]["macro_available"]) is False

    assert bool(result.iloc[1]["macro_available"]) is True


def test_unsorted_input_produces_deterministic_output():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-02-01",
                available_at="2020-03-06",
                value=3.6,
            ),
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-03-10",
                value=3.7,
            ),
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-03-15",
            "2020-02-01",
        ],
        macro_vintages=frame,
    )

    assert list(result.columns) == MACRO_STATE_COLUMNS

    assert result["date"].is_monotonic_increasing

    march = result[
        result["date"]
        == pd.Timestamp(
            "2020-03-15",
            tz="UTC",
        )
    ].iloc[0]

    # February remains the current economic
    # observation even though January received
    # a later revision.
    assert march["observation_date"] == pd.Timestamp(
        "2020-02-01",
        tz="UTC",
    )

    assert march["value"] == pytest.approx(3.6)


def test_duplicate_decision_dates_are_collapsed():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    result = build_macro_state_asof(
        decision_dates=[
            "2020-02-10",
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
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="invalid",
    ):
        build_macro_state_asof(
            decision_dates=[
                "not-a-date",
            ],
            macro_vintages=frame,
        )


def test_empty_decision_dates_raise():
    frame = _macro_frame(
        [
            _macro_row(
                observation_date="2020-01-01",
                available_at="2020-02-07",
                value=3.5,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="At least one decision date",
    ):
        build_macro_state_asof(
            decision_dates=[],
            macro_vintages=frame,
        )
