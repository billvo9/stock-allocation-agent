from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.universe.eligibility import (
    build_dynamic_universe,
)


def _panel(
    rows: list[dict],
) -> pd.DataFrame:
    normalized_rows = [
        {
            "adjusted_close": 100.0,
            **row,
        }
        for row in rows
    ]

    frame = pd.DataFrame(normalized_rows)

    frame["date"] = pd.to_datetime(
        frame["date"],
        utc=True,
    )

    return frame


def test_old_asset_becomes_investable_after_minimum_observations():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-05",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-06",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-07",
                "symbol": "MU",
                "fundamental_available": True,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=3,
    )

    assert result["price_history_observations"].tolist() == [
        1,
        2,
        3,
        4,
    ]

    assert result["has_required_price_history"].tolist() == [
        False,
        False,
        True,
        True,
    ]

    assert result["is_investable"].tolist() == [
        False,
        False,
        True,
        True,
    ]


def test_new_asset_is_preserved_before_it_becomes_investable():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
            {
                "date": "2026-01-05",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=3,
    )

    assert len(result) == 2

    assert result["symbol"].tolist() == [
        "SNDK",
        "SNDK",
    ]

    assert result["price_history_observations"].tolist() == [
        1,
        2,
    ]

    assert not result["is_investable"].any()


def test_adding_new_asset_does_not_truncate_existing_asset_history():
    frame = _panel(
        [
            {
                "date": "2020-01-02",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2020-01-03",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2020-01-06",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2025-02-24",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
            {
                "date": "2025-02-25",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=2,
    )

    mu = result[result["symbol"] == "MU"].sort_values("date").reset_index(drop=True)

    assert len(mu) == 3

    assert mu.loc[
        0,
        "date",
    ] == pd.Timestamp("2020-01-02T00:00:00Z")

    assert mu["price_history_observations"].tolist() == [
        1,
        2,
        3,
    ]


def test_eligibility_is_calculated_independently_by_symbol():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-02",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
            {
                "date": "2026-01-05",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-06",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-06",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=3,
    )

    mu = result[result["symbol"] == "MU"].sort_values("date").reset_index(drop=True)

    sndk = result[result["symbol"] == "SNDK"].sort_values("date").reset_index(drop=True)

    assert mu["price_history_observations"].tolist() == [
        1,
        2,
        3,
    ]

    assert sndk["price_history_observations"].tolist() == [
        1,
        2,
    ]

    assert bool(mu.iloc[-1]["is_investable"])

    assert not bool(sndk.iloc[-1]["is_investable"])


def test_missing_fundamentals_do_not_automatically_block_investment():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
            {
                "date": "2026-01-05",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
            {
                "date": "2026-01-06",
                "symbol": "SNDK",
                "fundamental_available": False,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=3,
    )

    latest = result.sort_values("date").iloc[-1]

    assert not bool(latest["fundamental_available"])

    assert bool(latest["has_required_price_history"])

    assert bool(latest["is_investable"])


def test_price_history_count_uses_chronological_history_only():
    # Deliberately give the rows out of order.
    #
    # This prevents the implementation from simply
    # counting rows in input order, which could leak
    # future observations into historical dates.
    frame = _panel(
        [
            {
                "date": "2026-01-06",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-05",
                "symbol": "MU",
                "fundamental_available": True,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=3,
    )

    result = result.sort_values("date").reset_index(drop=True)

    assert result["price_history_observations"].tolist() == [
        1,
        2,
        3,
    ]

    assert result["is_investable"].tolist() == [
        False,
        False,
        True,
    ]


def test_duplicate_symbol_date_raises():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "fundamental_available": True,
            },
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "fundamental_available": True,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        build_dynamic_universe(
            frame,
            min_price_observations=2,
        )


def test_missing_price_does_not_count_as_price_history():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "adjusted_close": 100.0,
                "fundamental_available": True,
            },
            {
                "date": "2026-01-05",
                "symbol": "MU",
                "adjusted_close": None,
                "fundamental_available": True,
            },
            {
                "date": "2026-01-06",
                "symbol": "MU",
                "adjusted_close": 102.0,
                "fundamental_available": True,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=3,
    )

    result = result.sort_values("date").reset_index(drop=True)

    assert result["price_history_observations"].tolist() == [
        1,
        1,
        2,
    ]

    assert not result["is_investable"].any()


def test_nonpositive_price_is_not_usable_history():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "adjusted_close": 100.0,
                "fundamental_available": True,
            },
            {
                "date": "2026-01-05",
                "symbol": "MU",
                "adjusted_close": 0.0,
                "fundamental_available": True,
            },
            {
                "date": "2026-01-06",
                "symbol": "MU",
                "adjusted_close": -5.0,
                "fundamental_available": True,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=2,
    )

    assert result["price_history_observations"].tolist() == [
        1,
        1,
        1,
    ]

    assert not result["is_investable"].any()


def test_missing_price_for_one_asset_does_not_affect_another():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "adjusted_close": 100.0,
                "fundamental_available": True,
            },
            {
                "date": "2026-01-05",
                "symbol": "MU",
                "adjusted_close": 101.0,
                "fundamental_available": True,
            },
            {
                "date": "2026-01-02",
                "symbol": "SNDK",
                "adjusted_close": None,
                "fundamental_available": False,
            },
            {
                "date": "2026-01-05",
                "symbol": "SNDK",
                "adjusted_close": 50.0,
                "fundamental_available": False,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=2,
    )

    mu = result[result["symbol"] == "MU"].sort_values("date")

    sndk = result[result["symbol"] == "SNDK"].sort_values("date")

    assert mu["price_history_observations"].tolist() == [
        1,
        2,
    ]

    assert sndk["price_history_observations"].tolist() == [
        0,
        1,
    ]

    assert bool(mu.iloc[-1]["is_investable"])

    assert not bool(sndk.iloc[-1]["is_investable"])


def test_malformed_price_does_not_crash_or_count_as_history():
    frame = _panel(
        [
            {
                "date": "2026-01-02",
                "symbol": "MU",
                "adjusted_close": 100.0,
            },
            {
                "date": "2026-01-05",
                "symbol": "MU",
                "adjusted_close": "N/A",
            },
            {
                "date": "2026-01-06",
                "symbol": "MU",
                "adjusted_close": ".",
            },
            {
                "date": "2026-01-07",
                "symbol": "MU",
                "adjusted_close": 103.0,
            },
        ]
    )

    result = build_dynamic_universe(
        frame,
        min_price_observations=2,
    )

    assert result["price_history_observations"].tolist() == [
        1,
        1,
        1,
        2,
    ]

    assert result["is_investable"].tolist() == [
        False,
        False,
        False,
        True,
    ]
