import pandas as pd
import pytest

from stock_agent.universe.availability import (
    build_feature_availability,
)


def test_builds_feature_availability_flags():
    frame = pd.DataFrame(
        {
            "symbol": [
                "MU",
                "SNDK",
            ],
            "beta": [
                2.1,
                None,
            ],
            "gross_margin": [
                0.40,
                0.35,
            ],
            "revenue_growth_yoy": [
                None,
                0.25,
            ],
        }
    )

    result = build_feature_availability(
        frame,
        feature_columns=[
            "beta",
            "gross_margin",
            "revenue_growth_yoy",
        ],
    )

    assert result["beta_available"].tolist() == [
        True,
        False,
    ]

    assert result["gross_margin_available"].tolist() == [
        True,
        True,
    ]

    assert result["revenue_growth_yoy_available"].tolist() == [
        False,
        True,
    ]

    assert result["available_feature_count"].tolist() == [
        2,
        2,
    ]

    assert result["available_feature_fraction"].tolist() == pytest.approx(
        [
            2 / 3,
            2 / 3,
        ]
    )


def test_provider_missing_tokens_are_unavailable():
    frame = pd.DataFrame(
        {
            "forward_pe": [
                "N/A",
                ".",
                "",
                "   ",
                None,
                15.0,
            ]
        }
    )

    result = build_feature_availability(
        frame,
        feature_columns=[
            "forward_pe",
        ],
    )

    assert result["forward_pe_available"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_any_and_all_availability_flags():
    frame = pd.DataFrame(
        {
            "beta": [
                1.5,
                None,
                None,
            ],
            "gross_margin": [
                0.40,
                0.30,
                None,
            ],
        }
    )

    result = build_feature_availability(
        frame,
        feature_columns=[
            "beta",
            "gross_margin",
        ],
    )

    assert result["any_requested_feature_available"].tolist() == [
        True,
        True,
        False,
    ]

    assert result["all_requested_features_available"].tolist() == [
        True,
        False,
        False,
    ]


def test_missing_requested_feature_raises():
    frame = pd.DataFrame(
        {
            "beta": [
                1.5,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="missing requested features",
    ):
        build_feature_availability(
            frame,
            feature_columns=[
                "beta",
                "gross_margin",
            ],
        )


def test_empty_feature_list_raises():
    frame = pd.DataFrame(
        {
            "beta": [
                1.5,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        build_feature_availability(
            frame,
            feature_columns=[],
        )


def test_duplicate_feature_names_raise():
    frame = pd.DataFrame(
        {
            "beta": [
                1.5,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        build_feature_availability(
            frame,
            feature_columns=[
                "beta",
                "beta",
            ],
        )


def test_input_frame_is_not_modified():
    frame = pd.DataFrame(
        {
            "symbol": [
                "MU",
                "SNDK",
            ],
            "beta": [
                2.1,
                None,
            ],
        }
    )

    original = frame.copy(deep=True)

    result = build_feature_availability(
        frame,
        feature_columns=[
            "beta",
        ],
    )

    pd.testing.assert_frame_equal(
        frame,
        original,
    )

    assert result["symbol"].tolist() == [
        "MU",
        "SNDK",
    ]


def test_existing_generated_column_raises():
    frame = pd.DataFrame(
        {
            "beta": [
                1.5,
            ],
            "beta_available": [
                True,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="overwrite",
    ):
        build_feature_availability(
            frame,
            feature_columns=[
                "beta",
            ],
        )
