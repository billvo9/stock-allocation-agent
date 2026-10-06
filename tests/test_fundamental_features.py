from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.features import (
    build_quarterly_fundamental_features,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
)


def _fundamental_row(
    *,
    symbol: str,
    period_end: str,
    revenue: float | None = 100.0,
    gross_profit: float | None = 40.0,
    operating_income: float | None = 20.0,
    net_income: float | None = 15.0,
    diluted_eps: float | None = 1.0,
    total_assets: float | None = 500.0,
    total_debt: float | None = 100.0,
    free_cash_flow: float | None = 10.0,
) -> dict:
    period_timestamp = pd.Timestamp(
        period_end,
        tz="UTC",
    )

    filing_date = period_timestamp + pd.Timedelta(days=25)

    available_at = filing_date + pd.Timedelta(days=1)

    return {
        "symbol": symbol,
        "provider_symbol": symbol,
        "period_end": period_timestamp,
        "available_at": available_at,
        "retrieved_at": pd.Timestamp("2027-12-31T12:00:00Z"),
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "diluted_eps": diluted_eps,
        "diluted_average_shares": 100.0,
        "total_assets": total_assets,
        "total_debt": total_debt,
        "cash_and_cash_equivalents": 50.0,
        "inventory": 30.0,
        "stockholders_equity": 300.0,
        "operating_cash_flow": 25.0,
        "capital_expenditure": -15.0,
        "free_cash_flow": free_cash_flow,
        "filing_date": filing_date,
        "sec_form_type": "10-Q",
        "sec_accession_number": None,
        "availability_source": ("sec_filing_date_plus_1d"),
        "currency": "USD",
        "source": "yfinance",
    }


def _frame(
    rows: list[dict],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=QUARTERLY_FUNDAMENTAL_COLUMNS,
    )


def test_builds_margin_features():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=1000.0,
                gross_profit=400.0,
                operating_income=250.0,
                free_cash_flow=100.0,
            )
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    row = result.iloc[0]

    assert row["gross_margin"] == pytest.approx(0.40)

    assert row["operating_margin"] == pytest.approx(0.25)

    assert row["fcf_margin"] == pytest.approx(0.10)


def test_builds_debt_to_assets():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                total_debt=100.0,
                total_assets=500.0,
            )
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    assert result.loc[
        0,
        "debt_to_assets",
    ] == pytest.approx(0.20)


def test_revenue_yoy_matches_same_period_last_year():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-06-30",
                revenue=110.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-09-30",
                revenue=115.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-12-31",
                revenue=118.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z")].iloc[0]

    assert current["revenue_growth_yoy"] == pytest.approx(0.20)


def test_missing_prior_quarter_does_not_use_wrong_quarter():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-06-30",
                revenue=110.0,
            ),
            # 2025-09-30 intentionally missing.
            _fundamental_row(
                symbol="MU",
                period_end="2025-12-31",
                revenue=130.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-06-30",
                revenue=140.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-09-30",
                revenue=150.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-09-30T00:00:00Z")].iloc[0]

    assert pd.isna(current["revenue_growth_yoy"])


def test_two_missing_prior_quarters_remain_missing():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-06-30",
                revenue=110.0,
            ),
            # 2025 Q3 and Q4 intentionally missing.
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-06-30",
                revenue=130.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-09-30",
                revenue=140.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-12-31",
                revenue=150.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    q3 = result[result["period_end"] == pd.Timestamp("2026-09-30T00:00:00Z")].iloc[0]

    q4 = result[result["period_end"] == pd.Timestamp("2026-12-31T00:00:00Z")].iloc[0]

    assert pd.isna(q3["revenue_growth_yoy"])

    assert pd.isna(q4["revenue_growth_yoy"])


def test_yoy_growth_is_independent_by_symbol():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="NVDA",
                period_end="2025-03-31",
                revenue=200.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
            _fundamental_row(
                symbol="NVDA",
                period_end="2026-03-31",
                revenue=300.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    mu = result[
        (result["symbol"] == "MU") & (result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z"))
    ].iloc[0]

    nvda = result[
        (result["symbol"] == "NVDA")
        & (result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z"))
    ].iloc[0]

    assert mu["revenue_growth_yoy"] == pytest.approx(0.20)

    assert nvda["revenue_growth_yoy"] == pytest.approx(0.50)


def test_zero_denominators_return_nan():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=0.0,
                gross_profit=10.0,
                operating_income=5.0,
                free_cash_flow=2.0,
                total_debt=50.0,
                total_assets=0.0,
            )
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    row = result.iloc[0]

    assert pd.isna(row["gross_margin"])

    assert pd.isna(row["operating_margin"])

    assert pd.isna(row["fcf_margin"])

    assert pd.isna(row["debt_to_assets"])


def test_negative_to_positive_eps_is_positive_growth():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                diluted_eps=-1.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                diluted_eps=1.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z")].iloc[0]

    assert current["eps_growth_yoy"] == pytest.approx(2.0)


def test_yoy_allows_small_fiscal_period_shift():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-04-10",
                revenue=120.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-04-10T00:00:00Z")].iloc[0]

    assert current["revenue_growth_yoy"] == pytest.approx(0.20)


def test_yoy_rejects_large_fiscal_period_shift():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-04-25",
                revenue=120.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-04-25T00:00:00Z")].iloc[0]

    assert pd.isna(current["revenue_growth_yoy"])


# ---------------------------------------------------------------------------
# Point-in-time guard for YoY prior-year matching.
#
# Rule: a current row may only use prior-year rows of the same symbol whose
# available_at is known and <= the current row's available_at. Among those,
# period matching is unchanged; ties on the matched period_end resolve to the
# latest available_at; same-time conflicting values give NaN.
# ---------------------------------------------------------------------------


def _versioned_row(
    *,
    period_end: str,
    available_at: str | None,
    revenue: float | None,
    accession: str,
    form_type: str = "10-Q",
    symbol: str = "MU",
) -> dict:
    row = _fundamental_row(
        symbol=symbol,
        period_end=period_end,
        revenue=revenue,
    )

    row["available_at"] = pd.NaT if available_at is None else pd.Timestamp(available_at, tz="UTC")

    row["sec_accession_number"] = accession

    row["sec_form_type"] = form_type

    return row


def _growth_by_accession(
    result: pd.DataFrame,
) -> dict[str, float]:
    return dict(
        zip(
            result["sec_accession_number"],
            result["revenue_growth_yoy"],
            strict=True,
        )
    )


def _t10_rows() -> list[dict]:
    return [
        # Prior-year original 10-Q.
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-04-26",
            revenue=100.0,
            accession="prior-original",
        ),
        # Current-quarter original 10-Q.
        _versioned_row(
            period_end="2026-03-31",
            available_at="2026-04-26",
            revenue=120.0,
            accession="current-original",
        ),
        # Prior-year 10-Q/A restatement, published AFTER the current 10-Q.
        _versioned_row(
            period_end="2025-03-31",
            available_at="2026-06-01",
            revenue=80.0,
            accession="prior-amendment",
            form_type="10-Q/A",
        ),
        # Later current-quarter version, available after the restatement.
        _versioned_row(
            period_end="2026-03-31",
            available_at="2026-07-01",
            revenue=120.0,
            accession="current-later",
            form_type="10-Q/A",
        ),
    ]


def test_t10_later_prior_year_amendment_does_not_leak_into_current_yoy():
    result = build_quarterly_fundamental_features(_frame(_t10_rows()))

    growth = _growth_by_accession(result)

    # Only the original prior-year value (100) was public on 2026-04-26.
    assert growth["current-original"] == pytest.approx(0.20)

    # The later current version may use the restated prior-year value (80).
    assert growth["current-later"] == pytest.approx(0.50)


def test_yoy_uses_latest_eligible_prior_year_version():
    rows = [
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-04-26",
            revenue=100.0,
            accession="prior-v1",
        ),
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-08-01",
            revenue=96.0,
            accession="prior-v2",
            form_type="10-Q/A",
        ),
        _versioned_row(
            period_end="2026-03-31",
            available_at="2026-04-26",
            revenue=120.0,
            accession="current",
        ),
    ]

    result = build_quarterly_fundamental_features(_frame(rows))

    assert _growth_by_accession(result)["current"] == pytest.approx(0.25)


def test_yoy_same_time_conflicting_prior_year_values_are_nan():
    rows = [
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-04-26",
            revenue=100.0,
            accession="prior-a",
        ),
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-04-26",
            revenue=90.0,
            accession="prior-b",
        ),
        _versioned_row(
            period_end="2026-03-31",
            available_at="2026-04-26",
            revenue=120.0,
            accession="current",
        ),
    ]

    result = build_quarterly_fundamental_features(_frame(rows))

    assert pd.isna(_growth_by_accession(result)["current"])


def test_yoy_same_time_equal_prior_year_values_are_used():
    rows = [
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-04-26",
            revenue=100.0,
            accession="prior-a",
        ),
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-04-26",
            revenue=100.0,
            accession="prior-b",
        ),
        _versioned_row(
            period_end="2026-03-31",
            available_at="2026-04-26",
            revenue=120.0,
            accession="current",
        ),
    ]

    result = build_quarterly_fundamental_features(_frame(rows))

    assert _growth_by_accession(result)["current"] == pytest.approx(0.20)


def test_yoy_unknown_prior_year_availability_is_not_eligible():
    rows = [
        _versioned_row(
            period_end="2025-03-31",
            available_at=None,
            revenue=100.0,
            accession="prior-unknown",
        ),
        _versioned_row(
            period_end="2026-03-31",
            available_at="2026-04-26",
            revenue=120.0,
            accession="current",
        ),
    ]

    result = build_quarterly_fundamental_features(_frame(rows))

    assert pd.isna(_growth_by_accession(result)["current"])


def test_yoy_unknown_current_availability_yields_nan():
    rows = [
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-04-26",
            revenue=100.0,
            accession="prior",
        ),
        _versioned_row(
            period_end="2026-03-31",
            available_at=None,
            revenue=120.0,
            accession="current-unknown",
        ),
    ]

    result = build_quarterly_fundamental_features(_frame(rows))

    assert pd.isna(_growth_by_accession(result)["current-unknown"])


def test_yoy_growth_is_independent_of_input_row_order():
    rows = _t10_rows() + [
        _versioned_row(
            period_end="2025-03-31",
            available_at="2025-08-01",
            revenue=95.0,
            accession="prior-v2",
            form_type="10-Q/A",
        ),
        _versioned_row(
            period_end="2025-06-30",
            available_at="2025-07-26",
            revenue=110.0,
            accession="prior-q2",
        ),
        _versioned_row(
            period_end="2026-06-30",
            available_at="2026-07-26",
            revenue=121.0,
            accession="current-q2",
        ),
    ]

    baseline = _growth_by_accession(build_quarterly_fundamental_features(_frame(rows)))

    assert baseline["current-original"] == pytest.approx(120.0 / 95.0 - 1.0)
    assert baseline["current-later"] == pytest.approx(0.50)
    assert baseline["current-q2"] == pytest.approx(0.10)

    frame = _frame(rows)

    for seed in range(5):
        shuffled = frame.sample(frac=1.0, random_state=seed)

        growth = _growth_by_accession(build_quarterly_fundamental_features(shuffled))

        assert growth.keys() == baseline.keys()

        for accession, value in baseline.items():
            if pd.isna(value):
                assert pd.isna(growth[accession])
            else:
                assert growth[accession] == pytest.approx(value)
