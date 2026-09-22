from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from stock_agent.data.macro.schema import validate_macro_frame

UNEMPLOYMENT_SERIES_ID = "us_unemployment_rate"
CPI_SERIES_ID = "us_consumer_price_index"
GDP_SERIES_ID = "us_real_gdp"
FED_FUNDS_SERIES_ID = "us_fed_funds_rate"
M2_SERIES_ID = "us_m2_money_supply"

CORE_MACRO_SERIES = [
    UNEMPLOYMENT_SERIES_ID,
    CPI_SERIES_ID,
    GDP_SERIES_ID,
    FED_FUNDS_SERIES_ID,
    M2_SERIES_ID,
]

MACRO_FEATURE_COLUMNS = [
    "date",
    "macro_coverage_ratio",
    "us_unemployment_rate",
    "us_unemployment_change_1m_pp",
    "us_unemployment_change_3m_pp",
    "us_unemployment_age_days",
    "us_unemployment_available",
    "us_cpi_yoy",
    "us_cpi_yoy_change_1m",
    "us_cpi_age_days",
    "us_cpi_available",
    "us_real_gdp_yoy",
    "us_real_gdp_qoq_annualized",
    "us_real_gdp_age_days",
    "us_real_gdp_available",
    "us_fed_funds_rate",
    "us_fed_funds_change_1m_pp",
    "us_fed_funds_change_3m_pp",
    "us_fed_funds_age_days",
    "us_fed_funds_available",
    "us_m2_yoy",
    "us_m2_yoy_change_1m",
    "us_m2_age_days",
    "us_m2_available",
]


MacroObservation = dict[str, object]
MacroHistory = dict[pd.Timestamp, MacroObservation]
KnownMacroHistory = dict[str, MacroHistory]


def _normalize_decision_dates(
    decision_dates: Iterable[object],
) -> pd.DatetimeIndex:
    """
    Parse, normalize, deduplicate, and sort decision dates.

    All macro decisions use UTC calendar dates in Agent 1.
    """

    values = list(decision_dates)

    if not values:
        raise ValueError("At least one decision date is required.")

    parsed = pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed).any():
        raise ValueError("Decision dates contain invalid values.")

    normalized = pd.DatetimeIndex(parsed).normalize()

    return normalized.unique().sort_values()


def _prepare_macro_events(
    macro_vintages: pd.DataFrame,
) -> list[MacroObservation]:
    """
    Normalize and chronologically order macro vintage events.

    Sorting by availability first ensures that later revisions are
    introduced only once they were actually known.
    """

    validate_macro_frame(macro_vintages)

    frame = macro_vintages.copy()

    for column in [
        "observation_date",
        "available_at",
        "vintage_date",
    ]:
        frame[column] = pd.to_datetime(
            frame[column],
            errors="raise",
            utc=True,
        ).dt.normalize()

    frame = frame.sort_values(
        [
            "available_at",
            "vintage_date",
            "series_id",
            "observation_date",
        ]
    ).reset_index(drop=True)

    return frame.to_dict("records")


def _latest_observation(
    history: MacroHistory | None,
) -> MacroObservation | None:
    """Return the latest economic observation currently known."""

    if not history:
        return None

    latest_date = max(history)

    return history[latest_date]


def _value_at(
    history: MacroHistory | None,
    observation_date: pd.Timestamp,
) -> float:
    """
    Return the value for an exact economic period.

    We intentionally do not substitute a neighboring month/quarter
    when an expected historical observation is missing.
    """

    if not history:
        return float("nan")

    row = history.get(observation_date)

    if row is None:
        return float("nan")

    return float(row["value"])


def _safe_difference(
    current: float,
    previous: float,
) -> float:
    """Calculate an absolute difference when both values exist."""

    if pd.isna(current) or pd.isna(previous):
        return float("nan")

    return current - previous


def _safe_growth(
    current: float,
    previous: float,
) -> float:
    """
    Calculate percentage growth as a decimal.

    Example:
        103 / 100 - 1 = 0.03
    """

    if pd.isna(current) or pd.isna(previous):
        return float("nan")

    if previous == 0:
        return float("nan")

    return current / previous - 1.0


def _growth_for_observation(
    history: MacroHistory | None,
    observation_date: pd.Timestamp,
    lag_months: int,
) -> float:
    """Calculate date-aware growth against an exact prior period."""

    current = _value_at(
        history,
        observation_date,
    )

    previous_date = observation_date - pd.DateOffset(months=lag_months)

    previous = _value_at(
        history,
        previous_date,
    )

    return _safe_growth(
        current,
        previous,
    )


def _annualized_quarterly_growth(
    current: float,
    previous_quarter: float,
) -> float:
    """
    Convert quarter-over-quarter level growth to an annualized rate.

    Returned as a decimal:
        0.08 == 8%
    """

    if pd.isna(current) or pd.isna(previous_quarter):
        return float("nan")

    if current <= 0 or previous_quarter <= 0:
        return float("nan")

    return (current / previous_quarter) ** 4 - 1.0


def _observation_age_days(
    decision_date: pd.Timestamp,
    current: MacroObservation | None,
) -> float:
    """
    Measure age of the latest economic reference period.

    This differs from release age. It tells us how old the underlying
    economic period is on the decision date.
    """

    if current is None:
        return float("nan")

    observation_date = pd.Timestamp(current["observation_date"])

    return float((decision_date - observation_date).days)


def _unemployment_features(
    decision_date: pd.Timestamp,
    history: MacroHistory | None,
) -> dict[str, object]:
    current = _latest_observation(history)

    if current is None:
        return {
            "us_unemployment_rate": float("nan"),
            "us_unemployment_change_1m_pp": float("nan"),
            "us_unemployment_change_3m_pp": float("nan"),
            "us_unemployment_age_days": float("nan"),
            "us_unemployment_available": False,
        }

    observation_date = pd.Timestamp(current["observation_date"])

    current_value = float(current["value"])

    previous_1m = _value_at(
        history,
        observation_date - pd.DateOffset(months=1),
    )

    previous_3m = _value_at(
        history,
        observation_date - pd.DateOffset(months=3),
    )

    return {
        "us_unemployment_rate": current_value,
        "us_unemployment_change_1m_pp": (
            _safe_difference(
                current_value,
                previous_1m,
            )
        ),
        "us_unemployment_change_3m_pp": (
            _safe_difference(
                current_value,
                previous_3m,
            )
        ),
        "us_unemployment_age_days": (
            _observation_age_days(
                decision_date,
                current,
            )
        ),
        "us_unemployment_available": True,
    }


def _cpi_features(
    decision_date: pd.Timestamp,
    history: MacroHistory | None,
) -> dict[str, object]:
    current = _latest_observation(history)

    if current is None:
        return {
            "us_cpi_yoy": float("nan"),
            "us_cpi_yoy_change_1m": float("nan"),
            "us_cpi_age_days": float("nan"),
            "us_cpi_available": False,
        }

    observation_date = pd.Timestamp(current["observation_date"])

    yoy = _growth_for_observation(
        history,
        observation_date,
        lag_months=12,
    )

    previous_month = observation_date - pd.DateOffset(months=1)

    previous_yoy = _growth_for_observation(
        history,
        previous_month,
        lag_months=12,
    )

    return {
        "us_cpi_yoy": yoy,
        "us_cpi_yoy_change_1m": (
            _safe_difference(
                yoy,
                previous_yoy,
            )
        ),
        "us_cpi_age_days": (
            _observation_age_days(
                decision_date,
                current,
            )
        ),
        "us_cpi_available": True,
    }


def _gdp_features(
    decision_date: pd.Timestamp,
    history: MacroHistory | None,
) -> dict[str, object]:
    current = _latest_observation(history)

    if current is None:
        return {
            "us_real_gdp_yoy": float("nan"),
            "us_real_gdp_qoq_annualized": float("nan"),
            "us_real_gdp_age_days": float("nan"),
            "us_real_gdp_available": False,
        }

    observation_date = pd.Timestamp(current["observation_date"])

    current_value = float(current["value"])

    previous_quarter = _value_at(
        history,
        observation_date - pd.DateOffset(months=3),
    )

    return {
        "us_real_gdp_yoy": (
            _growth_for_observation(
                history,
                observation_date,
                lag_months=12,
            )
        ),
        "us_real_gdp_qoq_annualized": (
            _annualized_quarterly_growth(
                current_value,
                previous_quarter,
            )
        ),
        "us_real_gdp_age_days": (
            _observation_age_days(
                decision_date,
                current,
            )
        ),
        "us_real_gdp_available": True,
    }


def _fed_funds_features(
    decision_date: pd.Timestamp,
    history: MacroHistory | None,
) -> dict[str, object]:
    current = _latest_observation(history)

    if current is None:
        return {
            "us_fed_funds_rate": float("nan"),
            "us_fed_funds_change_1m_pp": float("nan"),
            "us_fed_funds_change_3m_pp": float("nan"),
            "us_fed_funds_age_days": float("nan"),
            "us_fed_funds_available": False,
        }

    observation_date = pd.Timestamp(current["observation_date"])

    current_value = float(current["value"])

    previous_1m = _value_at(
        history,
        observation_date - pd.DateOffset(months=1),
    )

    previous_3m = _value_at(
        history,
        observation_date - pd.DateOffset(months=3),
    )

    return {
        "us_fed_funds_rate": current_value,
        "us_fed_funds_change_1m_pp": (
            _safe_difference(
                current_value,
                previous_1m,
            )
        ),
        "us_fed_funds_change_3m_pp": (
            _safe_difference(
                current_value,
                previous_3m,
            )
        ),
        "us_fed_funds_age_days": (
            _observation_age_days(
                decision_date,
                current,
            )
        ),
        "us_fed_funds_available": True,
    }


def _m2_features(
    decision_date: pd.Timestamp,
    history: MacroHistory | None,
) -> dict[str, object]:
    current = _latest_observation(history)

    if current is None:
        return {
            "us_m2_yoy": float("nan"),
            "us_m2_yoy_change_1m": float("nan"),
            "us_m2_age_days": float("nan"),
            "us_m2_available": False,
        }

    observation_date = pd.Timestamp(current["observation_date"])

    yoy = _growth_for_observation(
        history,
        observation_date,
        lag_months=12,
    )

    previous_month = observation_date - pd.DateOffset(months=1)

    previous_yoy = _growth_for_observation(
        history,
        previous_month,
        lag_months=12,
    )

    return {
        "us_m2_yoy": yoy,
        "us_m2_yoy_change_1m": (
            _safe_difference(
                yoy,
                previous_yoy,
            )
        ),
        "us_m2_age_days": (
            _observation_age_days(
                decision_date,
                current,
            )
        ),
        "us_m2_available": True,
    }


def _build_feature_row(
    decision_date: pd.Timestamp,
    known_history: KnownMacroHistory,
) -> dict[str, object]:
    """Build one leakage-safe macro feature row."""

    unemployment = _unemployment_features(
        decision_date,
        known_history.get(UNEMPLOYMENT_SERIES_ID),
    )

    cpi = _cpi_features(
        decision_date,
        known_history.get(CPI_SERIES_ID),
    )

    gdp = _gdp_features(
        decision_date,
        known_history.get(GDP_SERIES_ID),
    )

    fed_funds = _fed_funds_features(
        decision_date,
        known_history.get(FED_FUNDS_SERIES_ID),
    )

    m2 = _m2_features(
        decision_date,
        known_history.get(M2_SERIES_ID),
    )

    availability = [
        bool(unemployment["us_unemployment_available"]),
        bool(cpi["us_cpi_available"]),
        bool(gdp["us_real_gdp_available"]),
        bool(fed_funds["us_fed_funds_available"]),
        bool(m2["us_m2_available"]),
    ]

    coverage_ratio = sum(availability) / len(CORE_MACRO_SERIES)

    return {
        "date": decision_date,
        "macro_coverage_ratio": (coverage_ratio),
        **unemployment,
        **cpi,
        **gdp,
        **fed_funds,
        **m2,
    }


def build_macro_features(
    decision_dates: Iterable[object],
    macro_vintages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build leakage-safe macro features for each decision date.

    Historical macro revisions are applied only after their
    ``available_at`` date.

    Growth features are returned as decimals:
        0.03 == 3%

    Rate changes are returned in percentage points:
        5.25 -> 5.00 == -0.25

    Exact economic-period matching is used for all lags. Missing
    months or quarters remain missing rather than being silently
    replaced by neighboring observations.
    """

    dates = _normalize_decision_dates(decision_dates)

    events = _prepare_macro_events(macro_vintages)

    known_history: KnownMacroHistory = {}

    event_index = 0
    rows: list[dict[str, object]] = []

    for decision_date in dates:
        while (
            event_index < len(events)
            and pd.Timestamp(events[event_index]["available_at"]) <= decision_date
        ):
            event = events[event_index]

            series_id = str(event["series_id"])

            observation_date = pd.Timestamp(event["observation_date"])

            series_history = known_history.setdefault(
                series_id,
                {},
            )

            # Later available vintages overwrite older versions of
            # the same economic observation only after they become
            # known.
            series_history[observation_date] = event

            event_index += 1

        rows.append(
            _build_feature_row(
                decision_date,
                known_history,
            )
        )

    return pd.DataFrame(
        rows,
        columns=MACRO_FEATURE_COLUMNS,
    )
