"""
Feature contracts: every model feature declares when it is known and what
happens when it is missing (docs/research/ml_features.md).

A contract names the feature's family, its availability rule, its lookback,
its missingness policy and its inputs. `validate_feature_contracts` refuses
any feature without a contract (no availability semantics, no feature), then
checks the frame against each contract:

    session_close       computed from same-symbol prices and volumes through
                        the close of the row date d (backward-looking only;
                        prefix stability is tested per builder). This is
                        the DECISION_CONVENTION of model_validation.audit:
                        features include the close of d, labels enter at the
                        close of d + entry_lag.
    per_row_timestamp   every value carries the availability timestamp of
                        the observation it came from, in `availability_column`;
                        a value may be used on row date d only if
                        available_at <= d at 00:00 UTC, the existing panel
                        rule (features.panel, "fundamentals and the panel
                        validator use 00:00 UTC on d"). A value without a
                        timestamp is refused.

Missingness policies (rows are never dropped: membership and labels stay
independent of feature missingness, owner decision 2026-10-13):

    impute_training_median  the estimator imputes with its fit rows' medians
    native_missing          the estimator handles NaN itself (gradient boosting)
    refuse                  a missing value is a contract violation

Validation never modifies data. Leakage violations raise LeakageError;
other contract violations raise FeatureContractError. Both list every
problem found.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.model_validation.audit import LeakageError
from stock_agent.model_validation.checks import validate_feature_columns

FAMILIES = ("relative_technical", "fundamental_quality", "risk_liquidity")
SESSION_CLOSE = "session_close"
PER_ROW_TIMESTAMP = "per_row_timestamp"
AVAILABILITY_RULES = (SESSION_CLOSE, PER_ROW_TIMESTAMP)
MISSINGNESS_POLICIES = ("impute_training_median", "native_missing", "refuse")
CROSS_SECTIONAL = ("none", "within_date_rank")


class FeatureContractError(ValueError):
    """A feature has no contract, or its contract is invalid or violated."""


@dataclass(frozen=True)
class FeatureContract:
    """
    Availability and missingness semantics of one model feature.

    `lookback_sessions` is the number of earlier sessions of the same symbol
    the value needs besides the row's own (a 20-session return needs 20);
    before that much history exists the value is missing for reason warmup.
    `cross_sectional` = within_date_rank means the value is ranked among the
    eligible names of the same date only (never across dates).
    """

    name: str
    family: str
    availability: str
    lookback_sessions: int
    missingness: str
    inputs: tuple[str, ...]
    description: str
    availability_column: str | None = None
    cross_sectional: str = "none"

    def __post_init__(self) -> None:
        problems = []
        if self.family not in FAMILIES:
            problems.append(f"family {self.family!r} not in {FAMILIES}")
        if self.availability not in AVAILABILITY_RULES:
            problems.append(f"availability {self.availability!r} not in {AVAILABILITY_RULES}")
        if self.missingness not in MISSINGNESS_POLICIES:
            problems.append(f"missingness {self.missingness!r} not in {MISSINGNESS_POLICIES}")
        if self.cross_sectional not in CROSS_SECTIONAL:
            problems.append(f"cross_sectional {self.cross_sectional!r} not in {CROSS_SECTIONAL}")
        if (
            isinstance(self.lookback_sessions, bool)
            or not isinstance(self.lookback_sessions, int)
            or self.lookback_sessions < 0
        ):
            problems.append("lookback_sessions must be a non-negative integer")
        if (self.availability == PER_ROW_TIMESTAMP) != (self.availability_column is not None):
            problems.append("availability_column is required for per_row_timestamp, and only then")
        if not self.inputs or not self.description:
            problems.append("inputs and description are required")
        if problems:
            raise FeatureContractError(f"Contract {self.name!r}: " + "; ".join(problems))

    def describe(self) -> dict[str, object]:
        """JSON-safe description, recorded with every experiment."""

        return {
            "name": self.name,
            "family": self.family,
            "availability": self.availability,
            "availability_column": self.availability_column,
            "lookback_sessions": self.lookback_sessions,
            "missingness": self.missingness,
            "cross_sectional": self.cross_sectional,
            "inputs": list(self.inputs),
            "description": self.description,
        }


def contract_index(contracts: Sequence[FeatureContract]) -> dict[str, FeatureContract]:
    names = [contract.name for contract in contracts]
    if len(names) != len(set(names)):
        raise FeatureContractError("Feature contracts must have unique names.")
    return {contract.name: contract for contract in contracts}


def validate_feature_contracts(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    contracts: Mapping[str, FeatureContract],
    *,
    estimator_handles_missing: bool = False,
) -> list[str]:
    """
    Return `feature_columns` if every feature has a contract and the frame
    honours it; raise otherwise.

    `estimator_handles_missing` must be True for any feature whose policy is
    native_missing (the estimator receives NaN as is).
    """

    columns = list(feature_columns)
    uncontracted = [column for column in columns if column not in contracts]
    if uncontracted:
        raise FeatureContractError(
            f"Features without a contract (no availability semantics): {uncontracted}. "
            "Every model feature must declare when it is known and how missing values "
            "are handled."
        )
    validate_feature_columns(frame, columns)  # names, keys, labels, levels, dtypes

    leaks, problems = [], []
    dates = pd.to_datetime(frame["date"], utc=True)
    for column in columns:
        contract = contracts[column]
        if contract.name != column:
            problems.append(f"{column}: indexed under another contract's name {contract.name!r}")
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        present = ~np.isnan(values)
        if np.isinf(values).any():
            problems.append(f"{column}: infinite values")
        if contract.missingness == "refuse" and not present.all():
            problems.append(f"{column}: {int((~present).sum())} missing values (policy refuse)")
        if contract.missingness == "native_missing" and not estimator_handles_missing:
            problems.append(f"{column}: policy native_missing needs an estimator that handles NaN")
        if contract.availability != PER_ROW_TIMESTAMP:
            continue
        stamp_column = contract.availability_column
        if stamp_column not in frame.columns:
            problems.append(f"{column}: availability column {stamp_column!r} is missing")
            continue
        stamps = frame[stamp_column]
        if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
            problems.append(f"{column}: {stamp_column} must be timezone-aware datetimes")
            continue
        stamps = stamps.dt.tz_convert("UTC")
        unstamped = present & stamps.isna().to_numpy()
        if unstamped.any():
            leaks.append(f"{column}: {int(unstamped.sum())} values without an availability time")
        late = present & (stamps > dates).fillna(False).to_numpy()
        if late.any():
            first = int(np.flatnonzero(late)[0])
            leaks.append(
                f"{column}: {int(late.sum())} values used before they were available "
                f"(e.g. date {dates.iloc[first]}, available_at {stamps.iloc[first]})"
            )
    if leaks:
        raise LeakageError("Feature availability violated: " + "; ".join(leaks))
    if problems:
        raise FeatureContractError("Feature contracts violated: " + "; ".join(problems))
    return columns
