"""
Feature missingness diagnostics, by reason, scope and fold.

Every missing feature value gets one machine-readable reason, from its
contract:

    warmup              session_close: the symbol has no more than
                        `lookback_sessions` price rows up to the date
    not_yet_available   per_row_timestamp: no observation was available yet
                        (the availability column is empty)
    source_missing      the inputs existed but gave no value

Scopes: "panel" (all rows, also per symbol), "training" (one fold's training
rows) and "evaluation" (one fold's test rows). A rate above
`warning_rate` gets status "warning" (high_missing_rate); nothing is
imputed, dropped or modified here. Missingness never changes membership or
labels (frozen_universe.assert_membership_independent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from stock_agent.features.contracts import PER_ROW_TIMESTAMP, FeatureContract
from stock_agent.model_validation.folds import WalkForwardFold

REASONS = ("warmup", "not_yet_available", "source_missing")
COLUMNS = (
    "scope",
    "fold_id",
    "group",
    "feature",
    "family",
    "policy",
    "n_rows",
    "n_missing",
    "missing_rate",
    *(f"n_{reason}" for reason in REASONS),
    "status",
    "reason",
)


def missing_reasons(
    frame: pd.DataFrame,
    contract: FeatureContract,
    *,
    history_column: str = "history_sessions",
) -> pd.Series:
    """Per row: None when present, else one of REASONS."""

    values = frame[contract.name].to_numpy(dtype=np.float64)
    missing = np.isnan(values)
    if contract.availability == PER_ROW_TIMESTAMP:
        absent = frame[contract.availability_column].isna().to_numpy()
        reason = np.where(absent, "not_yet_available", "source_missing")
    else:
        if history_column not in frame.columns:
            raise ValueError(f"Missingness reasons need the {history_column!r} column.")
        short = frame[history_column].to_numpy() <= contract.lookback_sessions
        reason = np.where(short, "warmup", "source_missing")
    return pd.Series(np.where(missing, reason, None), index=frame.index, dtype="object")


def _rows(scope, fold_id, group, contract, reasons: pd.Series, warning_rate: float) -> dict:
    n = len(reasons)
    counts = reasons.value_counts()
    n_missing = int(reasons.notna().sum())
    rate = n_missing / n if n else float("nan")
    high = n > 0 and rate > warning_rate
    return {
        "scope": scope,
        "fold_id": fold_id,
        "group": group,
        "feature": contract.name,
        "family": contract.family,
        "policy": contract.missingness,
        "n_rows": n,
        "n_missing": n_missing,
        "missing_rate": rate,
        **{f"n_{reason}": int(counts.get(reason, 0)) for reason in REASONS},
        "status": "warning" if high else ("ok" if n else "unavailable"),
        "reason": "high_missing_rate" if high else (None if n else "no_rows"),
    }


def missingness_table(
    frame: pd.DataFrame,
    contracts: Mapping[str, FeatureContract],
    *,
    folds: Sequence[WalkForwardFold] = (),
    warning_rate: float = 0.25,
    history_column: str = "history_sessions",
) -> pd.DataFrame:
    """Tidy missingness counts per feature, scope, fold and group."""

    rows = []
    for contract in contracts.values():
        reasons = missing_reasons(frame, contract, history_column=history_column)
        rows.append(_rows("panel", None, "all", contract, reasons, warning_rate))
        for symbol, part in reasons.groupby(frame["symbol"], sort=True):
            rows.append(_rows("panel", None, f"symbol={symbol}", contract, part, warning_rate))
        for fold in folds:
            for scope, positions in (("training", fold.train_rows), ("evaluation", fold.test_rows)):
                part = reasons.iloc[positions]
                rows.append(_rows(scope, fold.fold_id, "all", contract, part, warning_rate))
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    return table.astype({"fold_id": "Int64", "missing_rate": "float64"})
