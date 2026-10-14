"""
Batch validation shared by the ingest writer and the reader.

Deterministic checks only; nothing is repaired, dropped or filled. A failure
refuses the whole batch with a machine-readable reason. Expected data gaps
(sessions with no close) are not failures: they are reported as structured
missing-session diagnostics.

Order: request (the lockbox cap, coverage), table schemas and values, batch
ids, the lockbox on every date column, requested funds and range, calendar
sessions, retrieval order, then the missing-session report.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.funds.schema import (
    DATA_TABLES,
    DATE,
    DISTRIBUTIONS,
    MISSING_SESSIONS,
    OPTIONAL_DATE,
    PRICES,
    SPLITS,
    BatchRecord,
    FundDataRefused,
    RefusalReason,
    check_id,
    conform,
    refuse,
    utc_date,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

MISSING_STATUS = "unavailable"
MISSING_REASON = "missing_session"
MISSING_DETAIL = "no close for this NYSE session in the requested range; not repaired"

_ONE_DAY = pd.Timedelta(days=1)
# The column that places each data row inside the requested range.
_RANGE_COLUMNS = {
    PRICES.name: "session_date",
    DISTRIBUTIONS.name: "ex_date",
    SPLITS.name: "effective_date",
}


def validate_request(
    fund_ids: Sequence[str], start: object, end_exclusive: object
) -> tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]:
    """
    A request for [start, end_exclusive) as session dates. The cap is the
    lockbox: `end_exclusive` may not pass `MODEL_HOLDOUT_START`. Adapters
    call this before retrieving anything.
    """

    if isinstance(fund_ids, str) or not isinstance(fund_ids, Sequence) or not fund_ids:
        raise refuse(RefusalReason.INVALID_REQUEST, "fund_ids must be a non-empty sequence")
    try:
        ids = tuple(check_id("fund_id", fund_id) for fund_id in fund_ids)
    except FundDataRefused as error:
        raise refuse(RefusalReason.INVALID_REQUEST, error.detail) from error
    if len(set(ids)) != len(ids):
        raise refuse(RefusalReason.INVALID_REQUEST, f"fund_ids repeat: {list(ids)}")
    first = utc_date("start", start)
    end = utc_date("end_exclusive", end_exclusive)
    if end > MODEL_HOLDOUT_START:
        raise refuse(
            RefusalReason.REQUEST_AFTER_CAP,
            f"end_exclusive {end.date()} is after the holdout start {MODEL_HOLDOUT_START.date()}",
        )
    if first < nyse.COVERAGE.first or end - _ONE_DAY > nyse.COVERAGE.last:
        raise refuse(
            RefusalReason.OUTSIDE_COVERAGE,
            f"[{first.date()}, {end.date()}) is outside the calendar {nyse.CALENDAR_ID}",
        )
    if first >= end:
        raise refuse(RefusalReason.INVALID_REQUEST, f"start {first.date()} >= end_exclusive")
    return ids, first, end


def requested_sessions(record: BatchRecord) -> pd.DatetimeIndex:
    try:
        return nyse.sessions(record.start, record.end_exclusive - _ONE_DAY)
    except nyse.CalendarRefused as error:  # validate_request runs first; kept fail closed
        raise refuse(RefusalReason.OUTSIDE_COVERAGE, str(error)) from error


def missing_sessions(record: BatchRecord, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per requested fund and requested session with no price row."""

    expected = pd.MultiIndex.from_product(
        [list(record.fund_ids), requested_sessions(record)], names=["fund_id", "session_date"]
    )
    present = pd.MultiIndex.from_frame(prices[["fund_id", "session_date"]])
    missing = expected[~expected.isin(present)].to_frame(index=False)
    missing["status"] = MISSING_STATUS
    missing["reason"] = MISSING_REASON
    missing["detail"] = MISSING_DETAIL
    missing["batch_id"] = record.batch_id
    return conform(MISSING_SESSIONS, missing)


def _check_lockbox(tables: dict[str, pd.DataFrame]) -> None:
    for spec in DATA_TABLES:
        for column in spec.columns:
            if column.kind not in (DATE, OPTIONAL_DATE):
                continue
            dates = tables[spec.name][column.name]
            late = dates[dates >= MODEL_HOLDOUT_START]
            if len(late):
                raise refuse(
                    RefusalReason.LOCKBOX_VIOLATION,
                    f"{spec.name}.{column.name} has {len(late)} date(s) on or after "
                    f"{MODEL_HOLDOUT_START.date()}, first {late.min().date()}; batch refused",
                )


def validate_batch(
    record: BatchRecord,
    prices: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """The conformed tables and the missing-session report, or a refusal."""

    validate_request(record.fund_ids, record.start, record.end_exclusive)
    frames = {PRICES.name: prices, DISTRIBUTIONS.name: distributions, SPLITS.name: splits}
    tables = {spec.name: conform(spec, frames[spec.name]) for spec in DATA_TABLES}

    for name, table in tables.items():
        foreign = sorted(set(table["batch_id"]) - {record.batch_id})
        if foreign:
            raise refuse(RefusalReason.UNKNOWN_BATCH, f"{name} rows name batch {foreign[:3]}")
    _check_lockbox(tables)

    sessions = requested_sessions(record)
    for name, column in _RANGE_COLUMNS.items():
        table = tables[name]
        unrequested = sorted(set(table["fund_id"]) - set(record.fund_ids))
        if unrequested:
            raise refuse(RefusalReason.UNREQUESTED_FUND, f"{name} has funds {unrequested[:5]}")
        dates = table[column]
        outside = (dates < record.start) | (dates >= record.end_exclusive)
        if outside.any():
            raise refuse(
                RefusalReason.OUTSIDE_REQUEST,
                f"{name}.{column} {dates[outside].min().date()} is outside the request",
            )
        off = ~dates.isin(sessions)
        if off.any():
            raise refuse(
                RefusalReason.NOT_A_SESSION,
                f"{name}.{column} {dates[off].min().date()} is not a session of {nyse.CALENDAR_ID}",
            )

    observed = tables[PRICES.name]["session_date"]
    if (observed >= record.retrieved_at).any():
        raise refuse(
            RefusalReason.OBSERVED_AFTER_RETRIEVAL,
            f"a close dated {observed.max().date()} cannot be retrieved at {record.retrieved_at}",
        )
    return tables, missing_sessions(record, tables[PRICES.name])
