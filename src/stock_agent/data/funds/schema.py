"""
The fund-data contract: closed enums, typed tables, the batch record and
structured refusals.

Tables (one batch's rows; the store-wide key adds `batch_id`, because a later
retrieval of the same fund and date is a new observation, never an overwrite):

    prices            key (fund_id, session_date, adjustment_basis)
    distributions     key (fund_id, ex_date, distribution_type, source_sequence)
    splits            key (fund_id, effective_date)
    missing_sessions  the structured missing-session report (status, reason, detail)

A close is stored only as the source provided it, with its declared
`adjustment_basis`; a raw and a vendor-adjusted close for one session can
coexist. Nothing here computes a return or an adjusted series.

Every date column holds dates as 00:00 UTC (`datetime64[us, UTC]`). A date
says when something happened in the market, never when it became known:
`BatchRecord.retrieved_at` is when this project obtained it. No availability
rule is approved (OD9), so `assumed_availability_rule` must be None in
version 1. Nothing here reads a clock, a file or the network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from stock_agent.calendars import nyse

SCHEMA_VERSION = "1.0"

# ASCII classes only (`\d` also matches other scripts' digits). Every pattern is
# applied with fullmatch: `$` alone would accept a trailing newline.
_TIMESTAMP_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z")
_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}")
_SOURCE_TEXT_PATTERN = re.compile(r"[\x21-\x7e]{1,64}")  # printable ASCII, no spaces
BATCH_ID_PATTERN = re.compile(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}")


class AdjustmentBasis(StrEnum):
    """How a stored close relates to the traded price on its session date."""

    UNADJUSTED = "unadjusted"
    # Adjusted by the source with the actions it knew at retrieval: the level of
    # an old close depends on later events, so it is not a point-in-time value.
    SPLIT_ADJUSTED_AS_OF_RETRIEVAL = "split_adjusted_as_of_retrieval"
    SPLIT_AND_DISTRIBUTION_ADJUSTED_AS_OF_RETRIEVAL = (
        "split_and_distribution_adjusted_as_of_retrieval"
    )


class AmountBasis(StrEnum):
    """The share basis of a distribution amount."""

    PER_SHARE_AS_PAID = "per_share_as_paid"
    SPLIT_ADJUSTED_AS_OF_RETRIEVAL = "split_adjusted_as_of_retrieval"


class DistributionType(StrEnum):
    INCOME_DIVIDEND = "income_dividend"
    CAPITAL_GAIN_SHORT = "capital_gain_short"
    CAPITAL_GAIN_LONG = "capital_gain_long"
    RETURN_OF_CAPITAL = "return_of_capital"
    UNKNOWN = "unknown"


class RefusalReason(StrEnum):
    INVALID_REQUEST = "invalid_request"
    REQUEST_AFTER_CAP = "request_after_cap"
    OUTSIDE_COVERAGE = "outside_coverage"
    LOCKBOX_VIOLATION = "lockbox_violation"
    PENDING_OD9 = "pending_od9"
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_DATE = "invalid_date"
    INVALID_DATE_ORDER = "invalid_date_order"
    INVALID_TIMESTAMP = "invalid_timestamp"
    INVALID_FIELD = "invalid_field"
    UNKNOWN_ENUM = "unknown_enum"
    INVALID_VALUE = "invalid_value"
    NOT_A_SESSION = "not_a_session"
    OUTSIDE_REQUEST = "outside_request"
    UNMAPPED_FUND = "unmapped_fund"
    DUPLICATE_KEY = "duplicate_key"
    UNKNOWN_BATCH = "unknown_batch"
    OBSERVED_AFTER_RETRIEVAL = "observed_after_retrieval"
    INVALID_JSON = "invalid_json"
    NOT_A_BATCH = "not_a_batch"
    SYMLINK = "symlink"
    FILE_SET = "file_set"
    HASH_MISMATCH = "hash_mismatch"
    ID_MISMATCH = "id_mismatch"
    DIAGNOSTICS_MISMATCH = "diagnostics_mismatch"
    ALREADY_EXISTS = "already_exists"


class FundDataRefused(Exception):
    """Fund data was refused; `reason` is machine-readable."""

    def __init__(self, reason: RefusalReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


def refuse(reason: RefusalReason, detail: str) -> FundDataRefused:
    return FundDataRefused(reason, detail)


# --- typed tables -----------------------------------------------------------

TEXT, ENUM, DATE, OPTIONAL_DATE, POSITIVE_FLOAT, OPTIONAL_COUNT, SEQUENCE = (
    "text",
    "enum",
    "date",
    "optional_date",
    "positive_float",
    "optional_count",
    "sequence",
)


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    enum: type[StrEnum] | None = None


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


PRICES = TableSpec(
    "prices",
    (
        Column("fund_id", TEXT),
        Column("session_date", DATE),
        Column("close", POSITIVE_FLOAT),
        Column("adjustment_basis", ENUM, AdjustmentBasis),
        Column("volume", OPTIONAL_COUNT),
        Column("source_id", TEXT),
        Column("batch_id", TEXT),
    ),
    key=("fund_id", "session_date", "adjustment_basis"),
)
DISTRIBUTIONS = TableSpec(
    "distributions",
    (
        Column("fund_id", TEXT),
        Column("ex_date", DATE),
        Column("declaration_date", OPTIONAL_DATE),
        Column("record_date", OPTIONAL_DATE),
        Column("pay_date", OPTIONAL_DATE),
        Column("amount", POSITIVE_FLOAT),
        Column("amount_basis", ENUM, AmountBasis),
        Column("distribution_type", ENUM, DistributionType),
        # 0 unless the source lists several distributions of one type on one
        # ex-date (for example a supplemental dividend); the adapter sets it.
        Column("source_sequence", SEQUENCE),
        Column("source_id", TEXT),
        Column("batch_id", TEXT),
    ),
    key=("fund_id", "ex_date", "distribution_type", "source_sequence"),
)
SPLITS = TableSpec(
    "splits",
    (
        Column("fund_id", TEXT),
        Column("effective_date", DATE),
        Column("ratio_new_per_old", POSITIVE_FLOAT),  # < 1 for a reverse split
        Column("source_id", TEXT),
        Column("batch_id", TEXT),
    ),
    key=("fund_id", "effective_date"),
)
MISSING_SESSIONS = TableSpec(
    "missing_sessions",
    (
        Column("fund_id", TEXT),
        Column("session_date", DATE),
        Column("status", TEXT),
        Column("reason", TEXT),
        Column("detail", TEXT),
        Column("batch_id", TEXT),
    ),
    key=("fund_id", "session_date"),
)
DATA_TABLES = (PRICES, DISTRIBUTIONS, SPLITS)

# The on-disk layout of one batch directory (`ingest` writes it, `reader` checks it).
BATCH_FILE = "batch.json"
PAYLOAD_FILE = "raw_payload.bin"
MANIFEST_FILE = "manifest.json"
TABLE_FILES = {spec.name: f"{spec.name}.parquet" for spec in (*DATA_TABLES, MISSING_SESSIONS)}
# A batch is written in a hidden staging directory and renamed into place; the
# store reader skips only entries with this prefix.
STAGING_PREFIX = ".staging-"

# The batches table (one row per stored batch; `read_store` builds it).
BATCH_COLUMNS = (
    "batch_id",
    "adapter_id",
    "adapter_version",
    "request_fund_ids",
    "request_start",
    "request_end_exclusive",
    "instruments",
    "retrieved_at",
    "raw_payload_sha256",
    "terms_note",
    "vintage_note",
    "assumed_availability_rule",
    "calendar_id",
    "schema_version",
)


def _text(spec: TableSpec, column: Column, values: pd.Series) -> pd.Series:
    if not values.map(lambda value: isinstance(value, str) and value != "").all():
        raise refuse(
            RefusalReason.INVALID_FIELD, f"{spec.name}.{column.name} needs non-empty strings"
        )
    if column.enum is not None:
        unknown = sorted(set(values) - {member.value for member in column.enum})
        if unknown:
            raise refuse(RefusalReason.UNKNOWN_ENUM, f"{spec.name}.{column.name} has {unknown[:5]}")
    return values.astype(object)


def _dates(spec: TableSpec, column: Column, values: pd.Series) -> pd.Series:
    label = f"{spec.name}.{column.name}"
    dtype = values.dtype
    if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC":
        # Only an empty column, or an all-missing optional one, may lack the dtype.
        if len(values) and not (column.kind == OPTIONAL_DATE and values.isna().all()):
            raise refuse(RefusalReason.INVALID_DATE, f"{label} must be UTC datetimes, not {dtype}")
        values = pd.Series(pd.NaT, index=values.index, dtype="datetime64[us, UTC]")
    values = values.astype("datetime64[us, UTC]")
    if column.kind == DATE and values.isna().any():
        raise refuse(RefusalReason.INVALID_DATE, f"{label} has missing dates")
    present = values.dropna()
    if (present != present.dt.normalize()).any():
        raise refuse(RefusalReason.INVALID_DATE, f"{label} has a time of day")
    return values


def _numeric(label: str, values: pd.Series) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        raise refuse(RefusalReason.INVALID_VALUE, f"{label} must be numeric, not {values.dtype}")
    return values.dropna().astype("float64").to_numpy()


def _floats(spec: TableSpec, column: Column, values: pd.Series) -> pd.Series:
    label = f"{spec.name}.{column.name}"
    if not len(values):
        return values.astype("float64")
    _numeric(label, values)
    numbers = values.astype("float64")
    array = numbers.to_numpy()
    if not (np.isfinite(array) & (array > 0)).all():
        raise refuse(RefusalReason.INVALID_VALUE, f"{label} needs finite positive values")
    return numbers


def _whole(label: str, values: pd.Series) -> None:
    present = _numeric(label, values)
    if not (np.isfinite(present) & (present >= 0) & (present == np.floor(present))).all():
        raise refuse(RefusalReason.INVALID_VALUE, f"{label} needs non-negative whole numbers")


def _counts(spec: TableSpec, column: Column, values: pd.Series) -> pd.Series:
    if values.isna().all():
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    _whole(f"{spec.name}.{column.name}", values)
    return values.astype("Int64")


def _sequences(spec: TableSpec, column: Column, values: pd.Series) -> pd.Series:
    label = f"{spec.name}.{column.name}"
    if not len(values):
        return values.astype("int64")
    if values.isna().any():
        raise refuse(RefusalReason.INVALID_VALUE, f"{label} has missing values")
    _whole(label, values)
    return values.astype("int64")


_CONVERTERS = {
    TEXT: _text,
    ENUM: _text,
    DATE: _dates,
    OPTIONAL_DATE: _dates,
    POSITIVE_FLOAT: _floats,
    OPTIONAL_COUNT: _counts,
    SEQUENCE: _sequences,
}


def conform(spec: TableSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """
    The table in canonical form: exactly the spec's columns, canonical dtypes,
    rows sorted by key. Refuses rather than coerces: a wrong column set,
    missing required values, unknown enum values or invalid numbers.
    """

    if not isinstance(frame, pd.DataFrame):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{spec.name} is not a DataFrame")
    if list(frame.columns) != list(dict.fromkeys(frame.columns)) or set(frame.columns) != set(
        spec.names
    ):
        raise refuse(
            RefusalReason.SCHEMA_MISMATCH,
            f"{spec.name} has {list(frame.columns)}, expected {list(spec.names)}",
        )
    frame = frame.reset_index(drop=True)
    data = {
        column.name: _CONVERTERS[column.kind](spec, column, frame[column.name])
        for column in spec.columns
    }
    table = pd.DataFrame(data, index=frame.index)
    duplicated = table.duplicated(list(spec.key))
    if duplicated.any():
        first = table.loc[duplicated, list(spec.key)].iloc[0].tolist()
        raise refuse(RefusalReason.DUPLICATE_KEY, f"{spec.name} repeats key {first}")
    return table.sort_values(list(spec.key), kind="stable").reset_index(drop=True)


def empty(spec: TableSpec) -> pd.DataFrame:
    """A zero-row table in canonical form."""

    dtypes = {
        TEXT: object,
        ENUM: object,
        DATE: "datetime64[us, UTC]",
        OPTIONAL_DATE: "datetime64[us, UTC]",
        POSITIVE_FLOAT: "float64",
        OPTIONAL_COUNT: "Int64",
        SEQUENCE: "int64",
    }
    return pd.DataFrame(
        {column.name: pd.Series(dtype=dtypes[column.kind]) for column in spec.columns}
    )


# --- strict JSON and timestamps --------------------------------------------


def _reject_constant(token: str) -> Any:
    raise refuse(RefusalReason.INVALID_JSON, f"non-finite number token {token!r}")


def _finite_float(text: str) -> float:
    """parse_float hook: refuses literals that overflow to infinity, such as 1e999."""

    value = float(text)
    if not math.isfinite(value):
        raise refuse(RefusalReason.INVALID_JSON, f"non-finite number {text!r}")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise refuse(RefusalReason.INVALID_JSON, f"duplicate key {key!r}")
        result[key] = value
    return result


def loads_strict(text: str) -> Any:
    """Parse JSON, refusing NaN/Infinity tokens, overflowing floats and duplicate keys."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except FundDataRefused:
        raise
    except (TypeError, ValueError) as error:
        failure = str(error)
    # Raised after the except block: a JSONDecodeError holds the whole document
    # (`.doc`), which may be a provider response, so the refusal never chains it.
    raise refuse(RefusalReason.INVALID_JSON, failure)


def dumps_canonical(value: Any) -> str:
    """Compact, key-sorted JSON; non-finite numbers are refused."""

    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        )
    except (TypeError, ValueError) as error:
        raise refuse(RefusalReason.INVALID_JSON, str(error)) from error


def dumps_file(value: Any) -> str:
    """The on-disk form: indented, key-sorted JSON with a trailing newline; no NaN."""

    try:
        return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise refuse(RefusalReason.INVALID_JSON, str(error)) from error


def utc_timestamp(label: str, value: object) -> pd.Timestamp:
    """A timezone-aware UTC `Timestamp` with at most microsecond precision."""

    if not isinstance(value, pd.Timestamp) or value is pd.NaT:
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} must be a pandas Timestamp")
    if value.tzinfo is None or str(value.tz) != "UTC":
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} must be timezone-aware UTC")
    if value.nanosecond:
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} has sub-microsecond precision")
    return value.as_unit("us")


def utc_date(label: str, value: object) -> pd.Timestamp:
    """A date as a UTC `Timestamp` at exactly 00:00."""

    timestamp = utc_timestamp(label, value)
    if timestamp != timestamp.normalize():
        raise refuse(RefusalReason.INVALID_DATE, f"{label} {timestamp} has a time of day")
    return timestamp


def format_timestamp(label: str, value: object) -> str:
    return utc_timestamp(label, value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_timestamp(label: str, value: object) -> pd.Timestamp:
    if not isinstance(value, str) or not _TIMESTAMP_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} {value!r} is not in the format")
    try:
        return pd.Timestamp(value).as_unit("us")
    except ValueError as error:
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} {value!r}") from error


def parse_date(label: str, value: object) -> pd.Timestamp:
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.INVALID_DATE, f"{label} {value!r} is not YYYY-MM-DD")
    try:
        return pd.Timestamp(value, tz="UTC").as_unit("us")
    except ValueError as error:
        raise refuse(RefusalReason.INVALID_DATE, f"{label} {value!r}") from error


def check_id(label: str, value: object) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.INVALID_FIELD, f"{label} {value!r} is not a valid id")
    return value


def check_source_text(label: str, value: object) -> str:
    """A source's own identifier (symbol or security id): printable ASCII, no spaces."""

    if not isinstance(value, str) or not _SOURCE_TEXT_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.INVALID_FIELD, f"{label} {value!r} is not a source identifier")
    return value


def check_note(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise refuse(RefusalReason.INVALID_FIELD, f"{label} must be a non-empty string")
    return value


def check_availability_rule(value: object) -> None:
    """Version 1 stores no availability rule: OD9 has approved none."""

    if value is not None:
        raise refuse(
            RefusalReason.PENDING_OD9,
            f"assumed_availability_rule {value!r}: no rule is approved until OD9",
        )


# --- the batch record ---------------------------------------------------------

BATCH_RECORD_FIELDS = (
    "batch_id",
    "schema_version",
    "adapter_id",
    "adapter_version",
    "request",
    "instruments",
    "retrieved_at",
    "raw_payload_sha256",
    "terms_note",
    "vintage_note",
    "assumed_availability_rule",
    "calendar_id",
)
REQUEST_FIELDS = ("fund_ids", "start", "end_exclusive")
INSTRUMENT_FIELDS = ("fund_id", "source_symbol", "source_security_id")


@dataclass(frozen=True)
class Instrument:
    """How the source identified one fund in this batch."""

    fund_id: str
    source_symbol: str
    source_security_id: str | None = None

    def __post_init__(self) -> None:
        check_id("instrument fund_id", self.fund_id)
        check_source_text("source_symbol", self.source_symbol)
        if self.source_security_id is not None:
            check_source_text("source_security_id", self.source_security_id)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fund_id": self.fund_id,
            "source_symbol": self.source_symbol,
            "source_security_id": self.source_security_id,
        }

    @classmethod
    def from_json_dict(cls, data: object) -> Instrument:
        if not isinstance(data, dict) or set(data) != set(INSTRUMENT_FIELDS):
            raise refuse(RefusalReason.INVALID_FIELD, "an instrument has the wrong fields")
        return cls(data["fund_id"], data["source_symbol"], data["source_security_id"])


@dataclass(frozen=True)
class BatchRecord:
    """
    One retrieval: who fetched what, when, and under which terms.

    `request` covers [start, end_exclusive) as session dates. `instruments`
    maps each requested fund to the source's symbol and optional security id.
    `retrieved_at` is the actual retrieval time (UTC). `raw_payload_sha256`
    hashes the adapter's raw payload stored beside the tables (the provider's
    bytes, or an adapter-documented canonical envelope). `vintage_note`
    describes what the source's values reflect at retrieval.
    `assumed_availability_rule` must be None until OD9 approves a rule.
    """

    batch_id: str
    adapter_id: str
    adapter_version: str
    fund_ids: tuple[str, ...]
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    instruments: tuple[Instrument, ...]
    retrieved_at: pd.Timestamp
    raw_payload_sha256: str
    terms_note: str
    vintage_note: str
    assumed_availability_rule: None = None
    calendar_id: str = nyse.CALENDAR_ID
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        check_availability_rule(self.assumed_availability_rule)

    def to_json_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "batch_id": self.batch_id,
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "request": {
                "fund_ids": list(self.fund_ids),
                "start": self.start.strftime("%Y-%m-%d"),
                "end_exclusive": self.end_exclusive.strftime("%Y-%m-%d"),
            },
            "instruments": [instrument.to_json_dict() for instrument in self.instruments],
            "retrieved_at": format_timestamp("retrieved_at", self.retrieved_at),
            "raw_payload_sha256": self.raw_payload_sha256,
            "terms_note": self.terms_note,
            "vintage_note": self.vintage_note,
            "assumed_availability_rule": None,
            "calendar_id": self.calendar_id,
        }
        if not include_id:
            del data["batch_id"]
        return data

    @classmethod
    def from_json_dict(cls, data: object) -> BatchRecord:
        if not isinstance(data, dict) or set(data) != set(BATCH_RECORD_FIELDS):
            found = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise refuse(RefusalReason.FILE_SET, f"batch record has {found}")
        request = data["request"]
        if not isinstance(request, dict) or set(request) != set(REQUEST_FIELDS):
            raise refuse(RefusalReason.INVALID_FIELD, "batch request has the wrong fields")
        if data["schema_version"] != SCHEMA_VERSION:
            raise refuse(RefusalReason.INVALID_FIELD, f"schema {data['schema_version']!r}")
        if data["calendar_id"] != nyse.CALENDAR_ID:
            raise refuse(RefusalReason.INVALID_FIELD, f"calendar {data['calendar_id']!r}")
        fund_ids = request["fund_ids"]
        if not isinstance(fund_ids, list):
            raise refuse(RefusalReason.INVALID_FIELD, "request fund_ids must be a list")
        instruments = data["instruments"]
        if not isinstance(instruments, list):
            raise refuse(RefusalReason.INVALID_FIELD, "instruments must be a list")
        check_availability_rule(data["assumed_availability_rule"])
        sha = data["raw_payload_sha256"]
        if not isinstance(sha, str) or not _SHA256_PATTERN.fullmatch(sha):
            raise refuse(RefusalReason.INVALID_FIELD, "raw_payload_sha256 is not a sha256")
        batch_id = data["batch_id"]
        if not isinstance(batch_id, str) or not BATCH_ID_PATTERN.fullmatch(batch_id):
            raise refuse(RefusalReason.INVALID_FIELD, f"batch_id {batch_id!r}")
        return cls(
            batch_id=batch_id,
            adapter_id=check_id("adapter_id", data["adapter_id"]),
            adapter_version=check_id("adapter_version", data["adapter_version"]),
            fund_ids=tuple(check_id("fund_id", fund_id) for fund_id in fund_ids),
            start=parse_date("request start", request["start"]),
            end_exclusive=parse_date("request end_exclusive", request["end_exclusive"]),
            instruments=tuple(Instrument.from_json_dict(entry) for entry in instruments),
            retrieved_at=parse_timestamp("retrieved_at", data["retrieved_at"]),
            raw_payload_sha256=sha,
            terms_note=check_note("terms_note", data["terms_note"]),
            vintage_note=check_note("vintage_note", data["vintage_note"]),
        )


def derive_batch_id(record_without_id: dict[str, Any]) -> str:
    """`YYYYMMDDTHHMMSSffffffZ-<12 hex>`: the retrieval time and a content hash."""

    retrieved_at = parse_timestamp("retrieved_at", record_without_id["retrieved_at"])
    digest = hashlib.sha256(dumps_canonical(record_without_id).encode("utf-8")).hexdigest()
    return f"{retrieved_at.strftime('%Y%m%dT%H%M%S%f')}Z-{digest[:12]}"


def batches_frame(records: list[BatchRecord]) -> pd.DataFrame:
    """The batches table: one row per record; fund ids and instruments as JSON lists."""

    rows = [
        {
            "batch_id": record.batch_id,
            "adapter_id": record.adapter_id,
            "adapter_version": record.adapter_version,
            "request_fund_ids": dumps_canonical(list(record.fund_ids)),
            "request_start": record.start,
            "request_end_exclusive": record.end_exclusive,
            "instruments": dumps_canonical([item.to_json_dict() for item in record.instruments]),
            "retrieved_at": record.retrieved_at,
            "raw_payload_sha256": record.raw_payload_sha256,
            "terms_note": record.terms_note,
            "vintage_note": record.vintage_note,
            "assumed_availability_rule": None,
            "calendar_id": record.calendar_id,
            "schema_version": record.schema_version,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows, columns=list(BATCH_COLUMNS))
    for column in ("request_start", "request_end_exclusive", "retrieved_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[us, UTC]")
    return frame.sort_values("batch_id", kind="stable").reset_index(drop=True)
