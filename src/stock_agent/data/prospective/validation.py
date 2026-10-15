"""
The prospective store's contract: the root marker, the batch record and the
batch checks shared by the writer (`store`) and the reader (`inference`).

Layout: `<root>/`
    PROSPECTIVE_STORE.json      the marker: store kind, schema, H_end, lookback,
                                window start and calendar; a root without exactly
                                this marker is not a prospective store
    P<retrieved_at>-<12 hex>/   one batch (F1 batch ids carry no `P`):
        prospective_batch.json  the batch record
        raw_payload.bin         the adapter's canonical envelope
        prices.parquet, distributions.parquet, splits.parquet,
        missing_sessions.parquet (F1's table specs, unchanged)
        manifest.json           file and table hashes, written last

Batch checks (each refuses the whole batch; nothing is repaired or dropped)
    - the request: valid unique fund ids, [start, end] inside the window
      (`window.check_window`), and an end no later than the last complete
      session at `retrieved_at` (the 18:00 America/New_York request rule);
    - the instrument mapping covers exactly the requested funds;
    - F1's column sets, types and values (`schema.conform`) and batch ids;
    - only unadjusted closes, amounts per share as paid, and no declaration,
      record or pay date (the store's only source provides none);
    - every row's fund mapped, inside the request and on a session of the
      rules calendar (`calendars/nyse_rules.py`);
    then the missing-session report, as in F1.

The record states the request rule it was collected under
(`window.REQUEST_CUTOFF_RULE`); like F1, no row carries an availability
timestamp.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.schema import (
    DATA_TABLES,
    DISTRIBUTIONS,
    MISSING_SESSIONS,
    PAYLOAD_FILE,
    PRICES,
    SPLITS,
    TABLE_FILES,
    AdjustmentBasis,
    AmountBasis,
    Instrument,
    RefusalReason,
    check_id,
    check_note,
    conform,
    dumps_canonical,
    format_timestamp,
    loads_strict,
    parse_date,
    parse_timestamp,
    refuse,
)
from stock_agent.data.funds.validation import MISSING_DETAIL, MISSING_REASON, MISSING_STATUS
from stock_agent.data.prospective.window import (
    H_END,
    LOOKBACK_SESSIONS,
    REQUEST_CUTOFF_RULE,
    WINDOW_START,
    ProspectiveRefusal,
    ProspectiveRefused,
    check_request,
)

SCHEMA_VERSION = "prospective-1"
STORE_KIND = "prospective"
MARKER_FILE = "PROSPECTIVE_STORE.json"
RECORD_FILE = "prospective_batch.json"
MANIFEST_FILE = "manifest.json"
FILE_SET = frozenset({RECORD_FILE, PAYLOAD_FILE, MANIFEST_FILE, *TABLE_FILES.values()})
MANIFEST_FIELDS = frozenset({"prospective_schema_version", "batch_id", "files"})
BATCH_ID_PATTERN = re.compile(r"P[0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}")
STAGING_PREFIX = ".staging-"
PURPOSE = (
    "inputs to frozen models only; never train, tune or evaluate; read only through "
    "stock_agent.data.prospective.inference"
)

RECORD_FIELDS = (
    "batch_id",
    "schema_version",
    "store_kind",
    "h_end",
    "lookback_sessions",
    "window_start",
    "calendar_id",
    "request_cutoff_rule",
    "adapter_id",
    "adapter_version",
    "request",
    "instruments",
    "retrieved_at",
    "raw_payload_sha256",
    "anchors_sha256",
    "terms_note",
    "vintage_note",
)
REQUEST_FIELDS = ("fund_ids", "start", "end")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RANGE_COLUMNS = {
    PRICES.name: "session_date",
    DISTRIBUTIONS.name: "ex_date",
    SPLITS.name: "effective_date",
}
_OPTIONAL_DATES = ("declaration_date", "record_date", "pay_date")


def _constants() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "store_kind": STORE_KIND,
        "h_end": f"{H_END:%Y-%m-%d}",
        "lookback_sessions": LOOKBACK_SESSIONS,
        "window_start": f"{WINDOW_START:%Y-%m-%d}",
        "calendar_id": nyse_rules.CALENDAR_ID,
    }


def marker_document() -> dict[str, Any]:
    """The exact content of a prospective root's marker file."""

    return {**_constants(), "purpose": PURPOSE}


def check_marker(root: str | Path) -> Path:
    """The root, if it is a directory holding exactly this store's marker."""

    base = Path(root)
    if base.is_symlink() or not base.is_dir():
        raise ProspectiveRefused(
            ProspectiveRefusal.NOT_PROSPECTIVE_STORE, f"{base} is not a directory"
        )
    marker = base / MARKER_FILE
    if marker.is_symlink() or not marker.is_file():
        raise ProspectiveRefused(
            ProspectiveRefusal.NOT_PROSPECTIVE_STORE, f"{base} has no {MARKER_FILE}"
        )
    try:
        document = loads_strict(marker.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        document = None
    if document != marker_document():
        raise ProspectiveRefused(
            ProspectiveRefusal.NOT_PROSPECTIVE_STORE,
            f"{base / MARKER_FILE} is not this store's marker (another H_end, window, calendar "
            "or schema)",
        )
    return base


def _sha(label: str, value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise refuse(RefusalReason.INVALID_FIELD, f"{label} is not a sha256")
    return value


@dataclass(frozen=True)
class ProspectiveRecord:
    """
    One prospective retrieval. `start` and `end` are the requested session
    dates, both inclusive. `anchors_sha256` hashes the private anchors file
    the distribution basis was checked against (None when none existed).
    """

    batch_id: str
    adapter_id: str
    adapter_version: str
    fund_ids: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    instruments: tuple[Instrument, ...]
    retrieved_at: pd.Timestamp
    raw_payload_sha256: str
    anchors_sha256: str | None
    terms_note: str
    vintage_note: str

    def to_json_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "batch_id": self.batch_id,
            **_constants(),
            "request_cutoff_rule": REQUEST_CUTOFF_RULE,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "request": {
                "fund_ids": list(self.fund_ids),
                "start": f"{self.start:%Y-%m-%d}",
                "end": f"{self.end:%Y-%m-%d}",
            },
            "instruments": [instrument.to_json_dict() for instrument in self.instruments],
            "retrieved_at": format_timestamp("retrieved_at", self.retrieved_at),
            "raw_payload_sha256": self.raw_payload_sha256,
            "anchors_sha256": self.anchors_sha256,
            "terms_note": self.terms_note,
            "vintage_note": self.vintage_note,
        }
        if not include_id:
            del data["batch_id"]
        return data

    @classmethod
    def from_json_dict(cls, data: object) -> ProspectiveRecord:
        if not isinstance(data, dict) or set(data) != set(RECORD_FIELDS):
            found = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise refuse(RefusalReason.FILE_SET, f"prospective record has {found}")
        constants = {**_constants(), "request_cutoff_rule": REQUEST_CUTOFF_RULE}
        differing = sorted(name for name, value in constants.items() if data[name] != value)
        if differing:
            raise ProspectiveRefused(
                ProspectiveRefusal.NOT_PROSPECTIVE_STORE,
                f"the record's {differing} differ from this store's",
            )
        request = data["request"]
        if not isinstance(request, dict) or set(request) != set(REQUEST_FIELDS):
            raise refuse(RefusalReason.INVALID_FIELD, "record request has the wrong fields")
        if not isinstance(request["fund_ids"], list) or not isinstance(data["instruments"], list):
            raise refuse(RefusalReason.INVALID_FIELD, "fund_ids and instruments must be lists")
        batch_id = data["batch_id"]
        if not isinstance(batch_id, str) or not BATCH_ID_PATTERN.fullmatch(batch_id):
            raise refuse(RefusalReason.INVALID_FIELD, f"batch_id {batch_id!r}")
        return cls(
            batch_id=batch_id,
            adapter_id=check_id("adapter_id", data["adapter_id"]),
            adapter_version=check_id("adapter_version", data["adapter_version"]),
            fund_ids=tuple(check_id("fund_id", fund_id) for fund_id in request["fund_ids"]),
            start=parse_date("request start", request["start"]),
            end=parse_date("request end", request["end"]),
            instruments=tuple(Instrument.from_json_dict(entry) for entry in data["instruments"]),
            retrieved_at=parse_timestamp("retrieved_at", data["retrieved_at"]),
            raw_payload_sha256=_sha("raw_payload_sha256", data["raw_payload_sha256"]),
            anchors_sha256=_sha("anchors_sha256", data["anchors_sha256"], optional=True),
            terms_note=check_note("terms_note", data["terms_note"]),
            vintage_note=check_note("vintage_note", data["vintage_note"]),
        )


def derive_batch_id(record_without_id: dict[str, Any]) -> str:
    """`P<YYYYMMDDTHHMMSSffffff>Z-<12 hex>`: the retrieval time and a content hash."""

    retrieved_at = parse_timestamp("retrieved_at", record_without_id["retrieved_at"])
    digest = hashlib.sha256(dumps_canonical(record_without_id).encode("utf-8")).hexdigest()
    return f"P{retrieved_at.strftime('%Y%m%dT%H%M%S%f')}Z-{digest[:12]}"


def requested_sessions(record: ProspectiveRecord) -> pd.DatetimeIndex:
    return nyse_rules.sessions(record.start, record.end)


def missing_sessions(record: ProspectiveRecord, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per requested fund and requested session with no close."""

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


def _check_instruments(record: ProspectiveRecord) -> None:
    mapped = [instrument.fund_id for instrument in record.instruments]
    if len(set(mapped)) != len(mapped) or set(mapped) != set(record.fund_ids):
        raise refuse(
            RefusalReason.UNMAPPED_FUND,
            f"instruments map {sorted(mapped)}, the request names {sorted(record.fund_ids)}",
        )


def _check_bases(tables: dict[str, pd.DataFrame]) -> None:
    prices, distributions = tables[PRICES.name], tables[DISTRIBUTIONS.name]
    if (prices["adjustment_basis"] != AdjustmentBasis.UNADJUSTED.value).any():
        raise refuse(RefusalReason.INVALID_VALUE, "the prospective store holds unadjusted closes")
    if (distributions["amount_basis"] != AmountBasis.PER_SHARE_AS_PAID.value).any():
        raise refuse(
            RefusalReason.INVALID_VALUE, "the prospective store holds amounts per share as paid"
        )
    for column in _OPTIONAL_DATES:
        if distributions[column].notna().any():
            raise refuse(
                RefusalReason.INVALID_FIELD,
                f"distributions.{column} must be empty: the store's source provides none",
            )


def validate_batch(
    record: ProspectiveRecord,
    prices: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """The conformed tables and the missing-session report, or a refusal."""

    check_request(record.fund_ids, record.start, record.end, record.retrieved_at)
    _check_instruments(record)
    frames = {PRICES.name: prices, DISTRIBUTIONS.name: distributions, SPLITS.name: splits}
    tables = {spec.name: conform(spec, frames[spec.name]) for spec in DATA_TABLES}
    for name, table in tables.items():
        foreign = sorted(set(table["batch_id"]) - {record.batch_id})
        if foreign:
            raise refuse(RefusalReason.UNKNOWN_BATCH, f"{name} rows name batch {foreign[:3]}")
    _check_bases(tables)
    sessions = requested_sessions(record)
    mapped = set(record.fund_ids)
    for name, column in _RANGE_COLUMNS.items():
        table = tables[name]
        unmapped = sorted(set(table["fund_id"]) - mapped)
        if unmapped:
            raise refuse(RefusalReason.UNMAPPED_FUND, f"{name} has unmapped funds {unmapped[:5]}")
        dates = table[column]
        outside = (dates < record.start) | (dates > record.end)
        if outside.any():
            raise refuse(
                RefusalReason.OUTSIDE_REQUEST,
                f"{name}.{column} {dates[outside].min().date()} is outside the request",
            )
        off = ~dates.isin(sessions)
        if off.any():
            raise refuse(
                RefusalReason.NOT_A_SESSION,
                f"{name}.{column} {dates[off].min().date()} is not a session of "
                f"{nyse_rules.CALENDAR_ID}",
            )
    return tables, missing_sessions(record, tables[PRICES.name])


__all__ = [
    "BATCH_ID_PATTERN",
    "FILE_SET",
    "MANIFEST_FIELDS",
    "MANIFEST_FILE",
    "MARKER_FILE",
    "RECORD_FILE",
    "SCHEMA_VERSION",
    "STAGING_PREFIX",
    "STORE_KIND",
    "ProspectiveRecord",
    "check_marker",
    "derive_batch_id",
    "marker_document",
    "missing_sessions",
    "requested_sessions",
    "validate_batch",
]
