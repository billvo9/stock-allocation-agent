"""
The only reader of the prospective store: `load_inference_window` gives a
frozen forecaster its inputs, and nothing else reads stored prospective rows.
It never imports the writer (`store`).

Contract
    load_inference_window(store_root, assets, forecast_close, lookback=63,
                          *, retrieved_by) -> InferenceWindow

    store_root      the owner's private prospective root (it holds
                    `PROSPECTIVE_STORE.json`; anything else is refused)
    assets          a non-empty sequence of unique fund ids (Tiingo tickers)
    forecast_close  a session of the rules calendar on or after H_END
                    (2026-10-14): a `YYYY-MM-DD` string, a `datetime.date`, or a
                    naive or UTC midnight timestamp
    lookback        an int >= 1 (default `LOOKBACK_SESSIONS`, 63)
    retrieved_by    required UTC timestamp, normally the forecast's recorded
                    decision time: only batches retrieved at or before it are
                    read, so a re-run or a late run sees exactly what the
                    forecast could have seen

    `window.sessions` holds the lookback + 1 sessions ending at
    `forecast_close` (64 closes, 63 daily returns, by default). A forecast
    close before H_END is refused (`sealed_holdout`) whatever the lookback:
    the lookback sessions are inputs only, so no forecast may be made, and
    later scored, inside them. The window's first session must also be on or
    after `WINDOW_START` (2026-07-16), so lookback <= 63 at H_END. Every
    returned date lies in [sessions[0], forecast_close].

Tables (sorted by key; `batch_id` gives each row's lineage)
    prices          fund_id, session_date, close, adjustment_basis, batch_id
                    (`unadjusted` closes only)
    distributions   fund_id, ex_date, amount, amount_basis, distribution_type,
                    batch_id (Tiingo divCash per share as paid, stored only after
                    the spot check against the owner's hand-filled anchors;
                    type `unknown`)
    splits          fund_id, effective_date, ratio_new_per_old, batch_id
                    (verified share-count splits only)
    missing         fund_id, session_date, reason: `no_batch` (no eligible
                    batch covers that fund and session) or `missing_session`
                    (the covering batch served no close)
    batches         the batches used: batch_id, retrieved_at, request_start,
                    request_end, raw_payload_sha256, anchors_sha256
    revisions       fund_id, session_date, table, batch_id, earlier_batch_id,
                    change: where the selected batch disagrees with an older
                    eligible batch covering the same fund and date (`changed`,
                    `removed` or `added`, in `prices`, `distributions` or
                    `splits`); values are never shown
    `complete` is True only when `missing` is empty: a close missing from the
    selected batch. It says nothing about revisions, so a distribution that a
    later batch no longer serves appears in `revisions`, not in `missing`.
    Gaps are never filled; the forecast job decides what an incomplete or
    revised window means.

Vintage rule
    Each batch is a snapshot of its requested range. For each (asset,
    session) the loader uses the latest-retrieved eligible batch whose
    request covers that asset and session (ties: the larger batch id). What
    that batch lacks is missing; it never falls back to an older batch, and
    every disagreement with an older covering batch is listed in `revisions`.

Refusals
    `FundDataRefused` (a `ProspectiveRefused` for the prospective reasons)
    with a machine-readable `reason`: an invalid request, `sealed_holdout`,
    `not_prospective_store`, `foreign_entry` (anything in the root other than
    the marker, batches and a crashed write's staging directory), and every
    integrity failure of a stored batch (file set, symlinks, hashes, record,
    id, payload hash, table hashes, the full batch validation and the
    missing-session report).

Point in time
    Values are as Tiingo served them at `retrieved_at`, not a point-in-time
    vintage. The 18:00 America/New_York convention only gated which sessions
    could be requested; it is not a vendor publication time and is not
    evidence of forecasting validity. Nothing here reads a clock or the
    network, adjusts a close or computes a return.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from stock_agent.calendars import nyse_rules
from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.data.funds.schema import (
    DATA_TABLES,
    DISTRIBUTIONS,
    MISSING_SESSIONS,
    PAYLOAD_FILE,
    PRICES,
    SPLITS,
    TABLE_FILES,
    FundDataRefused,
    RefusalReason,
    TableSpec,
    check_id,
    conform,
    loads_strict,
    refuse,
    utc_timestamp,
)
from stock_agent.data.prospective.validation import (
    BATCH_ID_PATTERN,
    FILE_SET,
    MANIFEST_FIELDS,
    MANIFEST_FILE,
    MARKER_FILE,
    RECORD_FILE,
    SCHEMA_VERSION,
    STAGING_PREFIX,
    ProspectiveRecord,
    check_marker,
    derive_batch_id,
    validate_batch,
)
from stock_agent.data.prospective.window import (
    H_END,
    LOOKBACK_SESSIONS,
    WINDOW_START,
    ProspectiveRefusal,
    ProspectiveRefused,
)
from stock_agent.model_validation.audit import file_sha256, table_sha256

PRICE_COLUMNS = ("fund_id", "session_date", "close", "adjustment_basis", "batch_id")
DISTRIBUTION_COLUMNS = (
    "fund_id",
    "ex_date",
    "amount",
    "amount_basis",
    "distribution_type",
    "batch_id",
)
SPLIT_COLUMNS = ("fund_id", "effective_date", "ratio_new_per_old", "batch_id")
MISSING_COLUMNS = ("fund_id", "session_date", "reason")
BATCH_COLUMNS = (
    "batch_id",
    "retrieved_at",
    "request_start",
    "request_end",
    "raw_payload_sha256",
    "anchors_sha256",
)
REVISION_COLUMNS = (
    "fund_id",
    "session_date",
    "table",
    "batch_id",
    "earlier_batch_id",
    "change",
)
NO_BATCH = "no_batch"
MISSING_SESSION = "missing_session"
_DATE_COLUMNS = {"prices": "session_date", "distributions": "ex_date", "splits": "effective_date"}
_PLAIN_ENTRY = {"file_sha256"}
_TABLE_ENTRY = {"file_sha256", "table_sha256", "rows"}


@dataclass(frozen=True, eq=False)
class StoredProspectiveBatch:
    """One verified stored batch; `raw_payload` (hash-checked) is kept for re-parsing."""

    path: Path
    record: ProspectiveRecord
    prices: pd.DataFrame = field(repr=False)
    distributions: pd.DataFrame = field(repr=False)
    splits: pd.DataFrame = field(repr=False)
    missing_sessions: pd.DataFrame = field(repr=False)
    raw_payload: bytes = field(repr=False)


@dataclass(frozen=True, eq=False)
class InferenceWindow:
    """A frozen forecaster's inputs; see the module docstring for every field."""

    assets: tuple[str, ...]
    forecast_close: pd.Timestamp
    lookback: int
    sessions: pd.DatetimeIndex
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame
    missing: pd.DataFrame
    batches: pd.DataFrame
    revisions: pd.DataFrame
    calendar_id: str = nyse_rules.CALENDAR_ID

    @property
    def complete(self) -> bool:
        return self.missing.empty


# --- reading and verifying batches ------------------------------------------------


def _fields(label: str, value: object, expected: set[str] | frozenset[str]) -> dict:
    if not isinstance(value, dict) or set(value) != set(expected):
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise refuse(RefusalReason.FILE_SET, f"{label} has {found}, expected {sorted(expected)}")
    return value


def _read_json(path: Path) -> object:
    try:
        return loads_strict(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        pass
    raise refuse(RefusalReason.INVALID_JSON, f"{path.name} is not UTF-8")


def _check_directory(directory: Path) -> None:
    if directory.is_symlink():
        raise refuse(RefusalReason.SYMLINK, f"{directory.name} is a symlink")
    if not directory.is_dir() or not BATCH_ID_PATTERN.fullmatch(directory.name):
        raise refuse(RefusalReason.NOT_A_BATCH, str(directory))
    entries = list(directory.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise refuse(RefusalReason.SYMLINK, f"{directory.name} holds a symlink")
    names = {entry.name for entry in entries}
    if names != FILE_SET or not all(entry.is_file() for entry in entries):
        raise refuse(RefusalReason.FILE_SET, f"files {sorted(names)}, expected {sorted(FILE_SET)}")


def _read_table(directory: Path, spec: TableSpec, entry: dict) -> pd.DataFrame:
    filename = TABLE_FILES[spec.name]
    try:
        frame = pd.read_parquet(directory / filename)
    except Exception as error:  # untrusted bytes: any parser failure refuses
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{filename} is unreadable") from error
    table = conform(spec, frame)
    if type(entry["rows"]) is not int or entry["rows"] != len(table):
        raise refuse(RefusalReason.HASH_MISMATCH, f"{filename} row count differs")
    if entry["table_sha256"] != table_sha256(table):
        raise refuse(RefusalReason.HASH_MISMATCH, f"{filename} rows differ from the manifest")
    return table


def read_batch(path: str | Path) -> StoredProspectiveBatch:
    """
    Read and verify one stored batch (for `load_inference_window` and the
    collection script's read-back check): the file set and symlinks, strict
    JSON, every file hash, the record and its recomputed id, the payload hash,
    the table hashes and row counts, the full batch validation, and the
    stored missing-session report against a fresh one.
    """

    directory = Path(path)
    _check_directory(directory)
    manifest = _fields("manifest", _read_json(directory / MANIFEST_FILE), MANIFEST_FIELDS)
    if manifest["prospective_schema_version"] != SCHEMA_VERSION:
        raise refuse(RefusalReason.INVALID_FIELD, "the manifest is not this store's schema")
    if manifest["batch_id"] != directory.name:
        raise refuse(RefusalReason.ID_MISMATCH, "manifest id differs from the directory name")
    files = _fields("manifest files", manifest["files"], FILE_SET - {MANIFEST_FILE})
    for name, entry in files.items():
        expected = _TABLE_ENTRY if name in TABLE_FILES.values() else _PLAIN_ENTRY
        _fields(f"manifest entry {name}", entry, expected)
        if entry["file_sha256"] != file_sha256(directory / name):
            raise refuse(RefusalReason.HASH_MISMATCH, f"{name} does not match the manifest")

    record = ProspectiveRecord.from_json_dict(_read_json(directory / RECORD_FILE))
    if record.batch_id != directory.name:
        raise refuse(RefusalReason.ID_MISMATCH, "record id differs from the directory name")
    if derive_batch_id(record.to_json_dict(include_id=False)) != record.batch_id:
        raise refuse(RefusalReason.ID_MISMATCH, "the batch id does not match the record")
    raw_payload = (directory / PAYLOAD_FILE).read_bytes()
    if hashlib.sha256(raw_payload).hexdigest() != record.raw_payload_sha256:
        raise refuse(RefusalReason.HASH_MISMATCH, "the raw payload differs from the record")

    stored = {
        spec.name: _read_table(directory, spec, files[TABLE_FILES[spec.name]])
        for spec in (*DATA_TABLES, MISSING_SESSIONS)
    }
    tables, missing = validate_batch(
        record, stored[PRICES.name], stored[DISTRIBUTIONS.name], stored[SPLITS.name]
    )
    if table_sha256(missing) != table_sha256(stored[MISSING_SESSIONS.name]):
        raise refuse(RefusalReason.DIAGNOSTICS_MISMATCH, "the missing-session report is stale")
    return StoredProspectiveBatch(
        path=directory,
        record=record,
        prices=tables[PRICES.name],
        distributions=tables[DISTRIBUTIONS.name],
        splits=tables[SPLITS.name],
        missing_sessions=missing,
        raw_payload=raw_payload,
    )


def _read_store(root: str | Path) -> list[StoredProspectiveBatch]:
    base = check_marker(root)
    batches = []
    for entry in sorted(base.iterdir()):
        if entry.name == MARKER_FILE or entry.name.startswith(STAGING_PREFIX):
            continue
        if entry.is_symlink() or not entry.is_dir() or not BATCH_ID_PATTERN.fullmatch(entry.name):
            raise ProspectiveRefused(
                ProspectiveRefusal.FOREIGN_ENTRY,
                f"{entry.name} in {base} is not a prospective batch; the store is refused",
            )
        batches.append(read_batch(entry))
    return batches


# --- the inference window --------------------------------------------------------


def _assets(assets: Sequence[str]) -> tuple[str, ...]:
    if isinstance(assets, str) or not isinstance(assets, Sequence) or not assets:
        raise refuse(RefusalReason.INVALID_REQUEST, "assets must be a non-empty sequence")
    try:
        ids = tuple(check_id("asset", asset) for asset in assets)
    except FundDataRefused as error:
        raise refuse(RefusalReason.INVALID_REQUEST, error.detail) from None
    if len(set(ids)) != len(ids):
        raise refuse(RefusalReason.INVALID_REQUEST, f"assets repeat: {list(ids)}")
    return ids


def _window_sessions(forecast_close: object, lookback: object) -> pd.DatetimeIndex:
    if type(lookback) is not int or lookback < 1:
        raise refuse(RefusalReason.INVALID_REQUEST, f"lookback {lookback!r} must be an int >= 1")
    try:
        close = nyse_rules.session_date(forecast_close, label="forecast_close")
        first = nyse_rules.session_offset(close, -lookback)
    except CalendarRefused as error:
        failure = str(error)
    else:
        failure = None
    if failure is not None:
        raise refuse(RefusalReason.INVALID_REQUEST, failure)
    if close < H_END:
        raise ProspectiveRefused(
            ProspectiveRefusal.SEALED_HOLDOUT,
            f"forecast_close {close.date()} is before H_end ({H_END.date()}); the lookback "
            "sessions are inputs to frozen models only, never forecast or scored",
        )
    if first < WINDOW_START:
        raise ProspectiveRefused(
            ProspectiveRefusal.SEALED_HOLDOUT,
            f"a {lookback}-session window ending {close.date()} starts {first.date()}, before "
            f"the prospective window ({WINDOW_START.date()}); sealed-period sessions are never "
            "returned",
        )
    return nyse_rules.sessions(first, close)


def _covering(
    assets: tuple[str, ...],
    sessions: pd.DatetimeIndex,
    batches: list[StoredProspectiveBatch],
) -> dict[tuple[str, pd.Timestamp], list[str]]:
    """For each (asset, session), the ids of the batches covering it, newest first."""

    newest_first = sorted(
        batches, key=lambda batch: (batch.record.retrieved_at, batch.record.batch_id), reverse=True
    )
    return {
        (asset, session): [
            batch.record.batch_id
            for batch in newest_first
            if asset in batch.record.fund_ids and batch.record.start <= session <= batch.record.end
        ]
        for asset in assets
        for session in sessions
    }


def _sources(covering: dict[tuple[str, pd.Timestamp], list[str]]) -> pd.DataFrame:
    """One row per (asset, session): the selected batch's id, or None (an object column)."""

    keys = list(covering)
    return pd.DataFrame(
        {
            "fund_id": pd.Series([asset for asset, _ in keys], dtype=object),
            "session_date": pd.Series(
                [session for _, session in keys], dtype="datetime64[us, UTC]"
            ),
            "batch_id": pd.Series(
                [ids[0] if ids else None for ids in covering.values()], dtype=object
            ),
        }
    )


def _observations(batch: StoredProspectiveBatch, table: str) -> dict[tuple, object]:
    """(fund, date) -> what the batch served there: a close, a ratio, or every distribution."""

    frame = getattr(batch, table)
    column = _DATE_COLUMNS[table]
    if table == "distributions":
        served: dict[tuple, list] = {}
        for row in frame.itertuples(index=False):
            entry = (row.distribution_type, row.source_sequence, row.amount, row.amount_basis)
            served.setdefault((row.fund_id, getattr(row, column)), []).append(entry)
        return {key: tuple(sorted(entries)) for key, entries in served.items()}
    value = "close" if table == "prices" else "ratio_new_per_old"
    return dict(zip(zip(frame["fund_id"], frame[column], strict=True), frame[value], strict=True))


def _revisions(
    covering: dict[tuple[str, pd.Timestamp], list[str]],
    by_id: dict[str, StoredProspectiveBatch],
) -> pd.DataFrame:
    """Every disagreement between the selected batch and an older covering batch."""

    cache: dict[tuple[str, str], dict[tuple, object]] = {}

    def observed(batch_id: str, table: str) -> dict[tuple, object]:
        if (batch_id, table) not in cache:
            cache[batch_id, table] = _observations(by_id[batch_id], table)
        return cache[batch_id, table]

    rows = []
    for key, ids in covering.items():
        for table in _DATE_COLUMNS:
            for earlier in ids[1:]:
                selected = observed(ids[0], table).get(key)
                older = observed(earlier, table).get(key)
                if selected == older:
                    continue
                change = "removed" if selected is None else "added" if older is None else "changed"
                rows.append((*key, table, ids[0], earlier, change))
    return pd.DataFrame(
        {
            name: pd.Series(
                [row[index] for row in rows],
                dtype="datetime64[us, UTC]" if name == "session_date" else object,
            )
            for index, name in enumerate(REVISION_COLUMNS)
        }
    )


def _select(
    sources: pd.DataFrame,
    batches: list[StoredProspectiveBatch],
    table: str,
    date_column: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    stacked = [getattr(batch, table) for batch in batches if len(getattr(batch, table))]
    keys = sources.dropna(subset=["batch_id"]).rename(columns={"session_date": date_column})
    if not stacked or keys.empty:
        spec = {"prices": PRICES, "distributions": DISTRIBUTIONS, "splits": SPLITS}[table]
        return conform(spec, pd.DataFrame(columns=list(spec.names)))[list(columns)].iloc[0:0]
    rows = pd.concat(stacked, ignore_index=True)
    chosen = rows.merge(keys, on=["fund_id", date_column, "batch_id"], how="inner")
    return chosen[list(columns)].sort_values(["fund_id", date_column]).reset_index(drop=True)


def load_inference_window(
    store_root: str | Path,
    assets: Sequence[str],
    forecast_close: object,
    lookback: int = LOOKBACK_SESSIONS,
    *,
    retrieved_by: pd.Timestamp,
) -> InferenceWindow:
    """The inputs for a forecast at `forecast_close`; the contract is in the module docstring."""

    ids = _assets(assets)
    sessions = _window_sessions(forecast_close, lookback)
    cutoff = utc_timestamp("retrieved_by", retrieved_by)
    batches = [batch for batch in _read_store(store_root) if batch.record.retrieved_at <= cutoff]
    covering = _covering(ids, sessions, batches)
    sources = _sources(covering)
    used = [batch for batch in batches if batch.record.batch_id in set(sources["batch_id"])]

    prices = _select(sources, used, "prices", "session_date", PRICE_COLUMNS)
    distributions = _select(sources, used, "distributions", "ex_date", DISTRIBUTION_COLUMNS)
    splits = _select(sources, used, "splits", "effective_date", SPLIT_COLUMNS)

    present = pd.MultiIndex.from_frame(prices[["fund_id", "session_date"]])
    keys = pd.MultiIndex.from_frame(sources[["fund_id", "session_date"]])
    missing = sources[~keys.isin(present)].copy()
    missing["reason"] = [
        NO_BATCH if pd.isna(batch) else MISSING_SESSION for batch in missing["batch_id"]
    ]
    missing = missing[list(MISSING_COLUMNS)].reset_index(drop=True)
    missing["reason"] = missing["reason"].astype(object)

    lineage = pd.DataFrame(
        [
            (
                batch.record.batch_id,
                batch.record.retrieved_at,
                batch.record.start,
                batch.record.end,
                batch.record.raw_payload_sha256,
                batch.record.anchors_sha256,
            )
            for batch in sorted(used, key=lambda batch: batch.record.batch_id)
        ],
        columns=list(BATCH_COLUMNS),
    ).astype({"batch_id": object, "raw_payload_sha256": object, "anchors_sha256": object})

    return InferenceWindow(
        assets=ids,
        forecast_close=sessions[-1],
        lookback=lookback,
        sessions=sessions,
        prices=prices,
        distributions=distributions,
        splits=splits,
        missing=missing,
        batches=lineage,
        revisions=_revisions(covering, {batch.record.batch_id: batch for batch in batches}),
    )


__all__ = [
    "BATCH_COLUMNS",
    "DISTRIBUTION_COLUMNS",
    "MISSING_COLUMNS",
    "MISSING_SESSION",
    "NO_BATCH",
    "PRICE_COLUMNS",
    "REVISION_COLUMNS",
    "SPLIT_COLUMNS",
    "InferenceWindow",
    "ProspectiveRefused",
    "StoredProspectiveBatch",
    "load_inference_window",
    "read_batch",
]
