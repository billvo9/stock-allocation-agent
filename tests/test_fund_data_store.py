import dataclasses
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.calendars import nyse
from stock_agent.data.funds import ingest, schema, validation
from stock_agent.data.funds.adapter import (
    SYNTHETIC_FUND_IDS,
    FundDataAdapter,
    RawBatch,
    SyntheticFundAdapter,
)
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.reader import read_batch, read_store
from stock_agent.data.funds.schema import (
    AdjustmentBasis,
    FundDataRefused,
    Instrument,
    RefusalReason,
)
from stock_agent.model_validation.audit import file_sha256, table_sha256
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

RETRIEVED = pd.Timestamp("2026-10-13T12:34:56.123456Z")
SPLIT_WINDOW = ("2014-05-01", "2014-07-01")  # holds SYN_EQ's split and dividends


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC").as_unit("us")


def fetch(start=SPLIT_WINDOW[0], end=SPLIT_WINDOW[1], funds=SYNTHETIC_FUND_IDS, at=RETRIEVED):
    return SyntheticFundAdapter(retrieved_at=at).fetch(list(funds), day(start), day(end))


@pytest.fixture(scope="module")
def raw() -> RawBatch:
    return fetch()


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "private_root"
    path.mkdir()
    return path


def refused(reason, call, *args, **kwargs):
    with pytest.raises(FundDataRefused) as caught:
        call(*args, **kwargs)
    assert caught.value.reason is reason, caught.value
    return caught.value


def with_row(frame: pd.DataFrame, **changes) -> pd.DataFrame:
    """The frame plus a copy of its first row with `changes` applied."""

    return pd.concat([frame, frame.iloc[[0]].assign(**changes)], ignore_index=True)


def with_value(frame: pd.DataFrame, column: str, value, row: int = 0) -> pd.DataFrame:
    """A copy of the frame with one cell changed."""

    copy = frame.copy()
    copy.loc[copy.index[row], column] = value
    return copy


def changed(raw: RawBatch, **fields) -> RawBatch:
    return dataclasses.replace(raw, **fields)


# --- round trip ----------------------------------------------------------------


def test_the_synthetic_adapter_satisfies_the_protocol():
    assert isinstance(SyntheticFundAdapter(retrieved_at=RETRIEVED), FundDataAdapter)


def test_a_batch_round_trips_with_its_observations_unchanged(raw, root):
    result = ingest_batch(root, raw)
    assert result.path.parent == root and result.path.name == result.record.batch_id
    assert result.missing_sessions.empty
    stored = read_batch(result.path)
    for name in ("prices", "distributions", "splits"):
        expected = schema.conform(
            getattr(schema, name.upper()),
            getattr(raw, name).assign(batch_id=stored.record.batch_id),
        )
        pd.testing.assert_frame_equal(getattr(stored, name), expected)
    assert stored.record == result.record
    assert stored.record.fund_ids == SYNTHETIC_FUND_IDS
    assert (stored.record.start, stored.record.end_exclusive) == tuple(map(day, SPLIT_WINDOW))
    assert [item.fund_id for item in stored.record.instruments] == sorted(SYNTHETIC_FUND_IDS)
    assert stored.record.raw_payload_sha256 == file_sha256(result.path / schema.PAYLOAD_FILE)
    assert (result.path / schema.PAYLOAD_FILE).read_bytes() == raw.raw_payload
    assert stored.record.calendar_id == nyse.CALENDAR_ID
    assert len(stored.prices) == 3 * len(nyse.sessions("2014-05-01", "2014-06-30"))


def test_adjustment_basis_and_retrieved_at_are_preserved(raw, root):
    basis = raw.prices["fund_id"].map(
        {
            "SYN_EQ": AdjustmentBasis.SPLIT_ADJUSTED_AS_OF_RETRIEVAL.value,
            "SYN_BOND": AdjustmentBasis.SPLIT_AND_DISTRIBUTION_ADJUSTED_AS_OF_RETRIEVAL.value,
            "SYN_CASH": AdjustmentBasis.UNADJUSTED.value,
        }
    )
    batch = changed(raw, prices=raw.prices.assign(adjustment_basis=basis.astype(object)))
    stored = read_batch(ingest_batch(root, batch).path)
    by_fund = stored.prices.groupby("fund_id")["adjustment_basis"].unique()
    assert {fund: list(values) for fund, values in by_fund.items()} == {
        "SYN_BOND": ["split_and_distribution_adjusted_as_of_retrieval"],
        "SYN_CASH": ["unadjusted"],
        "SYN_EQ": ["split_adjusted_as_of_retrieval"],
    }
    assert stored.record.retrieved_at == RETRIEVED  # microseconds included
    assert read_store(root).batches["retrieved_at"].tolist() == [RETRIEVED]


def test_a_raw_and_a_vendor_adjusted_close_for_one_session_coexist(raw, root):
    vendor = raw.prices.assign(
        adjustment_basis=AdjustmentBasis.SPLIT_ADJUSTED_AS_OF_RETRIEVAL.value,
        close=raw.prices["close"] * 0.5,  # values as the source gave them
    )
    both = pd.concat([raw.prices, vendor], ignore_index=True)
    result = ingest_batch(root, changed(raw, prices=both))
    stored = read_batch(result.path)
    assert len(stored.prices) == 2 * len(raw.prices)
    assert result.missing_sessions.empty
    key = ["fund_id", "session_date", "adjustment_basis"]
    expected = both.sort_values(key, kind="stable")["close"].to_numpy()
    np.testing.assert_array_equal(stored.prices["close"].to_numpy(), expected)
    one_basis_twice = pd.concat([raw.prices, raw.prices.iloc[[0]]], ignore_index=True)
    refused(RefusalReason.DUPLICATE_KEY, ingest_batch, root, changed(raw, prices=one_basis_twice))


COMPUTED = ("return", "growth", "wealth", "factor", "cumulative", "adj_close", "adjusted_close")


def test_the_store_never_computes_returns_or_adjusted_series(raw, root):
    stored = read_batch(ingest_batch(root, raw).path)
    for spec in schema.DATA_TABLES:
        assert tuple(getattr(stored, spec.name).columns) == spec.names
        assert not [name for name in spec.names if any(token in name for token in COMPUTED)]
    # The only price level is the source's close, read with its declared basis.
    numeric = [c.name for c in schema.PRICES.columns if c.kind == schema.POSITIVE_FLOAT]
    assert numeric == ["close"]
    # The split shows in the raw closes: nothing back-adjusts the earlier level.
    equity = stored.prices[stored.prices["fund_id"] == "SYN_EQ"].set_index("session_date")
    before, after = equity.loc[day("2014-06-06"), "close"], equity.loc[day("2014-06-09"), "close"]
    assert after < 0.6 * before
    np.testing.assert_array_equal(
        stored.prices["close"].to_numpy(),
        schema.conform(schema.PRICES, raw.prices.assign(batch_id="x"))["close"].to_numpy(),
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in Path(ingest.__file__).parent.glob("*.py")
    )
    for token in ("pct_change", "cumprod", ".shift(", ".diff(", "np.log", "available_at"):
        assert token not in source, token


def test_the_cap_boundary_is_accepted_and_ends_on_the_last_2024_session(root):
    batch = fetch("2024-12-02", "2025-01-01")
    assert batch.end_exclusive == MODEL_HOLDOUT_START
    stored = read_batch(ingest_batch(root, batch).path)
    assert stored.prices["session_date"].max() == day("2024-12-31")
    assert stored.missing_sessions.empty


# --- the lockbox cap -------------------------------------------------------------


def test_a_request_past_the_cap_is_refused_before_retrieval_and_at_ingest(raw, root):
    adapter = SyntheticFundAdapter(retrieved_at=RETRIEVED)
    for end in ("2025-01-02", "2026-01-01"):
        refused(
            RefusalReason.REQUEST_AFTER_CAP, adapter.fetch, ["SYN_EQ"], day("2024-01-02"), day(end)
        )
    late = dataclasses.replace(raw, end_exclusive=day("2025-01-02"))
    refused(RefusalReason.REQUEST_AFTER_CAP, ingest_batch, root, late)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("prices", "session_date", "2025-01-02"),
        ("distributions", "ex_date", "2025-01-02"),
        ("distributions", "declaration_date", "2025-01-02"),
        ("distributions", "pay_date", "2025-01-03"),
        ("distributions", "record_date", "2025-01-01"),
        ("splits", "effective_date", "2025-01-02"),
    ],
)
def test_one_row_dated_in_the_lockbox_refuses_the_whole_batch(table, column, value, root):
    batch = fetch("2024-11-01", "2025-01-01")
    frame = getattr(batch, table)
    if frame.empty:  # no split in this window: borrow the 2014 one
        frame = fetch().splits
    late = changed(batch, **{table: with_value(frame, column, day(value), row=-1)})
    detail = refused(RefusalReason.LOCKBOX_VIOLATION, ingest_batch, root, late).detail
    assert "batch refused" in detail
    assert list(root.iterdir()) == []


def test_the_lockbox_has_one_source_and_binds_the_store(raw, root, monkeypatch):
    assert validation.MODEL_HOLDOUT_START is MODEL_HOLDOUT_START
    monkeypatch.setattr(validation, "MODEL_HOLDOUT_START", day("2014-06-01"))
    refused(RefusalReason.REQUEST_AFTER_CAP, ingest_batch, root, raw)


# --- assumed availability (OD9 pending) ----------------------------------------------


@pytest.mark.parametrize(
    "rule", ["proposed_od9_session_1800_america_new_york", "anything", "", False]
)
def test_any_assumed_availability_rule_is_refused_until_od9(raw, root, rule):
    refused(RefusalReason.PENDING_OD9, ingest_batch, root, raw, assumed_availability_rule=rule)
    assert list(root.iterdir()) == []


def test_no_availability_label_or_timestamp_is_stored(raw, root):
    stored = read_batch(ingest_batch(root, raw).path)
    assert stored.record.assumed_availability_rule is None
    assert read_store(root).batches["assumed_availability_rule"].isna().all()
    for frame in (stored.prices, stored.distributions, stored.splits):
        assert not [name for name in frame.columns if "avail" in name]


# --- deterministic validation ------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "changes"),
    [
        ("prices", {"adjustment_basis": "total_return_adjusted"}),
        ("distributions", {"amount_basis": "per_share_adjusted"}),
        ("distributions", {"distribution_type": "special_dividend"}),
    ],
)
def test_an_unknown_enum_value_is_refused(raw, root, table, changes):
    frame = getattr(raw, table)
    batch = changed(raw, **{table: frame.assign(**changes)})
    refused(RefusalReason.UNKNOWN_ENUM, ingest_batch, root, batch)


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("prices", "session_date", "2014-05-26"),  # Memorial Day
        ("prices", "session_date", "2014-05-31"),  # Saturday
        ("distributions", "ex_date", "2014-05-26"),
        ("splits", "effective_date", "2014-05-26"),
    ],
)
def test_a_date_off_the_calendar_is_refused(raw, root, table, column, value):
    batch = changed(raw, **{table: with_row(getattr(raw, table), **{column: day(value)})})
    refused(RefusalReason.NOT_A_SESSION, ingest_batch, root, batch)


@pytest.mark.parametrize("table", ["prices", "distributions", "splits"])
def test_a_duplicate_key_is_refused(raw, root, table):
    batch = changed(raw, **{table: with_row(getattr(raw, table))})
    refused(RefusalReason.DUPLICATE_KEY, ingest_batch, root, batch)


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("prices", "close", np.nan),
        ("prices", "close", np.inf),
        ("prices", "close", 0.0),
        ("prices", "close", -101.5),
        ("prices", "volume", -5),
        ("prices", "volume", 2.5),
        ("distributions", "amount", np.inf),
        ("distributions", "amount", np.nan),
        ("distributions", "source_sequence", -1),
        ("distributions", "source_sequence", 0.5),
        ("distributions", "source_sequence", np.nan),
        ("splits", "ratio_new_per_old", 0.0),
        ("splits", "ratio_new_per_old", -np.inf),
    ],
)
def test_non_finite_or_non_positive_values_are_refused(raw, root, table, column, value):
    frame = getattr(raw, table).copy()
    if column in ("volume", "source_sequence"):
        frame[column] = frame[column].astype("float64")
    frame.loc[0, column] = value
    refused(RefusalReason.INVALID_VALUE, ingest_batch, root, changed(raw, **{table: frame}))


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("pay_date", "2014-05-01", RefusalReason.INVALID_DATE_ORDER),  # before its ex-date
        ("record_date", "2014-05-01", RefusalReason.INVALID_DATE_ORDER),
        ("declaration_date", "2014-07-15", RefusalReason.INVALID_DATE_ORDER),  # after
        ("pay_date", "2002-12-31", RefusalReason.OUTSIDE_COVERAGE),
    ],
)
def test_distribution_dates_must_be_ordered_and_inside_coverage(raw, root, column, value, reason):
    distributions = raw.distributions
    late = distributions.index[distributions["ex_date"] >= day("2014-06-01")][0]
    frame = with_value(distributions, column, day(value), row=int(late))
    refused(reason, ingest_batch, root, changed(raw, distributions=frame))


def test_a_declaration_may_precede_calendar_coverage(raw, root):
    distributions = raw.distributions
    late = int(distributions.index[distributions["ex_date"] >= day("2014-06-01")][0])
    frame = with_value(distributions, "declaration_date", day("2002-12-31"), row=late)
    stored = read_batch(ingest_batch(root, changed(raw, distributions=frame)).path)
    assert (stored.distributions["declaration_date"] == day("2002-12-31")).sum() == 1


def test_a_same_type_supplemental_needs_its_own_sequence(raw, root):
    first = raw.distributions.iloc[[0]]
    supplemental = first.assign(amount=0.07, source_sequence=1)
    batch = changed(raw, distributions=pd.concat([raw.distributions, supplemental]))
    stored = read_batch(ingest_batch(root, batch).path)
    same_day = stored.distributions[
        (stored.distributions["fund_id"] == first["fund_id"].iloc[0])
        & (stored.distributions["ex_date"] == first["ex_date"].iloc[0])
    ]
    assert same_day["source_sequence"].tolist() == [0, 1]
    clash = pd.concat([raw.distributions, first.assign(amount=0.07)])
    refused(RefusalReason.DUPLICATE_KEY, ingest_batch, root, changed(raw, distributions=clash))


def test_rows_outside_the_request_or_the_mapping_are_refused(raw, root):
    early = changed(raw, prices=with_row(raw.prices, session_date=day("2014-04-30")))
    refused(RefusalReason.OUTSIDE_REQUEST, ingest_batch, root, early)
    bond_only = fetch(funds=("SYN_BOND",))
    stray = changed(bond_only, prices=pd.concat([bond_only.prices, raw.prices.iloc[-1:]]))
    refused(RefusalReason.UNMAPPED_FUND, ingest_batch, root, stray)


def test_every_requested_fund_needs_exactly_one_instrument_mapping(raw, root):
    two = changed(raw, instruments=raw.instruments[:2])
    refused(RefusalReason.UNMAPPED_FUND, ingest_batch, root, two)
    repeated = changed(raw, instruments=(*raw.instruments, raw.instruments[0]))
    refused(RefusalReason.UNMAPPED_FUND, ingest_batch, root, repeated)
    extra = changed(raw, instruments=(*raw.instruments, Instrument("SYN_OTHER", "SYNO")))
    refused(RefusalReason.UNMAPPED_FUND, ingest_batch, root, extra)
    for symbol in ("", "SYN EQ", "SYNE\n", "ＳＹＮ"):
        with pytest.raises(FundDataRefused) as caught:
            Instrument("SYN_EQ", symbol)
        assert caught.value.reason is RefusalReason.INVALID_FIELD


def test_a_close_dated_at_or_after_retrieval_is_refused(raw, root):
    """A sanity check on the record, not an availability rule (no close time is modelled)."""

    last = raw.prices["session_date"].max()
    assert last == day("2014-06-30")
    at_midnight = changed(raw, retrieved_at=pd.Timestamp("2014-06-30T00:00:00Z"))
    refused(RefusalReason.OBSERVED_AFTER_RETRIEVAL, ingest_batch, root, at_midnight)
    earlier = changed(raw, retrieved_at=pd.Timestamp("2014-06-15T00:00:00Z"))
    refused(RefusalReason.OBSERVED_AFTER_RETRIEVAL, ingest_batch, root, earlier)
    next_day = changed(raw, retrieved_at=pd.Timestamp("2014-07-01T00:00:00Z"))
    assert ingest_batch(root, next_day).record.retrieved_at == day("2014-07-01")


def test_storage_ids_belong_to_the_store(raw, root):
    preset = changed(raw, prices=raw.prices.assign(batch_id="chosen-by-adapter"))
    refused(RefusalReason.SCHEMA_MISMATCH, ingest_batch, root, preset)
    extra = changed(raw, prices=raw.prices.assign(adjusted_close=raw.prices["close"]))
    refused(RefusalReason.SCHEMA_MISMATCH, ingest_batch, root, extra)
    record = ingest.build_record(raw)
    foreign = raw.prices.assign(batch_id=record.batch_id)
    foreign.loc[0, "batch_id"] = "20000101T000000000000Z-000000000000"
    refused(
        RefusalReason.UNKNOWN_BATCH,
        validation.validate_batch,
        record,
        foreign,
        raw.distributions.assign(batch_id=record.batch_id),
        raw.splits.assign(batch_id=record.batch_id),
    )


def test_missing_sessions_are_reported_and_never_repaired(raw, root):
    gaps = [day("2014-05-12"), day("2014-06-09")]
    dropped = raw.prices[
        ~(raw.prices["fund_id"].eq("SYN_BOND") & raw.prices["session_date"].isin(gaps))
    ]
    result = ingest_batch(root, changed(raw, prices=dropped))
    report = result.missing_sessions
    assert report[["fund_id", "session_date"]].values.tolist() == [
        ["SYN_BOND", gaps[0]],
        ["SYN_BOND", gaps[1]],
    ]
    assert set(report["status"]) == {"unavailable"}
    assert set(report["reason"]) == {"missing_session"}
    assert report["detail"].str.len().min() > 0
    stored = read_batch(result.path)
    assert len(stored.prices) == len(raw.prices) - 2
    pd.testing.assert_frame_equal(stored.missing_sessions, report)


def test_a_fund_with_no_closes_reports_every_requested_session(raw, root):
    cash_only = fetch(funds=("SYN_CASH",))
    no_rows = changed(
        cash_only, prices=cash_only.prices.iloc[0:0], distributions=raw.distributions.iloc[0:0]
    )
    stored = read_batch(ingest_batch(root, no_rows).path)
    assert stored.prices.empty and stored.distributions.empty
    expected = nyse.sessions("2014-05-01", "2014-06-30")
    assert list(stored.missing_sessions["session_date"]) == list(expected)


# --- write once and fail closed ------------------------------------------------------


def test_an_existing_batch_id_is_refused(raw, root):
    first = ingest_batch(root, raw)
    refused(RefusalReason.ALREADY_EXISTS, ingest_batch, root, raw)
    assert [entry.name for entry in root.iterdir()] == [first.path.name]
    later = dataclasses.replace(raw, retrieved_at=RETRIEVED + pd.Timedelta(hours=1))
    assert ingest_batch(root, later).path.name != first.path.name


def test_the_root_is_caller_given_and_must_exist(raw, tmp_path):
    assert inspect.signature(ingest_batch).parameters["root"].default is inspect.Parameter.empty
    refused(RefusalReason.INVALID_FIELD, ingest_batch, tmp_path / "absent", raw)


def test_a_failed_write_leaves_nothing_behind(raw, root, monkeypatch):
    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "rename", fail)
    with pytest.raises(OSError, match="disk full"):
        ingest_batch(root, raw)
    assert list(root.iterdir()) == []


def _rewrite_manifest(path: Path, filename: str, table: pd.DataFrame | None = None) -> None:
    """Update one manifest entry to match the file (and, given a table, its table hash)."""

    manifest_path = path / schema.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][filename]
    entry["file_sha256"] = file_sha256(path / filename)
    if table is not None:
        entry["table_sha256"] = table_sha256(table)
        entry["rows"] = len(table)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_a_tampered_file_is_refused(raw, root):
    path = ingest_batch(root, raw).path
    prices = pd.read_parquet(path / "prices.parquet")
    prices.loc[0, "close"] = prices.loc[0, "close"] * 1.01
    prices.to_parquet(path / "prices.parquet", index=False)
    refused(RefusalReason.HASH_MISMATCH, read_batch, path)


def test_the_table_hash_catches_a_file_hash_only_rewrite(raw, root):
    path = ingest_batch(root, raw).path
    prices = pd.read_parquet(path / "prices.parquet")
    prices.loc[0, "close"] = prices.loc[0, "close"] * 1.01  # same row count
    prices.to_parquet(path / "prices.parquet", index=False)
    _rewrite_manifest(path, "prices.parquet")  # file_sha256 only
    detail = refused(RefusalReason.HASH_MISMATCH, read_batch, path).detail
    assert "rows differ from the manifest" in detail


def test_a_tampered_payload_is_refused_even_with_a_rewritten_manifest(raw, root):
    path = ingest_batch(root, raw).path
    (path / schema.PAYLOAD_FILE).write_bytes(raw.raw_payload + b" ")
    refused(RefusalReason.HASH_MISMATCH, read_batch, path)
    _rewrite_manifest(path, schema.PAYLOAD_FILE)
    refused(RefusalReason.HASH_MISMATCH, read_batch, path)  # the record's payload hash


def test_a_tampered_record_is_refused_even_with_a_rewritten_manifest(raw, root):
    path = ingest_batch(root, raw).path
    record = json.loads((path / schema.BATCH_FILE).read_text(encoding="utf-8"))
    record["retrieved_at"] = "2026-10-14T00:00:00.000000Z"
    (path / schema.BATCH_FILE).write_text(json.dumps(record), encoding="utf-8")
    refused(RefusalReason.HASH_MISMATCH, read_batch, path)
    _rewrite_manifest(path, schema.BATCH_FILE)
    refused(RefusalReason.ID_MISMATCH, read_batch, path)


def test_a_stored_availability_label_is_refused_on_read(raw, root):
    path = ingest_batch(root, raw).path
    record = json.loads((path / schema.BATCH_FILE).read_text(encoding="utf-8"))
    record["assumed_availability_rule"] = "proposed_od9_session_1800_america_new_york"
    (path / schema.BATCH_FILE).write_text(json.dumps(record), encoding="utf-8")
    _rewrite_manifest(path, schema.BATCH_FILE)
    refused(RefusalReason.PENDING_OD9, read_batch, path)


def test_a_tampered_manifest_batch_id_is_refused(raw, root):
    path = ingest_batch(root, raw).path
    manifest_path = path / schema.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["batch_id"] = "20261013T000000000000Z-0123456789ab"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    refused(RefusalReason.ID_MISMATCH, read_batch, path)


def _replace_text(text: str, old: str, new: str) -> str:
    assert old in text
    return text.replace(old, new, 1)


def _with_newline(text: str, field: str) -> str:
    """The record with a trailing newline appended to one string field."""

    record = json.loads(text)
    record[field] += "\n"
    return json.dumps(record)


@pytest.mark.parametrize(
    ("filename", "edit", "reason"),
    [
        (
            schema.MANIFEST_FILE,
            lambda t: _replace_text(t, '"rows": ', '"rows": NaN, "x": '),
            "json",
        ),
        (
            schema.MANIFEST_FILE,
            lambda t: _replace_text(t, '"rows": ', '"rows": 1e999, "x": '),
            "json",
        ),
        (schema.MANIFEST_FILE, lambda t: _replace_text(t, "{", '{"files": {},'), "json"),
        (
            schema.BATCH_FILE,
            lambda t: _replace_text(t, '"terms_note": ', '"terms_note": NaN, "x": '),
            "json",
        ),
        (
            schema.BATCH_FILE,
            lambda t: _replace_text(t, '"terms_note": ', '"terms_note": -Infinity, "x": '),
            "json",
        ),
        (
            schema.BATCH_FILE,
            lambda t: _replace_text(t, '"calendar_id": ', '"calendar_id": 1e999, "x": '),
            "json",
        ),
        (schema.BATCH_FILE, lambda t: _replace_text(t, "{", '{"adapter_id": "other",'), "json"),
        (schema.BATCH_FILE, lambda t: _with_newline(t, "batch_id"), "batch_id"),
        (schema.BATCH_FILE, lambda t: _with_newline(t, "retrieved_at"), "retrieved_at"),
        # pandas parses full-width digits, so the ASCII-only pattern is the guard.
        (schema.BATCH_FILE, lambda t: _replace_text(t, '"2014-05-01"', '"２０14-05-01"'), "date"),
        (
            schema.BATCH_FILE,
            lambda t: _replace_text(t, '"2026-10-13T', '"２０26-10-13T'),
            "retrieved_at",
        ),
    ],
    ids=[
        "manifest-nan",
        "manifest-1e999",
        "manifest-duplicate-key",
        "record-nan",
        "record-infinity",
        "record-1e999",
        "record-duplicate-key",
        "record-batch-id-newline",
        "record-retrieved-at-newline",
        "record-full-width-start",
        "record-full-width-retrieved-at",
    ],
)
def test_strict_parsing_refuses_non_finite_duplicates_newlines_and_non_ascii_digits(
    raw, root, filename, edit, reason
):
    path = ingest_batch(root, raw).path
    target = path / filename
    target.write_text(edit(target.read_text(encoding="utf-8")), encoding="utf-8")
    if filename != schema.MANIFEST_FILE:
        _rewrite_manifest(path, filename)  # so the parser, not the file hash, decides
    expected = {
        "json": RefusalReason.INVALID_JSON,
        "batch_id": RefusalReason.INVALID_FIELD,
        "retrieved_at": RefusalReason.INVALID_TIMESTAMP,
        "date": RefusalReason.INVALID_DATE,
    }[reason]
    refused(expected, read_batch, path)


def test_identifier_and_date_patterns_accept_ascii_digits_only():
    full_width = "２０２６１０１３T１２３４５６１２３４５６Z-0123456789ab"
    assert schema.BATCH_ID_PATTERN.fullmatch("20261013T123456123456Z-0123456789ab")
    assert schema.BATCH_ID_PATTERN.fullmatch(full_width) is None
    assert schema.BATCH_ID_PATTERN.fullmatch("20261013T123456123456Z-0123456789ab\n") is None
    refused(RefusalReason.INVALID_DATE, schema.parse_date, "d", "２０２４-01-02")
    refused(RefusalReason.INVALID_DATE, schema.parse_date, "d", "2024-01-02\n")
    stamp = "２０26-10-13T12:00:00.000000Z"
    refused(RefusalReason.INVALID_TIMESTAMP, schema.parse_timestamp, "t", stamp)


def test_a_consistent_rewrite_still_fails_validation(raw, root):
    path = ingest_batch(root, raw).path
    prices = read_batch(path).prices.iloc[1:].reset_index(drop=True)
    prices.to_parquet(path / "prices.parquet", index=False)
    _rewrite_manifest(path, "prices.parquet", prices)
    refused(RefusalReason.DIAGNOSTICS_MISMATCH, read_batch, path)
    holiday = with_row(prices, session_date=day("2014-05-26"))
    holiday = schema.conform(schema.PRICES, holiday)
    holiday.to_parquet(path / "prices.parquet", index=False)
    _rewrite_manifest(path, "prices.parquet", holiday)
    refused(RefusalReason.NOT_A_SESSION, read_batch, path)


def test_unexpected_files_and_symlinks_are_refused(raw, root, tmp_path):
    path = ingest_batch(root, raw).path
    (path / "extra.txt").write_text("x", encoding="utf-8")
    refused(RefusalReason.FILE_SET, read_batch, path)
    (path / "extra.txt").unlink()
    (path / ".DS_Store").write_bytes(b"\x00")  # an OS file inside a batch refuses it
    refused(RefusalReason.FILE_SET, read_batch, path)
    (path / ".DS_Store").unlink()
    link = tmp_path / "20261013T000000000000Z-0123456789ab"
    link.symlink_to(path)
    refused(RefusalReason.SYMLINK, read_batch, link)
    refused(RefusalReason.NOT_A_BATCH, read_batch, root)


def test_a_symlinked_table_inside_a_batch_is_refused(raw, root, tmp_path):
    path = ingest_batch(root, raw).path
    moved = tmp_path / "elsewhere.parquet"
    (path / "prices.parquet").rename(moved)
    (path / "prices.parquet").symlink_to(moved)
    refused(RefusalReason.SYMLINK, read_batch, path)


# --- the store ----------------------------------------------------------------------


def test_the_store_keeps_each_retrieval_as_its_own_observation(raw, root):
    ingest_batch(root, raw)
    ingest_batch(root, dataclasses.replace(raw, retrieved_at=RETRIEVED + pd.Timedelta(days=30)))
    (root / f"{schema.STAGING_PREFIX}crashed").mkdir()  # a crashed write's leftover is skipped
    store = read_store(root)
    assert len(store.batches) == 2
    assert list(store.batches.columns) == list(schema.BATCH_COLUMNS)
    assert len(store.prices) == 2 * len(raw.prices)
    assert not store.prices.duplicated(["batch_id", *schema.PRICES.key]).any()
    assert set(store.prices["batch_id"]) == set(store.batches["batch_id"])
    (root / "not-a-batch").mkdir()
    refused(RefusalReason.NOT_A_BATCH, read_store, root)


@pytest.mark.parametrize("hidden", [".DS_Store", ".renamed-batch"])
def test_only_staging_entries_are_skipped_in_the_store(raw, root, hidden):
    path = ingest_batch(root, raw).path
    if hidden == ".DS_Store":
        (root / hidden).write_bytes(b"\x00")
    else:
        path.rename(root / hidden)  # a valid batch hidden by a rename must not vanish
    refused(RefusalReason.NOT_A_BATCH, read_store, root)


def test_an_empty_store_reads_as_empty_tables(root):
    store = read_store(root)
    assert store.batches.empty and store.prices.empty
    assert tuple(store.prices.columns) == schema.PRICES.names


# --- the synthetic adapter ---------------------------------------------------------------


def test_the_synthetic_adapter_is_deterministic_and_prefix_stable():
    short, again = fetch(), fetch()
    assert short.raw_payload == again.raw_payload
    longer = fetch(SPLIT_WINDOW[0], "2014-12-01")
    for name in ("prices", "distributions", "splits"):
        head = getattr(longer, name)
        date_column = head.columns[1]
        head = head[head[date_column] < day(SPLIT_WINDOW[1])].reset_index(drop=True)
        pd.testing.assert_frame_equal(head, getattr(short, name).reset_index(drop=True))


def test_the_synthetic_adapter_uses_only_invented_funds_and_calendar_dates(raw):
    assert all(fund.startswith("SYN_") for fund in SYNTHETIC_FUND_IDS)
    refused(
        RefusalReason.INVALID_REQUEST,
        SyntheticFundAdapter(retrieved_at=RETRIEVED).fetch,
        ["SPY"],
        day("2014-05-01"),
        day("2014-06-01"),
    )
    sessions = nyse.sessions("2014-05-01", "2014-07-31")
    distributions = raw.distributions
    for column in ("ex_date", "record_date", "pay_date", "declaration_date"):
        dates = distributions[column].dropna()
        assert dates.isin(sessions).all(), column
    assert set(distributions["fund_id"]) == set(SYNTHETIC_FUND_IDS)
    assert raw.splits[["fund_id", "ratio_new_per_old"]].values.tolist() == [["SYN_EQ", 2.0]]
    assert raw.prices.loc[raw.prices["fund_id"] == "SYN_CASH", "volume"].isna().all()


def test_the_synthetic_fixture_covers_the_awkward_cases(root):
    batch = fetch("2018-12-01", "2019-04-02")
    stored = read_batch(ingest_batch(root, batch).path)
    distributions = stored.distributions
    # Two types on one ex-date, plus a same-type supplemental with sequence 1.
    december = distributions[
        (distributions["fund_id"] == "SYN_BOND") & (distributions["ex_date"].dt.month == 12)
    ]
    assert december[["distribution_type", "source_sequence"]].values.tolist() == [
        ["capital_gain_short", 0],
        ["income_dividend", 0],
        ["income_dividend", 1],
    ]
    assert december["ex_date"].nunique() == 1
    # Paid the following month.
    march = distributions[
        (distributions["fund_id"] == "SYN_EQ") & (distributions["ex_date"].dt.month == 3)
    ]
    assert march["pay_date"].dt.month.tolist() == [4]
    assert (march["declaration_date"] <= march["ex_date"]).all()
    # A 1-for-4 reverse split: raw closes jump about fourfold, amounts follow the shares.
    assert stored.splits[["fund_id", "ratio_new_per_old"]].values.tolist() == [["SYN_BOND", 0.25]]
    bond = stored.prices[stored.prices["fund_id"] == "SYN_BOND"].set_index("session_date")
    ratio = bond.loc[day("2019-03-11"), "close"] / bond.loc[day("2019-03-08"), "close"]
    assert 3.9 < ratio < 4.1
    income = distributions[
        (distributions["fund_id"] == "SYN_BOND")
        & (distributions["distribution_type"] == "income_dividend")
        & (distributions["source_sequence"] == 0)
    ].set_index("ex_date")["amount"]
    assert income.loc[day("2019-03-01")] == pytest.approx(0.21)  # before the reverse split
    assert income.loc[day("2019-04-01")] == pytest.approx(0.84)  # per post-split share
