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
    AssumedAvailabilityRule,
    FundDataRefused,
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


def changed(raw: RawBatch, **tables) -> RawBatch:
    return dataclasses.replace(raw, **tables)


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
    assert stored.record.assumed_availability_rule is None


def test_the_store_never_computes_returns(raw, root):
    stored = read_batch(ingest_batch(root, raw).path)
    for spec in schema.DATA_TABLES:
        assert tuple(getattr(stored, spec.name).columns) == spec.names
        assert not [name for name in spec.names if "return" in name or "adj_" in name]
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
        ("distributions", "pay_date", "2025-01-03"),
        ("distributions", "record_date", "2025-01-01"),
        ("splits", "effective_date", "2025-01-02"),
    ],
)
def test_one_row_dated_in_the_lockbox_refuses_the_whole_batch(table, column, value, root):
    batch = fetch("2024-11-01", "2025-01-01")
    frame = getattr(batch, table).copy()
    if frame.empty:  # no split in this window: borrow the 2014 one
        frame = fetch().splits.copy()
    frame.loc[frame.index[-1], column] = day(value)  # one row moves into the lockbox
    late = changed(batch, **{table: frame})
    detail = refused(RefusalReason.LOCKBOX_VIOLATION, ingest_batch, root, late).detail
    assert "batch refused" in detail
    assert list(root.iterdir()) == []


def test_the_lockbox_has_one_source_and_binds_the_store(raw, root, monkeypatch):
    assert validation.MODEL_HOLDOUT_START is MODEL_HOLDOUT_START
    monkeypatch.setattr(validation, "MODEL_HOLDOUT_START", day("2014-06-01"))
    refused(RefusalReason.REQUEST_AFTER_CAP, ingest_batch, root, raw)


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


def test_an_unknown_availability_rule_is_refused(raw, root):
    refused(
        RefusalReason.UNKNOWN_ENUM,
        ingest_batch,
        root,
        raw,
        assumed_availability_rule="od9_approved_close_plus_two_hours",
    )


def test_the_assumed_availability_rule_is_only_a_label(raw, root):
    rule = AssumedAvailabilityRule.PROPOSED_OD9_SESSION_1800_NEW_YORK
    stored = read_batch(ingest_batch(root, raw, assumed_availability_rule=rule).path)
    assert stored.record.assumed_availability_rule is rule
    assert "proposed" in rule.value
    for frame in (stored.prices, stored.distributions, stored.splits):
        assert not [name for name in frame.columns if "avail" in name]


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
        ("splits", "ratio_new_per_old", 0.0),
        ("splits", "ratio_new_per_old", -np.inf),
    ],
)
def test_non_finite_or_non_positive_values_are_refused(raw, root, table, column, value):
    frame = getattr(raw, table).copy()
    if column == "volume":
        frame[column] = frame[column].astype("float64")
    frame.loc[0, column] = value
    refused(RefusalReason.INVALID_VALUE, ingest_batch, root, changed(raw, **{table: frame}))


def test_rows_outside_the_request_are_refused(raw, root):
    early = changed(raw, prices=with_row(raw.prices, session_date=day("2014-04-30")))
    refused(RefusalReason.OUTSIDE_REQUEST, ingest_batch, root, early)
    bond_only = fetch(funds=("SYN_BOND",))
    stray = changed(bond_only, prices=pd.concat([bond_only.prices, raw.prices.iloc[-1:]]))
    refused(RefusalReason.UNREQUESTED_FUND, ingest_batch, root, stray)


def test_a_close_dated_after_retrieval_is_refused(raw, root):
    early_clock = dataclasses.replace(raw, retrieved_at=pd.Timestamp("2014-06-15T00:00:00Z"))
    refused(RefusalReason.OBSERVED_AFTER_RETRIEVAL, ingest_batch, root, early_clock)


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
    link = tmp_path / "20261013T000000000000Z-0123456789ab"
    link.symlink_to(path)
    refused(RefusalReason.SYMLINK, read_batch, link)
    refused(RefusalReason.NOT_A_BATCH, read_batch, root)


# --- the store ----------------------------------------------------------------------


def test_the_store_keeps_each_retrieval_as_its_own_observation(raw, root):
    ingest_batch(root, raw)
    ingest_batch(root, dataclasses.replace(raw, retrieved_at=RETRIEVED + pd.Timedelta(days=30)))
    (root / ".staging-crashed").mkdir()  # a crashed write's leftover is skipped
    store = read_store(root)
    assert len(store.batches) == 2
    assert list(store.batches.columns) == list(schema.BATCH_COLUMNS)
    assert len(store.prices) == 2 * len(raw.prices)
    assert not store.prices.duplicated(["batch_id", "fund_id", "session_date"]).any()
    assert set(store.prices["batch_id"]) == set(store.batches["batch_id"])
    (root / "not-a-batch").mkdir()
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
    assert distributions["ex_date"].isin(sessions).all()
    assert distributions["pay_date"].isin(sessions).all()
    assert set(distributions["fund_id"]) == set(SYNTHETIC_FUND_IDS)
    assert raw.splits[["fund_id", "ratio_new_per_old"]].values.tolist() == [["SYN_EQ", 2.0]]
    assert raw.prices.loc[raw.prices["fund_id"] == "SYN_CASH", "volume"].isna().all()
