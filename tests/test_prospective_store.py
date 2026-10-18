"""
The prospective window, the write-once prospective store and its only
reader, `load_inference_window`. Every row is an invented fixture.
"""

import dataclasses
import json
import os
import stat
from datetime import date

import pandas as pd
import pytest
from prospective_helpers import (
    AFTER_H_END,
    FUNDS,
    LATER,
    batch,
    close_for,
    day,
    new_store,
    stored,
    tables,
)

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.adapter import SyntheticFundAdapter
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.schema import FundDataRefused, RefusalReason
from stock_agent.data.prospective import inference, window
from stock_agent.data.prospective.inference import (
    MISSING_SESSION,
    NO_BATCH,
    load_inference_window,
    read_batch,
)
from stock_agent.data.prospective.store import (
    ingest_prospective,
    prepare_store,
    store_root_refusal,
)
from stock_agent.data.prospective.validation import (
    BATCH_ID_PATTERN,
    FILE_SET,
    MARKER_FILE,
    RECORD_FILE,
    marker_document,
)
from stock_agent.data.prospective.window import (
    H_END,
    LOOKBACK_SESSIONS,
    WINDOW_START,
    ProspectiveRefusal,
    check_request,
    close_convention_at,
    last_complete_session,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START


def refused(reason, call, *args, **kwargs) -> FundDataRefused:
    with pytest.raises(FundDataRefused) as caught:
        call(*args, **kwargs)
    assert caught.value.reason == reason, caught.value
    return caught.value


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_convert("UTC").as_unit("us")


# --- the window -------------------------------------------------------------------


def test_the_window_follows_the_owner_decisions():
    assert H_END == pd.Timestamp("2026-10-14", tz="UTC")
    assert LOOKBACK_SESSIONS == 63
    assert WINDOW_START == pd.Timestamp("2026-07-16", tz="UTC")
    assert window.MODEL_HOLDOUT_START is MODEL_HOLDOUT_START  # imported, never restated
    span = nyse_rules.sessions(WINDOW_START, H_END)
    assert len(span) == LOOKBACK_SESSIONS + 1 and span[-1] == H_END  # H_END and 63 before it


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-10-14T17:59:59-04:00", "2026-10-13"),  # before 18:00 ET: not yet complete
        ("2026-10-14T18:00:00-04:00", "2026-10-14"),
        ("2026-10-15T03:00:00+00:00", "2026-10-14"),  # 23:00 ET on the 14th
        ("2026-10-17T12:00:00-04:00", "2026-10-16"),  # Saturday: Friday's close
        ("2026-10-19T09:00:00-04:00", "2026-10-16"),  # Monday morning
        ("2026-11-26T20:00:00-05:00", "2026-11-25"),  # Thanksgiving is not a session
        ("2026-11-02T22:59:00+00:00", "2026-10-30"),  # 17:59 EST after the DST change
        ("2026-11-02T23:00:00+00:00", "2026-11-02"),  # 18:00 EST
        ("2026-07-16T18:00:00-04:00", "2026-07-16"),  # the window's first session
    ],
)
def test_the_last_complete_session_uses_18_00_new_york(now, expected):
    assert last_complete_session(utc(now)) == day(expected)


def test_the_close_convention_is_18_00_new_york_in_both_seasons():
    assert close_convention_at(day("2026-10-14")) == utc("2026-10-14T22:00:00Z")
    assert close_convention_at(day("2026-12-01")) == utc("2026-12-01T23:00:00Z")


def test_no_session_is_complete_before_the_window_or_past_the_calendar():
    refused(
        ProspectiveRefusal.INCOMPLETE_SESSION,
        last_complete_session,
        utc("2026-07-16T17:00:00-04:00"),
    )
    refused(RefusalReason.OUTSIDE_COVERAGE, last_complete_session, utc("2031-01-02T23:00:00Z"))
    refused(RefusalReason.INVALID_TIMESTAMP, last_complete_session, pd.Timestamp("2026-10-14"))


@pytest.mark.parametrize(
    ("start", "end", "reason"),
    [
        ("2026-07-15", "2026-10-14", ProspectiveRefusal.SEALED_HOLDOUT),
        ("2025-01-02", "2026-10-14", ProspectiveRefusal.SEALED_HOLDOUT),
        ("2024-12-31", "2026-10-14", ProspectiveRefusal.BEFORE_WINDOW),
        ("2026-07-16", "2026-10-15", ProspectiveRefusal.INCOMPLETE_SESSION),
        ("2026-10-14", "2026-07-16", RefusalReason.INVALID_REQUEST),
        ("2026-07-16", "2031-01-02", RefusalReason.OUTSIDE_COVERAGE),
    ],
)
def test_requests_outside_the_window_or_incomplete_are_refused(start, end, reason):
    refused(reason, check_request, FUNDS, day(start), day(end), AFTER_H_END)


def test_a_valid_request_and_its_input_checks():
    ids, first, last = check_request(list(FUNDS), WINDOW_START, H_END, AFTER_H_END)
    assert ids == FUNDS and (first, last) == (WINDOW_START, H_END)
    for funds in ((), "VTI", ("VTI", "VTI"), ("bad id",)):
        refused(
            RefusalReason.INVALID_REQUEST, check_request, funds, WINDOW_START, H_END, AFTER_H_END
        )
    late = pd.Timestamp("2026-10-14T15:00Z")
    refused(RefusalReason.INVALID_DATE, check_request, FUNDS, WINDOW_START, late, AFTER_H_END)


# --- the store root ---------------------------------------------------------------


def test_a_new_root_is_private_and_holds_only_the_marker(tmp_path):
    root = tmp_path / "a" / "prospective"
    assert prepare_store(root) is None
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert [entry.name for entry in root.iterdir()] == [MARKER_FILE]
    assert json.loads((root / MARKER_FILE).read_text()) == marker_document()
    assert marker_document()["h_end"] == "2026-10-14"
    assert marker_document()["window_start"] == "2026-07-16"
    assert marker_document()["calendar_id"] == "nyse_full_day_rules_2003_2030_v1"
    assert prepare_store(root) is None  # an existing store is accepted as it is


def test_an_existing_root_is_never_changed(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    for mode in (0o755, 0o750, 0o701):
        os.chmod(root, mode)
        refusal = prepare_store(root)
        assert refusal is not None and "restrict it yourself" in refusal
        assert stat.S_IMODE(root.stat().st_mode) == mode and not any(root.iterdir())
    os.chmod(root, 0o700)
    assert prepare_store(root) is None  # empty and private: the marker is written
    assert (root / MARKER_FILE).is_file()


def test_new_parents_are_private_and_existing_directories_are_never_changed(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    os.chmod(existing, 0o755)
    root = existing / "a" / "b" / "store"
    assert store_root_refusal(root) is None and not (existing / "a").exists()  # read-only
    assert prepare_store(root) is None
    for directory in (root, root.parent, root.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory
    assert stat.S_IMODE(existing.stat().st_mode) == 0o755


def test_the_read_only_root_check_creates_nothing(tmp_path):
    open_root = tmp_path / "open"
    open_root.mkdir()
    os.chmod(open_root, 0o755)
    assert "restrict it yourself" in store_root_refusal(open_root)
    assert store_root_refusal(tmp_path / "missing" / "root") is None
    assert not (tmp_path / "missing").exists()
    os.chmod(open_root, 0o700)
    assert store_root_refusal(open_root) is None and not any(open_root.iterdir())


def test_a_research_store_or_any_non_empty_directory_is_refused(tmp_path):
    research = tmp_path / "research"
    research.mkdir(mode=0o700)
    os.chmod(research, 0o700)
    raw = SyntheticFundAdapter(retrieved_at=pd.Timestamp("2026-10-14T00:00:00Z")).fetch(
        ["SYN_EQ"], day("2024-12-02"), day("2025-01-01")
    )
    ingest_batch(research, raw)
    refusal = prepare_store(research)
    assert refusal is not None and "never the research store" in refusal
    assert not (research / MARKER_FILE).exists()


@pytest.mark.parametrize(
    "change",
    [
        {"h_end": "2026-10-15"},
        {"lookback_sessions": 64},
        {"window_start": "2026-07-15"},
        {"calendar_id": "nyse_full_day_2003_2024_v1"},
        # The calendar before the owner's 2026-10-18 switch: a store made under it is refused.
        {"calendar_id": "nyse_full_day_rules_2003_2027_v1"},
        {"store_kind": "research"},
    ],
)
def test_a_marker_for_another_decision_refuses_the_store(tmp_path, change):
    root = new_store(tmp_path / "store")
    (root / MARKER_FILE).write_text(json.dumps({**marker_document(), **change}))
    refused(ProspectiveRefusal.NOT_PROSPECTIVE_STORE, ingest_prospective, root, batch())
    refused(
        ProspectiveRefusal.NOT_PROSPECTIVE_STORE,
        load_inference_window,
        root,
        FUNDS,
        H_END,
        retrieved_by=LATER,
    )


# --- writing and reading batches ----------------------------------------------------


def test_a_batch_is_written_once_and_read_back_exactly(tmp_path):
    root = new_store(tmp_path / "store")
    result = ingest_prospective(root, batch())
    assert BATCH_ID_PATTERN.fullmatch(result.path.name) and result.path.name.startswith("P")
    assert {path.name for path in result.path.iterdir()} == FILE_SET
    assert stat.S_IMODE(result.path.stat().st_mode) == 0o700
    back = read_batch(result.path)
    prices, distributions, _ = tables()
    assert back.prices.drop(columns="batch_id").equals(
        prices.sort_values(["fund_id", "session_date"]).reset_index(drop=True)
    )
    assert len(back.distributions) == len(distributions) == 4 and back.splits.empty
    assert back.missing_sessions.empty
    assert back.raw_payload == b"invented prospective fixture payload"
    record = json.loads((result.path / RECORD_FILE).read_text())
    assert record["request"] == {
        "fund_ids": list(FUNDS),
        "start": "2026-07-16",
        "end": "2026-10-14",
    }
    assert record["request_cutoff_rule"] == window.REQUEST_CUTOFF_RULE
    assert record["anchors_sha256"] == "0" * 64
    refused(RefusalReason.ALREADY_EXISTS, ingest_prospective, root, batch())


def test_missing_sessions_are_reported_never_filled(tmp_path):
    root = new_store(tmp_path / "store")
    result = ingest_prospective(root, batch(skip={("AGG", "2026-08-14")}))
    report = read_batch(result.path).missing_sessions
    assert list(zip(report["fund_id"], report["session_date"], strict=True)) == [
        ("AGG", day("2026-08-14"))
    ]
    assert set(report["reason"]) == {"missing_session"}


def _with_sealed_price(item):
    """A batch starting at WINDOW_START that smuggles a 2026-07-15 close."""

    row = item.prices.iloc[[0]].assign(session_date=day("2026-07-15"))
    return dataclasses.replace(item, prices=pd.concat([row, item.prices], ignore_index=True))


def _with_sealed_distribution(item):
    row = item.distributions.iloc[[0]].assign(ex_date=day("2026-07-15"))
    rows = pd.concat([row, item.distributions], ignore_index=True)
    return dataclasses.replace(item, distributions=rows)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: batch(start=day("2026-07-15")), ProspectiveRefusal.SEALED_HOLDOUT),
        (lambda: batch(start=day("2024-12-02")), ProspectiveRefusal.BEFORE_WINDOW),
        (
            lambda: batch(retrieved_at=pd.Timestamp("2026-10-14T21:59:00Z")),
            ProspectiveRefusal.INCOMPLETE_SESSION,
        ),
        (lambda: batch(splits=[("VTI", "2026-07-03", 2.0)]), RefusalReason.OUTSIDE_REQUEST),
        (lambda: batch(splits=[("VTI", "2026-09-07", 2.0)]), RefusalReason.NOT_A_SESSION),
        (lambda: batch(funds=("VTI", "VTI")), RefusalReason.INVALID_REQUEST),
        (lambda: _with_sealed_price(batch()), RefusalReason.OUTSIDE_REQUEST),
        (lambda: _with_sealed_distribution(batch()), RefusalReason.OUTSIDE_REQUEST),
    ],
)
def test_batches_outside_the_contract_are_refused_and_leave_nothing(tmp_path, make, reason):
    root = new_store(tmp_path / "store")
    refused(reason, ingest_prospective, root, make())
    assert [entry.name for entry in root.iterdir()] == [MARKER_FILE]


def test_only_unadjusted_closes_and_as_paid_amounts_without_dates_are_stored(tmp_path):
    root = new_store(tmp_path / "store")
    item = batch()
    adjusted = item.prices.assign(adjustment_basis="split_adjusted_as_of_retrieval")
    refused(
        RefusalReason.INVALID_VALUE,
        ingest_prospective,
        root,
        dataclasses.replace(item, prices=adjusted),
    )
    rebased = item.distributions.assign(amount_basis="split_adjusted_as_of_retrieval")
    refused(
        RefusalReason.INVALID_VALUE,
        ingest_prospective,
        root,
        dataclasses.replace(item, distributions=rebased),
    )
    dated = item.distributions.assign(pay_date=item.distributions["ex_date"])
    refused(
        RefusalReason.INVALID_FIELD,
        ingest_prospective,
        root,
        dataclasses.replace(item, distributions=dated),
    )
    assert [entry.name for entry in root.iterdir()] == [MARKER_FILE]


def test_an_f1_batch_is_not_a_prospective_batch(tmp_path):
    root = new_store(tmp_path / "store")
    raw = SyntheticFundAdapter(retrieved_at=pd.Timestamp("2026-10-14T00:00:00Z")).fetch(
        ["SYN_EQ"], day("2024-12-02"), day("2025-01-01")
    )
    refused(RefusalReason.INVALID_FIELD, ingest_prospective, root, raw)


def test_a_root_without_the_marker_is_refused(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    refused(ProspectiveRefusal.NOT_PROSPECTIVE_STORE, ingest_prospective, plain, batch())
    refused(
        ProspectiveRefusal.NOT_PROSPECTIVE_STORE,
        load_inference_window,
        plain,
        FUNDS,
        H_END,
        retrieved_by=LATER,
    )


def _only_batch(root):
    return next(entry for entry in root.iterdir() if entry.name != MARKER_FILE)


def test_the_reader_refuses_tampered_or_extra_files(tmp_path):
    root = stored(tmp_path / "store", batch())
    directory = _only_batch(root)
    (directory / ".DS_Store").write_bytes(b"")
    refused(RefusalReason.FILE_SET, read_batch, directory)
    (directory / ".DS_Store").unlink()
    prices = pd.read_parquet(directory / "prices.parquet")
    prices.loc[0, "close"] = 1.0
    prices.to_parquet(directory / "prices.parquet", index=False)
    refused(RefusalReason.HASH_MISMATCH, read_batch, directory)
    refused(
        RefusalReason.HASH_MISMATCH, load_inference_window, root, FUNDS, H_END, retrieved_by=LATER
    )


def test_foreign_entries_refuse_the_store_and_staging_is_skipped(tmp_path):
    root = stored(tmp_path / "store", batch())
    (root / ".staging-crashed").mkdir()
    assert load_inference_window(root, FUNDS, H_END, retrieved_by=LATER).complete
    (root / "20261014T230000000000Z-0123456789ab").mkdir()  # an F1-style batch name
    refused(
        ProspectiveRefusal.FOREIGN_ENTRY,
        load_inference_window,
        root,
        FUNDS,
        H_END,
        retrieved_by=LATER,
    )


def test_a_symlinked_batch_is_refused(tmp_path):
    root = stored(tmp_path / "store", batch())
    directory = _only_batch(root)
    link = root / ("P" + "1" * 8 + "T" + "1" * 12 + "Z-" + "a" * 12)
    link.symlink_to(directory)
    refused(
        ProspectiveRefusal.FOREIGN_ENTRY,
        load_inference_window,
        root,
        FUNDS,
        H_END,
        retrieved_by=LATER,
    )


# --- load_inference_window ------------------------------------------------------------


def test_the_default_window_is_the_lookback_and_h_end(tmp_path):
    root = stored(tmp_path / "store", batch())
    result = load_inference_window(root, FUNDS, "2026-10-14", retrieved_by=LATER)
    assert result.complete and result.lookback == 63 and result.assets == FUNDS
    assert result.revisions.empty and list(result.revisions.columns) == list(
        inference.REVISION_COLUMNS
    )
    assert len(result.sessions) == 64
    assert result.sessions[0] == WINDOW_START and result.sessions[-1] == H_END
    assert result.forecast_close == H_END
    assert str(result.sessions.dtype) == "datetime64[us, UTC]"
    assert list(result.prices.columns) == list(inference.PRICE_COLUMNS)
    assert list(result.distributions.columns) == list(inference.DISTRIBUTION_COLUMNS)
    assert list(result.splits.columns) == list(inference.SPLIT_COLUMNS)
    assert list(result.missing.columns) == list(inference.MISSING_COLUMNS)
    assert len(result.prices) == 3 * 64
    assert set(result.prices["adjustment_basis"]) == {"unadjusted"}
    agg = result.prices[result.prices["fund_id"] == "AGG"].set_index("session_date")["close"]
    assert agg[H_END] == close_for("AGG", H_END)  # a raw stored value, unchanged
    assert list(result.distributions["ex_date"].dt.strftime("%Y-%m-%d")) == [
        "2026-08-03",
        "2026-09-01",
        "2026-10-01",
        "2026-09-28",
    ]
    assert len(result.batches) == 1 and result.calendar_id == "nyse_full_day_rules_2003_2030_v1"


def test_nothing_after_the_forecast_close_or_before_the_window_is_returned(tmp_path):
    later = batch(end=day("2026-10-30"), retrieved_at=pd.Timestamp("2026-10-31T12:00:00Z"))
    root = stored(tmp_path / "store", later)
    result = load_inference_window(root, FUNDS, "2026-10-20", lookback=20, retrieved_by=LATER)
    close = day("2026-10-20")
    for frame, column in (
        (result.prices, "session_date"),
        (result.distributions, "ex_date"),
        (result.splits, "effective_date"),
    ):
        assert (frame[column] <= close).all() and (frame[column] >= result.sessions[0]).all()
    assert result.prices["session_date"].max() == close
    assert result.sessions[0] == day("2026-09-22")
    served = list(
        zip(result.distributions["fund_id"], result.distributions["ex_date"], strict=True)
    )
    assert served == [("AGG", day("2026-10-01")), ("VTI", day("2026-09-28"))]


@pytest.mark.parametrize(
    ("close", "lookback"),
    [
        ("2026-10-13", 63),
        ("2026-08-14", 63),
        ("2026-07-16", 1),
        # A short lookback inside the window: forecasting at 2026-08-14 would let the
        # lookback sessions score that forecast, so any close before H_END is refused.
        ("2026-08-14", 20),
        ("2026-10-13", 1),
        ("2026-10-14", 64),  # at H_END, a window starting before 2026-07-16
    ],
)
def test_a_forecast_inside_or_reaching_into_the_sealed_period_is_refused(tmp_path, close, lookback):
    root = stored(tmp_path / "store", batch())
    refused(
        ProspectiveRefusal.SEALED_HOLDOUT,
        load_inference_window,
        root,
        FUNDS,
        close,
        lookback,
        retrieved_by=LATER,
    )


def test_a_shorter_lookback_at_h_end_is_allowed(tmp_path):
    root = stored(tmp_path / "store", batch())
    result = load_inference_window(
        root, ("AGG",), date(2026, 10, 14), lookback=20, retrieved_by=LATER
    )
    assert result.sessions[0] == day("2026-09-16") and len(result.sessions) == 21
    assert result.complete


def test_the_latest_batch_covering_a_session_wins_and_never_falls_back(tmp_path):
    first = batch(retrieved_at=AFTER_H_END)
    second = batch(
        start=day("2026-09-01"),
        retrieved_at=pd.Timestamp("2026-10-15T23:00:00Z"),
        offset=1.0,
        skip={("AGG", "2026-09-15")},
    )
    root = stored(tmp_path / "store", first, second)
    result = load_inference_window(root, FUNDS, H_END, retrieved_by=LATER)
    by_key = result.prices.set_index(["fund_id", "session_date"])
    assert by_key.loc[("VTI", day("2026-08-31")), "close"] == close_for("VTI", day("2026-08-31"))
    assert by_key.loc[("VTI", day("2026-09-01")), "close"] == close_for(
        "VTI", day("2026-09-01"), 1.0
    )
    # The second batch covers AGG on 2026-09-15 and lacks it: missing, not the older value.
    assert ("AGG", day("2026-09-15")) not in by_key.index
    assert result.missing.to_dict("records") == [
        {"fund_id": "AGG", "session_date": day("2026-09-15"), "reason": MISSING_SESSION}
    ]
    assert not result.complete and len(result.batches) == 2
    amounts = result.distributions.set_index(["fund_id", "ex_date"])["amount"]
    assert amounts[("AGG", day("2026-08-03"))] == 0.25  # only the first batch covers it
    assert amounts[("AGG", day("2026-09-01"))] == 0.26  # the second batch's observation
    # Every disagreement with the older covering batch is listed, without values.
    revisions = result.revisions
    assert set(revisions["batch_id"]) == {result.batches["batch_id"].iloc[1]}
    removed = revisions[revisions["change"] == "removed"]
    assert removed[["fund_id", "session_date", "table"]].values.tolist() == [
        ["AGG", day("2026-09-15"), "prices"]
    ]
    changed = revisions[revisions["table"] == "distributions"]
    assert sorted(zip(changed["fund_id"], changed["session_date"], strict=True)) == [
        ("AGG", day("2026-09-01")),
        ("AGG", day("2026-10-01")),
        ("VTI", day("2026-09-28")),
    ]
    assert set(changed["change"]) == {"changed"}
    prices_changed = revisions[
        (revisions["table"] == "prices") & (revisions["change"] == "changed")
    ]
    assert len(prices_changed) == 3 * len(nyse_rules.sessions("2026-09-01", H_END)) - 1
    assert "0.25" not in revisions.to_string() and "0.26" not in revisions.to_string()


def test_a_distribution_a_later_batch_no_longer_serves_is_flagged(tmp_path):
    first = batch(retrieved_at=AFTER_H_END)
    second = batch(retrieved_at=pd.Timestamp("2026-10-15T23:00:00Z"), dividends={})
    root = stored(tmp_path / "store", first, second)
    result = load_inference_window(root, FUNDS, H_END, retrieved_by=LATER)
    assert result.complete and result.distributions.empty  # no fallback to the older batch
    removed = result.revisions[result.revisions["table"] == "distributions"]
    assert len(removed) == 4 and set(removed["change"]) == {"removed"}
    earlier = load_inference_window(root, FUNDS, H_END, retrieved_by=AFTER_H_END)
    assert len(earlier.distributions) == 4 and earlier.revisions.empty


def test_a_later_batch_never_changes_an_earlier_forecasts_inputs(tmp_path):
    first = batch(retrieved_at=AFTER_H_END)
    later = batch(retrieved_at=pd.Timestamp("2026-10-21T23:00:00Z"), offset=1.0, dividends={})
    alone = load_inference_window(
        stored(tmp_path / "alone", first), FUNDS, H_END, retrieved_by=AFTER_H_END
    )
    both = load_inference_window(
        stored(tmp_path / "both", first, later), FUNDS, H_END, retrieved_by=AFTER_H_END
    )
    for name in ("prices", "distributions", "splits", "missing", "batches", "revisions"):
        assert getattr(both, name).equals(getattr(alone, name)), name


def test_retrieved_by_reproduces_an_earlier_view(tmp_path):
    first = batch(retrieved_at=AFTER_H_END)
    second = batch(retrieved_at=pd.Timestamp("2026-10-15T23:00:00Z"), offset=1.0)
    root = stored(tmp_path / "store", first, second)
    now = load_inference_window(root, FUNDS, H_END, retrieved_by=LATER)
    then = load_inference_window(root, FUNDS, H_END, retrieved_by=AFTER_H_END)
    assert set(now.prices["batch_id"]) != set(then.prices["batch_id"])
    assert list(then.batches["retrieved_at"]) == [AFTER_H_END]
    nothing = load_inference_window(
        root, FUNDS, H_END, retrieved_by=pd.Timestamp("2026-10-14T00:00:00Z")
    )
    assert nothing.prices.empty and set(nothing.missing["reason"]) == {NO_BATCH}
    assert len(nothing.missing) == 3 * 64 and nothing.batches.empty


def test_an_asset_without_a_batch_is_reported_missing(tmp_path):
    root = stored(tmp_path / "store", batch(funds=("AGG",)))
    result = load_inference_window(root, ("AGG", "VTI"), H_END, retrieved_by=LATER)
    assert set(result.prices["fund_id"]) == {"AGG"}
    missing = result.missing
    assert set(missing["fund_id"]) == {"VTI"} and set(missing["reason"]) == {NO_BATCH}
    assert len(missing) == 64


def test_an_empty_store_gives_typed_empty_tables(tmp_path):
    root = new_store(tmp_path / "store")
    result = load_inference_window(root, ("AGG",), H_END, retrieved_by=LATER)
    assert result.prices.empty and result.distributions.empty and result.splits.empty
    assert str(result.prices["session_date"].dtype) == "datetime64[us, UTC]"
    assert str(result.prices["close"].dtype) == "float64"
    assert len(result.missing) == 64


@pytest.mark.parametrize(
    ("assets", "close", "lookback"),
    [
        ((), H_END, 63),
        ("AGG", H_END, 63),
        (("AGG", "AGG"), H_END, 63),
        (("AGG",), "2026-10-17", 63),  # a Saturday
        (("AGG",), "2026-10-14T00:00", 63),
        (("AGG",), pd.Timestamp("2026-10-14 15:00", tz="UTC"), 63),
        (("AGG",), "2031-01-02", 63),
        (("AGG",), H_END, 0),
        (("AGG",), H_END, True),
        (("AGG",), H_END, 63.0),
    ],
)
def test_invalid_requests_are_refused(tmp_path, assets, close, lookback):
    root = stored(tmp_path / "store", batch())
    refused(
        RefusalReason.INVALID_REQUEST,
        load_inference_window,
        root,
        assets,
        close,
        lookback,
        retrieved_by=LATER,
    )


def test_retrieved_by_is_required_and_must_be_utc(tmp_path):
    root = stored(tmp_path / "store", batch())
    with pytest.raises(TypeError):
        load_inference_window(root, FUNDS, H_END)  # no default: the decision time is explicit
    refused(
        RefusalReason.INVALID_TIMESTAMP,
        load_inference_window,
        root,
        FUNDS,
        H_END,
        retrieved_by=pd.Timestamp("2026-10-15"),
    )
