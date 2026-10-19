"""
The launch-readiness report (read-only): every weekday from the registration
time through the first record, classified as recorded, refused (with the
job's reason and its evidence), not recorded (operator-missed: eligible, no
record inside its window), market closed or pending; and, before the
registration pull request is merged, whether the store already holds the
verified window ending at the latest close. Nothing here changes the
session-1 rule, records anything or writes to the ledger.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from forecast_prospective_helpers import (
    ASSETS,
    exchange_calendar,
    prospective_store,
    record,
    register,
    without_close,
)

from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.forecasting.prospective import ledger, readiness

LOOKBACK = 63
FRIDAY_EVENING = pd.Timestamp("2026-10-16T22:00:00Z")  # 10-16's deadline is Monday 09:30


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "ledger.jsonl"


def _closes(path, store, now, loader=load_inference_window):
    return readiness.closes(
        ledger.read_ledger(path),
        calendar=exchange_calendar,
        now=pd.Timestamp(now),
        load_window=loader,
        store_root=store,
        assets=ASSETS,
        lookback=LOOKBACK,
    )


def _statuses(rows):
    return [(row.day, row.status, row.reason) for row in rows]


def test_eligible_closes_without_a_record_are_operator_missed(
    prospective_frozen, ledger_path, prospective_store_root
):
    register(prospective_frozen.models(), ledger_path)  # 2026-10-14 14:00 UTC
    before = ledger_path.read_bytes()
    rows = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING)
    assert _statuses(rows) == [
        ("2026-10-14", "not_recorded", "operator_missed"),
        ("2026-10-15", "not_recorded", "operator_missed"),
        ("2026-10-16", "pending", None),
    ]
    assert rows[0].evidence["retrieved_by"] == "2026-10-15T13:29:59+00:00"  # just before 09:30
    assert rows[0].evidence["batches"]
    assert ledger_path.read_bytes() == before  # read-only


def test_the_report_ends_at_the_first_record(
    prospective_frozen, ledger_path, prospective_store_root
):
    register(prospective_frozen.models(), ledger_path)
    record(
        prospective_frozen,
        ledger_path,
        prospective_store_root,
        as_of="2026-10-15",
        created=pd.Timestamp("2026-10-15T23:00:00Z"),
    )
    rows = _closes(ledger_path, prospective_store_root, pd.Timestamp("2026-10-23T23:00:00Z"))
    assert _statuses(rows) == [
        ("2026-10-14", "not_recorded", "operator_missed"),
        ("2026-10-15", "recorded", None),
    ]
    assert rows[1].evidence == {"entry": 1, "kind": "forecast"}


def test_a_close_not_after_the_registration_is_refused(
    prospective_frozen, ledger_path, prospective_store_root
):
    register(
        prospective_frozen.models(),
        ledger_path,
        registered_at=pd.Timestamp("2026-10-14T22:30:00Z"),  # 18:30 New York
        recorded_at=pd.Timestamp("2026-10-14T22:45:00Z"),
    )
    rows = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING)
    assert _statuses(rows)[0] == ("2026-10-14", "refused", "before_registration")
    assert rows[0].evidence == {
        "registered_at": "2026-10-14T22:30:00+00:00",
        "close_available_at": "2026-10-14T22:00:00+00:00",
    }
    assert _statuses(rows)[1] == ("2026-10-15", "not_recorded", "operator_missed")


def test_a_close_without_every_funds_window_is_refused_with_its_evidence(
    prospective_frozen, ledger_path, tmp_path
):
    gap = prospective_store(tmp_path / "gap", edit=without_close("VTI", ["2026-09-15"]))
    register(prospective_frozen.models(), ledger_path)
    rows = _closes(ledger_path, gap, FRIDAY_EVENING)
    assert _statuses(rows)[:2] == [
        ("2026-10-14", "refused", "not_first_eligible_close"),
        ("2026-10-15", "refused", "not_first_eligible_close"),
    ]
    assert rows[0].evidence["incomplete"] == {"VTI": ["2026-09-15"]}
    assert rows[0].evidence["unchecked_batches"] == []


def test_a_close_from_batches_without_a_spot_check_is_refused(
    prospective_frozen, ledger_path, prospective_store_root
):
    def unchecked(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        win = load_inference_window(
            store_root, assets, forecast_close, lookback, retrieved_by=retrieved_by
        )
        return dataclasses.replace(win, batches=win.batches.assign(anchors_sha256=None))

    register(prospective_frozen.models(), ledger_path)
    rows = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING, loader=unchecked)
    assert rows[0].status == "refused" and rows[0].reason == "not_first_eligible_close"
    assert rows[0].evidence["incomplete"] == {} and rows[0].evidence["unchecked_batches"]


def test_a_close_the_store_cannot_serve_is_refused_with_the_stores_reason(
    prospective_frozen, ledger_path, tmp_path
):
    register(prospective_frozen.models(), ledger_path)
    rows = _closes(ledger_path, tmp_path / "not_a_store", FRIDAY_EVENING)
    assert rows[0].status == "refused" and rows[0].reason == "window_refused"
    assert rows[0].evidence["refusal"] == "not_prospective_store"


def test_a_weekday_holiday_is_market_closed(prospective_frozen, ledger_path, tmp_path):
    register(
        prospective_frozen.models(),
        ledger_path,
        registered_at=pd.Timestamp("2026-11-26T15:00:00Z"),  # Thanksgiving, a Thursday
        recorded_at=pd.Timestamp("2026-11-26T16:00:00Z"),
    )
    rows = _closes(ledger_path, tmp_path / "unused", pd.Timestamp("2026-11-27T20:00:00Z"))
    assert _statuses(rows) == [
        ("2026-11-26", "market_closed", None),
        ("2026-11-27", "pending", None),
    ]


def test_the_rows_are_structured(prospective_frozen, ledger_path, prospective_store_root):
    register(prospective_frozen.models(), ledger_path)
    (row, *_) = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING)
    assert row.to_json()["status"] == "not_recorded"
    assert row.line().startswith("close=2026-10-14 status=not_recorded reason=operator_missed")


def test_an_unregistered_ledger_has_no_readiness(tmp_path, prospective_store_root):
    assert _closes(tmp_path / "none.jsonl", prospective_store_root, FRIDAY_EVENING) == ()


# --- before the registration pull request is merged ---------------------------------------


def _pre_merge(store, now, loader=load_inference_window):
    return readiness.pre_merge(
        load_window=loader,
        store_root=store,
        assets=ASSETS,
        calendar=exchange_calendar,
        now=pd.Timestamp(now),
        lookback=LOOKBACK,
    )


def test_before_the_merge_the_latest_verified_window_is_reported(prospective_store_root):
    report = _pre_merge(prospective_store_root, "2026-10-15T23:00:00Z")
    assert report.close == "2026-10-15"
    assert report.ready is True
    assert report.evidence["incomplete"] == {} and report.evidence["unchecked_batches"] == []
    assert report.evidence["retrieved_by"] == "2026-10-15T23:00:00+00:00"


def test_before_the_merge_a_close_not_yet_available_is_not_the_latest(prospective_store_root):
    # 17:00 New York on 10-15: the latest available close is 10-14's.
    assert _pre_merge(prospective_store_root, "2026-10-15T21:00:00Z").close == "2026-10-14"


@pytest.mark.parametrize("case", ["gap", "not_retrieved_yet"])
def test_before_the_merge_an_unready_store_is_reported(tmp_path, prospective_store_root, case):
    if case == "gap":
        store = prospective_store(tmp_path / "gap", edit=without_close("VTI", ["2026-09-15"]))
        report = _pre_merge(store, "2026-10-14T23:00:00Z")
        assert report.evidence["incomplete"] == {"VTI": ["2026-09-15"]}
    else:  # 18:10 New York: the close counts as available, but no batch was retrieved yet
        report = _pre_merge(prospective_store_root, "2026-10-14T22:10:00Z")
        assert set(report.evidence["incomplete"]) == set(ASSETS)
    assert report.ready is False


# --- boundaries, late batches and the registration entry -----------------------------------


@pytest.mark.parametrize(
    ("now", "pending"), [("2026-10-15T13:29:59Z", True), ("2026-10-15T13:30:00Z", False)]
)
def test_a_close_is_pending_until_its_deadline(
    prospective_frozen, ledger_path, prospective_store_root, now, pending
):
    register(prospective_frozen.models(), ledger_path)
    (first, *_) = _closes(ledger_path, prospective_store_root, now)
    assert (first.status == "pending") is pending


@pytest.mark.parametrize(
    ("registered_at", "refused"), [("2026-10-14T22:00:00Z", True), ("2026-10-14T21:59:00Z", False)]
)
def test_a_close_at_exactly_the_registration_time_is_refused(
    prospective_frozen, ledger_path, prospective_store_root, registered_at, refused
):
    when = pd.Timestamp(registered_at)
    register(prospective_frozen.models(), ledger_path, registered_at=when, recorded_at=when)
    (first, *_) = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING)
    assert (first.reason == "before_registration") is refused


def test_a_close_whose_window_ended_before_the_entry_was_appended_is_refused(
    prospective_frozen, ledger_path, prospective_store_root
):
    # Merged 14:00 UTC on 10-14, but `register` ran at 14:00 UTC on 10-15: the job could
    # not record 10-14 (its deadline was 13:30 UTC on 10-15) before the entry existed.
    register(
        prospective_frozen.models(),
        ledger_path,
        recorded_at=pd.Timestamp("2026-10-15T14:00:00Z"),
    )
    rows = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING)
    assert _statuses(rows)[:2] == [
        ("2026-10-14", "refused", "before_registration"),
        ("2026-10-15", "not_recorded", "operator_missed"),
    ]
    assert rows[0].evidence == {
        "registration_entry_appended": "2026-10-15T14:00:00+00:00",
        "record_deadline": "2026-10-15T13:30:00+00:00",
    }


def test_a_batch_retrieved_after_a_deadline_never_makes_that_close_eligible(
    prospective_frozen, ledger_path, prospective_store_root
):
    # The fixture store has no retrieval from 10-19 to 10-21: the 10-22 batch covers them,
    # but arrived after their deadlines.
    register(prospective_frozen.models(), ledger_path)
    rows = _closes(ledger_path, prospective_store_root, pd.Timestamp("2026-10-23T23:00:00Z"))
    assert _statuses(rows) == [
        ("2026-10-14", "not_recorded", "operator_missed"),
        ("2026-10-15", "not_recorded", "operator_missed"),
        ("2026-10-16", "not_recorded", "operator_missed"),
        ("2026-10-19", "refused", "not_first_eligible_close"),
        ("2026-10-20", "refused", "not_first_eligible_close"),
        ("2026-10-21", "refused", "not_first_eligible_close"),
        ("2026-10-22", "not_recorded", "operator_missed"),
        ("2026-10-23", "pending", None),
    ]
    assert rows[3].evidence["incomplete"] == {fund: ["2026-10-19"] for fund in ASSETS}


def test_a_loader_leaking_later_data_is_raised_not_classified(
    prospective_frozen, ledger_path, prospective_store_root
):
    from stock_agent.model_validation.audit import LeakageError

    def leaky(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        win = load_inference_window(
            store_root, assets, forecast_close, lookback, retrieved_by=retrieved_by
        )
        late = win.batches.assign(retrieved_at=retrieved_by + pd.Timedelta(seconds=1))
        return dataclasses.replace(win, batches=late)

    register(prospective_frozen.models(), ledger_path)
    with pytest.raises(LeakageError):
        _closes(ledger_path, prospective_store_root, FRIDAY_EVENING, loader=leaky)


def test_a_window_breaking_the_contract_is_refused_with_its_reason(
    prospective_frozen, ledger_path, prospective_store_root
):
    def short(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        win = load_inference_window(
            store_root, assets, forecast_close, lookback, retrieved_by=retrieved_by
        )
        return dataclasses.replace(win, sessions=win.sessions[1:])

    register(prospective_frozen.models(), ledger_path)
    (first, *_) = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING, loader=short)
    assert (first.status, first.reason) == ("refused", "window_refused")
    assert first.evidence["refusal"].startswith("window_contract")


def test_a_window_without_batch_lineage_is_never_eligible(
    prospective_frozen, ledger_path, prospective_store_root
):
    def unattributed(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        win = load_inference_window(
            store_root, assets, forecast_close, lookback, retrieved_by=retrieved_by
        )
        return dataclasses.replace(win, batches=win.batches.iloc[0:0])

    register(prospective_frozen.models(), ledger_path)
    (first, *_) = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING, loader=unattributed)
    assert (first.status, first.reason) == ("refused", "not_first_eligible_close")
    assert first.evidence["no_batch_lineage"] is True


def test_a_close_eligible_earlier_in_its_window_stays_operator_missed(
    prospective_frozen, ledger_path, tmp_path
):
    # The 18:30 batch holds every close; a second batch at 19:30 the same evening drops
    # VTI's 10-14 close. At 09:29:59 the later batch wins, but the job could have recorded
    # 10-14 between 18:30 and 19:30.
    from forecast_prospective_helpers import prospective_batch, prospective_tables
    from prospective_helpers import new_store

    from stock_agent.data.prospective.store import ingest_prospective

    root = new_store(tmp_path / "two_vintages")
    full = prospective_tables()
    ingest_prospective(root, prospective_batch(full, "2026-10-14"))
    dropped = without_close("VTI", ["2026-10-14"])("2026-10-14", tuple(t.copy() for t in full))
    later = dataclasses.replace(
        prospective_batch(dropped, "2026-10-14"),
        retrieved_at=pd.Timestamp("2026-10-14T23:30:00Z").as_unit("us"),
        raw_payload=b"a later vintage",
    )
    ingest_prospective(root, later)
    register(prospective_frozen.models(), ledger_path)
    (first, *_) = _closes(ledger_path, root, FRIDAY_EVENING)
    assert (first.day, first.status, first.reason) == (
        "2026-10-14",
        "not_recorded",
        "operator_missed",
    )
    # Just before the later batch arrived, the first one was the whole window.
    assert first.evidence["retrieved_by"] == "2026-10-14T23:29:59+00:00"


# --- the proposed ETF-R1 view: a fixed start, missed sessions reported (not in force) --------


def _fixed(path, now):
    return readiness.fixed_start(
        ledger.read_ledger(path), calendar=exchange_calendar, now=pd.Timestamp(now)
    )


def test_under_etf_r1_the_start_is_the_first_close_after_the_registration(
    prospective_frozen, ledger_path, prospective_store_root
):
    register(prospective_frozen.models(), ledger_path)  # 14:00 UTC on 10-14
    record(
        prospective_frozen,
        ledger_path,
        prospective_store_root,
        as_of="2026-10-15",
        created=pd.Timestamp("2026-10-15T23:00:00Z"),
    )
    view = _fixed(ledger_path, "2026-10-19T14:00:00Z")  # 10-16's deadline has passed
    assert view.session_1 == "2026-10-14"  # not 10-15, the first record
    assert [(r.day, r.status, r.evidence["number"]) for r in view.rows] == [
        ("2026-10-14", "missed", 1),
        ("2026-10-15", "recorded", 2),
        ("2026-10-16", "missed", 3),
        ("2026-10-19", "pending", 4),
    ]
    assert view.missed == ("2026-10-14", "2026-10-16")


def test_under_etf_r1_the_start_never_moves(
    prospective_frozen, ledger_path, prospective_store_root
):
    register(prospective_frozen.models(), ledger_path)
    before = _fixed(ledger_path, "2026-10-16T22:00:00Z").session_1
    record(
        prospective_frozen,
        ledger_path,
        prospective_store_root,
        as_of="2026-10-16",
        created=pd.Timestamp("2026-10-16T23:00:00Z"),
    )
    after = _fixed(ledger_path, "2026-10-23T23:00:00Z")
    assert before == after.session_1 == "2026-10-14"


def test_under_etf_r1_a_registration_after_18_00_starts_the_next_session(
    prospective_frozen, ledger_path
):
    register(
        prospective_frozen.models(),
        ledger_path,
        registered_at=pd.Timestamp("2026-10-14T22:00:00Z"),  # exactly 18:00 New York
        recorded_at=pd.Timestamp("2026-10-14T22:10:00Z"),
    )
    assert _fixed(ledger_path, "2026-10-16T22:00:00Z").session_1 == "2026-10-15"


def test_under_etf_r1_a_record_outside_its_window_is_missed(prospective_frozen, ledger_path):
    register(prospective_frozen.models(), ledger_path)
    late = {"as_of_session": "2026-10-14", "rows": [], "refusals": []}  # written by hand
    ledger.append_entry(
        ledger_path, "forecast", late, recorded_at=pd.Timestamp("2026-10-15T14:00:00Z")
    )
    view = _fixed(ledger_path, "2026-10-15T14:30:00Z")
    assert (view.rows[0].day, view.rows[0].status, view.rows[0].reason) == (
        "2026-10-14",
        "missed",
        "record_outside_window",
    )


# --- review follow-ups ------------------------------------------------------------------------


def _two_vintages(tmp_path):
    """A complete batch at 18:30 New York on 10-14, and one at 19:30 dropping VTI's 10-14 close."""

    from forecast_prospective_helpers import prospective_batch, prospective_tables
    from prospective_helpers import new_store

    from stock_agent.data.prospective.store import ingest_prospective

    root = new_store(tmp_path / "two_vintages")
    full = prospective_tables()
    ingest_prospective(root, prospective_batch(full, "2026-10-14"))
    dropped = without_close("VTI", ["2026-10-14"])("2026-10-14", tuple(t.copy() for t in full))
    ingest_prospective(
        root,
        dataclasses.replace(
            prospective_batch(dropped, "2026-10-14"),
            retrieved_at=pd.Timestamp("2026-10-14T23:30:00Z").as_unit("us"),
            raw_payload=b"a later vintage",
        ),
    )
    return root


def test_eligibility_never_uses_a_time_before_the_entry_was_appended(
    prospective_frozen, ledger_path, tmp_path
):
    # The entry was appended at 13:00 UTC on 10-15, inside 10-14's window: only the later
    # vintage (no VTI close) was in the store from then on.
    register(
        prospective_frozen.models(), ledger_path, recorded_at=pd.Timestamp("2026-10-15T13:00:00Z")
    )
    (first, *_) = _closes(ledger_path, _two_vintages(tmp_path), FRIDAY_EVENING)
    assert (first.status, first.reason) == ("refused", "not_first_eligible_close")
    assert all(
        t >= "2026-10-15T13:00:00"
        for t in first.evidence.get("checked_at", [first.evidence["retrieved_by"]])
    )


def test_an_entry_appended_exactly_at_the_deadline_refuses_that_close(
    prospective_frozen, ledger_path, prospective_store_root
):
    register(
        prospective_frozen.models(), ledger_path, recorded_at=pd.Timestamp("2026-10-15T13:30:00Z")
    )
    (first, *_) = _closes(ledger_path, prospective_store_root, FRIDAY_EVENING)
    assert (first.status, first.reason) == ("refused", "before_registration")


@pytest.mark.parametrize(
    ("recorded_at", "status", "reason"),
    [
        ("2026-10-14T21:59:59Z", "missed", "record_outside_window"),  # before 18:00 New York
        ("2026-10-14T22:00:00Z", "recorded", None),
        ("2026-10-15T13:29:59Z", "recorded", None),
        ("2026-10-15T13:30:00Z", "missed", "record_outside_window"),  # at the deadline
    ],
)
def test_under_etf_r1_a_record_counts_only_inside_its_window(
    prospective_frozen, ledger_path, recorded_at, status, reason
):
    register(prospective_frozen.models(), ledger_path)
    payload = {"as_of_session": "2026-10-14", "rows": [], "refusals": []}
    ledger.append_entry(ledger_path, "forecast", payload, recorded_at=pd.Timestamp(recorded_at))
    (first, *_) = _fixed(ledger_path, "2026-10-16T22:00:00Z").rows
    assert (first.status, first.reason) == (status, reason)


def test_under_etf_r1_a_session_without_any_record_is_missed_as_such(
    prospective_frozen, ledger_path
):
    register(prospective_frozen.models(), ledger_path)
    (first, *_) = _fixed(ledger_path, "2026-10-16T22:00:00Z").rows
    assert (first.status, first.reason) == ("missed", "no_record")


@pytest.mark.parametrize(
    ("registered_at", "session_1"),
    [
        ("2026-11-02T22:30:00Z", "2026-11-02"),  # 17:30 EST: before that day's 18:00
        ("2026-10-30T22:30:00Z", "2026-11-02"),  # 18:30 EDT on a Friday: the next Monday
        ("2026-11-25T22:00:00Z", "2026-11-25"),  # 17:00 EST, the day before Thanksgiving
        ("2026-11-26T00:00:00Z", "2026-11-27"),  # 19:00 EST: Thanksgiving is closed
    ],
)
def test_under_etf_r1_session_1_follows_new_york_time_and_holidays(
    prospective_frozen, ledger_path, registered_at, session_1
):
    when = pd.Timestamp(registered_at)
    register(prospective_frozen.models(), ledger_path, registered_at=when, recorded_at=when)
    # Viewed the moment it is registered: session 1 is known before its date arrives.
    assert _fixed(ledger_path, when).session_1 == session_1


def test_under_etf_r1_a_refusal_record_shows_its_kind(prospective_frozen, ledger_path):
    register(prospective_frozen.models(), ledger_path)
    payload = {"as_of_session": "2026-10-14", "refusals": []}
    ledger.append_entry(
        ledger_path, "refusal", payload, recorded_at=pd.Timestamp("2026-10-14T23:00:00Z")
    )
    (first, *_) = _fixed(ledger_path, "2026-10-16T22:00:00Z").rows
    assert (first.status, first.evidence["kind"]) == ("recorded", "refusal")


@pytest.mark.parametrize(
    ("now", "close"),
    [("2026-10-15T21:59:59Z", "2026-10-14"), ("2026-10-15T22:00:00Z", "2026-10-15")],
)
def test_before_the_merge_the_latest_close_turns_at_18_00(prospective_store_root, now, close):
    assert _pre_merge(prospective_store_root, now).close == close


def test_before_the_merge_a_refusing_calendar_is_reported_not_raised(prospective_store_root):
    from stock_agent.calendars.nyse import CalendarRefused

    def refusing(start, end):
        raise CalendarRefused("outside_coverage", "after the rules calendar")

    report = readiness.pre_merge(
        load_window=load_inference_window,
        store_root=prospective_store_root,
        assets=ASSETS,
        calendar=refusing,
        now=NOW_AFTER,
        lookback=LOOKBACK,
    )
    assert (report.ready, report.evidence["refusal"]) == (False, "calendar_refused")


NOW_AFTER = pd.Timestamp("2026-10-15T23:00:00Z")
