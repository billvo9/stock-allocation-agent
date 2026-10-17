"""
The Stock Lab forecast record's integrity rules: no backdated forecast, no
target that had opened before its forecast was made, at most one live
forecast per (stock, as-of session), the kept ledger head, and Model
Health's reading of a run that is not authorized yet. Synthetic data only.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from test_stock_lab_ledger import AS_OF, MADE, PRICES, SESSIONS, T0, T1, _forecast, _setup, daily

from stock_agent.health import fixtures as health_fixtures
from stock_agent.health import stock_lab as stock_lab_health
from stock_agent.health.inputs import load_latest
from stock_agent.health.model import HealthState
from stock_agent.health.stock_lab import ARTIFACT, stock_lab_checks
from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab import scoring, synthetic
from stock_agent.stock_lab import store as store_module
from stock_agent.stock_lab.contracts import DataClass, SymbolReason, SymbolStatus
from stock_agent.stock_lab.job import NOT_AUTHORIZED, record_run
from stock_agent.stock_lab.replay import SimulatedClock
from stock_agent.stock_lab.store import DB_NAME, GENESIS, StockLabStore, StoreRefused
from stock_agent.stock_lab.symbols import Classification, ListingRow

BARS = synthetic.demo_bars("DEMO_CALM")
HOUR = pd.Timedelta(hours=1)


def _store(path, start) -> tuple[StockLabStore, SimulatedClock]:
    clock = SimulatedClock(start)
    return StockLabStore(path / DB_NAME, clock=clock), clock


def _record_without_the_timing_rules(store: StockLabStore, record: dict, recorded_at) -> None:
    """Insert a forecast as a store without the timing rules did: the same row and ledger entry."""

    text = artifact.canonical(record)
    with store._write() as connection:
        connection.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record["forecast_id"],
                record["symbol"],
                record["as_of_session"],
                record["created_at"],
                record["data_class"],
                record["supersedes"],
                text,
            ),
        )
        store_module._append_ledger(
            connection, "forecast", record["forecast_id"], text, recorded_at
        )


def _reasons(store: StockLabStore) -> dict[str, str | None]:
    return {score["target"]: score["reason"] for score in store.scores("DEMO_CALM")}


# --- timing ---


def test_a_backdated_forecast_is_refused_once_its_outcomes_are_stored(tmp_path):
    # The review's case: outcomes stored an hour ago, then a forecast dated two days back.
    store, _ = _store(tmp_path, T1)
    store.record_prices("DEMO_CALM", BARS.iloc[: AS_OF + 6], retrieved_at=T1 - HOUR, **PRICES)
    with pytest.raises(StoreRefused, match="backdating"):
        store.record_forecast(_forecast(T1 - pd.Timedelta(days=2)))
    assert store.forecasts("DEMO_CALM") == () and store.verify_ledger() == (0, None)
    # Backdating alone is refused too: an hour back, before any target opened, nothing stored.
    early, _ = _store(tmp_path / "early", MADE + HOUR)
    with pytest.raises(StoreRefused, match="backdating"):
        early.record_forecast(_forecast(MADE))


def test_a_forecast_cannot_predate_data_already_stored_for_its_stock(tmp_path):
    store, _ = _store(tmp_path, MADE)
    store.record_prices("DEMO_CALM", BARS.iloc[: AS_OF + 1], retrieved_at=MADE, **PRICES)
    with pytest.raises(StoreRefused, match="predate"):
        store.record_forecast(_forecast(MADE - pd.Timedelta(minutes=1)))  # within the clock skew
    store.record_forecast(_forecast(MADE))
    assert len(store.forecasts("DEMO_CALM")) == 1


def test_a_forecast_must_be_made_before_its_first_target_session_opens(tmp_path):
    first = SESSIONS.iloc[AS_OF + 1].strftime("%Y-%m-%d")
    opens = pd.Timestamp(artifact.session_open(first)).tz_convert("UTC")
    assert opens.hour in (13, 14)  # 09:30 in New York
    late, _ = _store(tmp_path / "late", opens)
    with pytest.raises(StoreRefused, match="before its first target session opens"):
        late.record_forecast(_forecast(opens))
    in_time, _ = _store(tmp_path / "in_time", opens - pd.Timedelta(microseconds=1))
    in_time.record_forecast(_forecast(opens - pd.Timedelta(microseconds=1)))
    assert len(in_time.forecasts("DEMO_CALM")) == 1


def test_a_price_batch_retrieved_in_the_future_is_refused(tmp_path):
    store, _ = _store(tmp_path, T0)
    with pytest.raises(StoreRefused, match="future"):
        store.record_prices(
            "DEMO_CALM", BARS.iloc[:10], retrieved_at=T0 + pd.Timedelta(microseconds=1), **PRICES
        )
    assert store.batches("DEMO_CALM") == ()


def test_scoring_judges_what_a_forecast_saw_by_when_the_store_recorded_it(tmp_path):
    # A store written before the timing rules: the outcomes were stored, and an hour later a
    # forecast claiming to be made before them was recorded. Its chain verifies, but none of
    # its targets may be scored on outcomes the store already held when it recorded them.
    store, clock = _store(tmp_path, T0)
    store.record_prices("DEMO_CALM", BARS.iloc[: AS_OF + 1], retrieved_at=T0, **PRICES)
    clock.advance(T1)
    store.record_prices("DEMO_CALM", BARS.iloc[AS_OF + 1 : AS_OF + 6], retrieved_at=T1, **PRICES)
    clock.advance(T1 + HOUR)
    _record_without_the_timing_rules(store, _forecast(MADE), clock())
    assert store.verify_ledger()[1] is None
    scoring.score_store(store, ["DEMO_CALM"])
    reasons = _reasons(store)
    assert len(reasons) == 6 and set(reasons.values()) == {"outcome_retrieved_before_forecast"}


def test_a_target_that_had_opened_when_the_forecast_was_made_is_never_scored(tmp_path):
    # The review's other case: a forecast made in 2026 on 2019 sessions stored only afterwards.
    late = pd.Timestamp("2026-10-01T22:00:00Z")
    store, clock = _store(tmp_path, late - HOUR)
    store.record_prices("DEMO_CALM", BARS.iloc[: AS_OF + 1], retrieved_at=late - HOUR, **PRICES)
    clock.advance(late)
    _record_without_the_timing_rules(store, _forecast(late), late)
    clock.advance(late + pd.Timedelta(days=1))
    store.record_prices(
        "DEMO_CALM", BARS.iloc[AS_OF + 1 : AS_OF + 6], retrieved_at=clock(), **PRICES
    )
    scoring.score_store(store, ["DEMO_CALM"])
    reasons = _reasons(store)
    assert len(reasons) == 6 and set(reasons.values()) == {"forecast_after_target_opened"}


def test_a_target_session_on_the_as_of_session_is_refused():
    record = _forecast(MADE)
    shifted = {
        **record,
        "target_sessions": [record["as_of_session"], *record["target_sessions"][:4]],
    }
    shifted["daily"] = [
        {**row, "session": session}
        for row, session in zip(record["daily"], shifted["target_sessions"], strict=True)
    ]
    shifted["forecast_id"] = artifact.forecast_id(shifted)
    with pytest.raises(artifact.ForecastRefused, match="follow the as-of session"):
        artifact.check(shifted)


# --- supersession ---


def test_one_live_forecast_per_stock_and_as_of_session(tmp_path):
    store, _ = _store(tmp_path, MADE)
    store.record_prices("DEMO_CALM", BARS.iloc[: AS_OF + 1], retrieved_at=T0, **PRICES)
    first = _forecast(MADE)
    store.record_forecast(first)
    with pytest.raises(StoreRefused, match="already has a live forecast"):
        store.record_forecast(_forecast(MADE, cumulative=(0.3, 0.5, 0.2)))
    with pytest.raises(StoreRefused, match="does not hold"):
        store.record_forecast(_forecast(MADE, cumulative=(0.3, 0.5, 0.2), supersedes="0" * 64))
    with pytest.raises(StoreRefused, match="same stock, as-of session"):
        store.record_forecast(_forecast(MADE, as_of=AS_OF + 1, supersedes=first["forecast_id"]))
    recalculated = _forecast(MADE, cumulative=(0.3, 0.5, 0.2), supersedes=first["forecast_id"])
    store.record_forecast(recalculated)
    with pytest.raises(StoreRefused, match="already replaced"):
        store.record_forecast(
            _forecast(MADE, cumulative=(0.25, 0.5, 0.25), supersedes=first["forecast_id"])
        )
    assert len(store.forecasts("DEMO_CALM")) == 2  # the replaced forecast stays
    assert store.superseded_ids() == {first["forecast_id"]}
    as_of = first["as_of_session"]
    assert store.live_forecast("DEMO_CALM", as_of)["forecast_id"] == recalculated["forecast_id"]
    assert store.latest_forecast("DEMO_CALM")["forecast_id"] == recalculated["forecast_id"]


def test_a_superseded_forecast_is_left_out_of_the_rolling_score(tmp_path):
    store, clock = _store(tmp_path, MADE)
    store.record_prices("DEMO_CALM", BARS.iloc[: AS_OF + 1], retrieved_at=T0, **PRICES)
    first = _forecast(MADE)
    store.record_forecast(first)
    store.record_forecast(
        _forecast(MADE, cumulative=(0.3, 0.5, 0.2), supersedes=first["forecast_id"])
    )
    clock.advance(T1)
    store.record_prices("DEMO_CALM", BARS.iloc[AS_OF + 1 : AS_OF + 6], retrieved_at=T1, **PRICES)
    scoring.score_store(store, ["DEMO_CALM"])
    scores = store.scores("DEMO_CALM")
    assert sum(s["status"] == "scored" and s["target"] != "cum" for s in scores) == 10
    summary = scoring.rolling_summary(scores, superseded=store.superseded_ids())
    assert summary["n"] == 5 and summary["sessions"] == 5


# --- the ledger chain and the kept head ---


def _tamper(store: StockLabStore, *statements: str) -> None:
    """Change the store outside the Lab: drop the write-once triggers, then run `statements`."""

    connection = sqlite3.connect(store.path)
    for (trigger,) in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'"):
        connection.execute(f"DROP TRIGGER {trigger}")
    for statement in statements:
        connection.execute(statement)
    connection.commit()
    connection.close()


def test_an_entry_rehashed_to_name_another_previous_entry_breaks_the_chain(tmp_path):
    store, _, _ = _setup(tmp_path, known=100, later=slice(100, 105))
    scoring.score_store(store, ["DEMO_CALM"])
    connection = sqlite3.connect(store.path)
    seq, kind, ref, recorded_at, payload = connection.execute(
        "SELECT seq, kind, ref, recorded_at, payload_sha256 FROM ledger_entries WHERE seq = 2"
    ).fetchone()
    connection.close()
    other = "1" * 64  # the entry's own hash is recomputed, so only the link shows the change
    rehashed = store_module._entry_sha(seq, kind, ref, recorded_at, payload, other)
    _tamper(
        store,
        f"UPDATE ledger_entries SET prev_entry_sha256 = '{other}', "
        f"entry_sha256 = '{rehashed}' WHERE seq = 2",
    )
    _, problem = store.verify_ledger()
    assert problem == "entry 2 does not name the previous entry"


def _day(store, clock, tmp_path, number: int, kept, tickers=("DEMO_CALM",)) -> dict:
    when = T1 + pd.Timedelta(days=number)
    clock.advance(when)
    path = record_run(
        store,
        run_id=f"day-{number}",
        started=when,
        finished=when,
        status="ok",
        reason=None,
        expected=SESSIONS.iloc[AS_OF + 5],
        tickers=list(tickers),
        data_class=DataClass.SYNTHETIC,
        health_inputs=tmp_path / "inputs",
        sessions=lambda a, b: pd.bdate_range(a, b, tz="UTC"),
        calendar_id="synthetic_weekdays",
        kept_head=kept,
    )
    return load_latest(path.parent, ARTIFACT).document


def _ledger_check(tmp_path):
    checks = stock_lab_checks(tmp_path / "inputs" / ARTIFACT, allow_synthetic=True)
    check = next(c for c in checks if c.check_id == "forecast.stock_lab_ledger")
    return check.state, check.reason


def test_a_removed_last_entry_fails_the_kept_head_check(tmp_path):
    store, _, _ = _setup(tmp_path, known=100, later=slice(100, 105))
    clock = store._clock
    kept = tmp_path / "kept_head"
    first = _day(store, clock, tmp_path, 1, kept)
    assert first["ledger"]["kept_head"]["state"] == "started"
    assert _ledger_check(tmp_path) == (HealthState.INSUFFICIENT_EVIDENCE, "head_started")
    assert kept.read_text().strip() == store.ledger_head() and kept.stat().st_mode & 0o077 == 0
    second = _day(store, clock, tmp_path, 2, kept)
    assert second["ledger"]["kept_head"]["state"] == "match"
    assert _ledger_check(tmp_path) == (HealthState.HEALTHY, None)
    head = kept.read_text()
    # Remove the last run and its ledger entry outside the Lab: the chain alone still verifies.
    _tamper(
        store,
        "DELETE FROM job_runs WHERE run_id = 'day-2'",
        "DELETE FROM ledger_entries WHERE seq = (SELECT MAX(seq) FROM ledger_entries)",
    )
    assert store.verify_ledger()[1] is None
    third = _day(store, clock, tmp_path, 3, kept)
    assert third["ledger"]["kept_head"] == {"state": "not_in_ledger", "kept": head.strip()}
    assert _ledger_check(tmp_path) == (HealthState.FAILED, "head_not_in_ledger")
    assert kept.read_text() == head  # a mismatch keeps the old head, so the failure stays


def test_a_deleted_store_fails_the_kept_head_check(tmp_path):
    store, _, _ = _setup(tmp_path, known=100, later=slice(100, 105))
    clock = store._clock
    kept = tmp_path / "kept_head"
    _day(store, clock, tmp_path, 1, kept)
    for companion in tmp_path.glob(DB_NAME + "*"):
        companion.unlink()
    document = _day(store, clock, tmp_path, 2, kept)
    assert document["ledger"]["kept_head"]["state"] == "ledger_deleted"
    assert _ledger_check(tmp_path) == (HealthState.FAILED, "ledger_deleted")


def test_an_empty_ledger_matches_the_genesis_head(tmp_path):
    store, clock = _store(tmp_path, T1)
    kept = tmp_path / "kept_head"
    kept.write_text(GENESIS + "\n")
    document = _day(store, clock, tmp_path, 1, kept)
    assert document["ledger"]["kept_head"]["state"] == "match"


# --- Model Health ---


@pytest.mark.parametrize(
    ("change", "check_id", "state", "reason"),
    [
        (
            lambda d: d["job"].update(
                status="skipped", reason="needs A4", activation=NOT_AUTHORIZED
            ),
            "forecast.stock_lab_job",
            "insufficient_evidence",
            NOT_AUTHORIZED,
        ),
        (
            lambda d: d["job"].update(status="skipped", reason="the vendor was down"),
            "forecast.stock_lab_job",
            "degraded",
            "job_skipped",
        ),
        (
            lambda d: (
                d["job"].update(activation=NOT_AUTHORIZED),
                d["freshness"][0].update(sessions_behind=None),
            ),
            "data.stock_lab_freshness",
            "insufficient_evidence",
            NOT_AUTHORIZED,
        ),
        (
            lambda d: d["ledger"].update(kept_head={"state": "not_in_ledger", "kept": "b" * 64}),
            "forecast.stock_lab_ledger",
            "failed",
            "head_not_in_ledger",
        ),
        (
            lambda d: d["ledger"].update(kept_head={"state": "ledger_deleted", "kept": "b" * 64}),
            "forecast.stock_lab_ledger",
            "failed",
            "ledger_deleted",
        ),
        (
            lambda d: d["ledger"].update(kept_head={"state": "unusable", "kept": None}),
            "forecast.stock_lab_ledger",
            "unavailable",
            "head_unusable",
        ),
        (
            lambda d: d["ledger"].pop("kept_head"),
            "forecast.stock_lab_ledger",
            "insufficient_evidence",
            "head_not_kept",
        ),
        (
            lambda d: d["ledger"].update(kept_head={"state": "fine"}),
            "forecast.stock_lab_ledger",
            "unavailable",
            "unrecognized_format",
        ),
        (
            lambda d: d["ledger"].update(verified=False, reason="entry 3 does not match its hash"),
            "forecast.stock_lab_ledger",
            "failed",
            "ledger_broken",
        ),
    ],
)
def test_health_reads_activation_and_the_kept_head(tmp_path, change, check_id, state, reason):
    document = health_fixtures.stock_lab_monitor()
    change(document)
    health_fixtures.write_input(tmp_path, document)
    checks = {c.check_id: c for c in stock_lab_checks(tmp_path / ARTIFACT, allow_synthetic=True)}
    assert (checks[check_id].state.value, checks[check_id].reason) == (state, reason)


def test_a_run_that_is_not_authorized_yet_but_stopped_is_still_overdue(tmp_path):
    document = health_fixtures.stock_lab_monitor()
    document["job"].update(status="skipped", reason="needs A4", activation=NOT_AUTHORIZED)
    document["schedule"]["next_expected_at"] = "2025-01-02T01:00:00.000000Z"
    health_fixtures.write_input(tmp_path, document)
    late = datetime(2025, 1, 3, 12, tzinfo=UTC)
    checks = stock_lab_checks(tmp_path / ARTIFACT, allow_synthetic=True, now=late)
    job = next(c for c in checks if c.check_id == "forecast.stock_lab_job")
    assert (job.state, job.reason) == (HealthState.FAILED, "job_overdue")


def test_each_watched_tickers_listing_status_reaches_model_health(tmp_path):
    store, clock = _store(tmp_path, T1)
    listing = (ListingRow("DEMO_CALM", "NYSE", "Stock", "USD", date(2000, 1, 3), None),)
    snapshot = store.record_listing(
        listing, source="synthetic", retrieved_at=T1, data_class=DataClass.SYNTHETIC
    )
    store.record_checks(
        [
            Classification("DEMO_CALM", SymbolStatus.SUPPORTED, SymbolReason.LISTED_US_STOCK, ""),
            Classification("OLDCO", SymbolStatus.DELISTED, SymbolReason.COVERAGE_ENDED, ""),
        ],
        snapshot_id=snapshot,
    )
    document = _day(store, clock, tmp_path, 1, None, tickers=["DEMO_CALM", "OLDCO", "NEWCO"])
    assert document["symbols"]["tickers"] == [
        {"ticker": "DEMO_CALM", "status": "supported", "reason": "listed_us_stock"},
        {"ticker": "OLDCO", "status": "delisted", "reason": "provider_coverage_ended"},
        {"ticker": "NEWCO", "status": "unverified", "reason": "not_checked"},
    ]
    checks = {
        c.check_id: c
        for c in stock_lab_checks(tmp_path / "inputs" / ARTIFACT, allow_synthetic=True)
    }
    tickers = checks["data.stock_lab_tickers"]
    assert (tickers.state, tickers.reason, tickers.value) == (
        HealthState.DEGRADED,
        "tickers_not_supported",
        1,
    )
    assert "OLDCO (delisted or renamed)" in tickers.summary and "NEWCO" in tickers.summary
    freshness = checks["data.stock_lab_freshness"]  # nothing stored yet: named as such
    assert "3 have no stored session at all: DEMO_CALM, OLDCO, NEWCO" in freshness.summary


def test_health_knows_every_listing_status_the_store_records():
    assert stock_lab_health.SYMBOL_STATUSES == {status.value for status in SymbolStatus}
    assert set(stock_lab_health.SYMBOL_WORDS) == stock_lab_health.NOT_SUPPORTED
    assert stock_lab_health.NOT_SUPPORTED < stock_lab_health.SYMBOL_STATUSES - {"supported"}


def test_the_real_job_reads_as_not_activated_rather_than_degraded(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    inputs = tmp_path / "inputs"
    code = daily.main(
        ["--health-inputs", str(inputs)],
        store_factory=lambda: store,
        clock=lambda: pd.Timestamp("2026-10-17T01:00:00Z"),
        out=io.StringIO(),
    )
    assert code == 0
    document = load_latest(inputs / ARTIFACT, ARTIFACT).document
    assert document["job"]["activation"] == NOT_AUTHORIZED
    assert document["ledger"]["kept_head"]["state"] == "started"
    assert (tmp_path / daily.KEPT_HEAD_NAME).read_text().strip() == store.ledger_head()
    checks = {c.check_id: c for c in stock_lab_checks(inputs / ARTIFACT)}
    job = checks["forecast.stock_lab_job"]
    assert (job.state, job.reason) == (HealthState.INSUFFICIENT_EVIDENCE, NOT_AUTHORIZED)
    assert all(c.state is not HealthState.DEGRADED for c in checks.values())
