"""
The forecast record (PR 3): scoring as sessions close, the hash-chained
ledger, job runs, the synthetic replay, the Model Health input and its
checks, and the daily and schedule scripts. Synthetic data only.
"""

from __future__ import annotations

import importlib.util
import io
import json
import plistlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.health import fixtures as health_fixtures
from stock_agent.health.inputs import load_latest
from stock_agent.health.model import HealthState
from stock_agent.health.stock_lab import ARTIFACT, MIN_SCORED, stock_lab_checks
from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab import monitoring, scoring, synthetic
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.replay import SimulatedClock
from stock_agent.stock_lab.scores import rps
from stock_agent.stock_lab.store import DB_NAME, StockLabStore, StoreRefused, stamp

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


daily = load_script("stock_lab_daily")
schedule = load_script("stock_lab_schedule")


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    home = tmp_path_factory.mktemp("replay")
    store = StockLabStore(home / DB_NAME)
    inputs = home / "health_inputs"
    out = io.StringIO()
    assert daily.main(["--demo-replay", "6", "--health-inputs", str(inputs)],
                      store_factory=lambda: store, out=out) == 0  # fmt: skip
    return store, inputs


# --- scoring ---


def test_every_score_uses_an_outcome_retrieved_after_its_forecast(replayed):
    store, _ = replayed
    for stock_id in synthetic.DEMO_IDS:
        forecasts = {f["forecast_id"]: f for f in store.forecasts(stock_id)}
        retrieved = {b.batch_id: b.retrieved_at for b in store.batches(stock_id)}
        scores = store.scores(stock_id)
        assert scores
        for score in scores:
            assert retrieved[score["batch_id"]] > forecasts[score["forecast_id"]]["created_at"]


def test_scores_are_the_stored_probabilities_scored_never_recomputed(replayed):
    store, _ = replayed
    score = next(
        s for s in store.scores("DEMO_TREND") if s["status"] == "scored" and s["target"] == "1"
    )
    record = next(
        f for f in store.forecasts("DEMO_TREND") if f["forecast_id"] == score["forecast_id"]
    )
    row = record["daily"][0]
    p = np.array([[row["p_decrease"], row["p_stable"], row["p_increase"]]])
    assert score["rps"] == pytest.approx(rps(p, np.array([score["outcome"]]))[0])
    bars = store.bars("DEMO_TREND").bars
    session = bars[bars["session"] == pd.Timestamp(score["session"], tz="UTC")].index[0]
    expected = (
        bars["close"].iloc[session]
        * bars["split_factor"].iloc[session]
        / bars["close"].iloc[session - 1]
        - 1
    )
    assert score["realized_return"] == pytest.approx(expected)


def test_a_session_not_yet_closed_is_pending_and_nothing_is_written_for_it(replayed):
    store, _ = replayed
    latest = store.latest_forecast("DEMO_CALM")
    assert not [s for s in store.scores("DEMO_CALM") if s["forecast_id"] == latest["forecast_id"]]
    bars = store.bars("DEMO_CALM")
    batch_of = {row.session.strftime("%Y-%m-%d"): row.batch_id for row in bars.bars.itertuples()}
    pending = scoring.score_forecast(
        latest, bars.bars.drop(columns=["batch_id"]), batch_of, seen_before=set()
    )
    assert {s.status for s in pending} == {scoring.PENDING}


# A small store on a simulated clock: DEMO_CALM's first `known` sessions retrieved just after
# session 99's close (T0), a forecast made five minutes later as of session 99 (its targets are
# sessions 100..104), then later sessions downloaded after session 104's close (T1), optionally
# re-downloading sessions already known.
AS_OF = 99
SESSIONS = synthetic.demo_bars("DEMO_CALM")["session"]
T0 = SESSIONS.iloc[AS_OF] + pd.Timedelta(hours=21)
MADE = T0 + pd.Timedelta(minutes=5)
T1 = SESSIONS.iloc[AS_OF + 5] + pd.Timedelta(hours=21)
PRICES = {"source": "f", "data_class": DataClass.SYNTHETIC, "vintage_note": "v"}


def _forecast(created_at, *, as_of=AS_OF, cumulative=(0.2, 0.6, 0.2), supersedes=None):
    return artifact.build(
        symbol="DEMO_CALM",
        data_class="synthetic",
        as_of_session=SESSIONS.iloc[as_of].strftime("%Y-%m-%d"),
        created_at=stamp(created_at),
        target_sessions=[d.strftime("%Y-%m-%d") for d in SESSIONS.iloc[as_of + 1 : as_of + 6]],
        daily=[[0.2, 0.6, 0.2]] * 5,
        cumulative=list(cumulative),
        band=0.005,
        cumulative_band=0.005 * 5**0.5,
        estimator="B1",
        model_version="test",
        feature_version="stock_lab_features_v1",
        evaluation_run_id=None,
        data={},
        evidence_status="synthetic_demo",
        calibration_status="test",
        signals={},
        limitations=[],
        supersedes=supersedes,
    )


def _setup(tmp_path, *, known: int, later: slice, cumulative=(0.2, 0.6, 0.2)):
    clock = SimulatedClock(T0)
    store = StockLabStore(tmp_path / DB_NAME, clock=clock)
    bars = synthetic.demo_bars("DEMO_CALM")
    store.record_prices("DEMO_CALM", bars.iloc[:known], retrieved_at=T0, **PRICES)
    clock.advance(MADE)
    record = _forecast(MADE, cumulative=cumulative)
    store.record_forecast(record)
    clock.advance(T1)
    store.record_prices("DEMO_CALM", bars.iloc[later], retrieved_at=T1, **PRICES)
    return store, bars, record


def _by_target(store):
    return {s["target"]: s for s in store.scores("DEMO_CALM")}


def test_a_later_re_download_never_makes_an_already_seen_outcome_scorable(tmp_path):
    # Sessions 100-102 (targets 1-3) were stored BEFORE the forecast; a later window
    # re-downloads 95..104. Only targets 4 and 5, and nothing spanning 1-3, may be scored.
    store, _, _ = _setup(tmp_path, known=103, later=slice(95, 105))
    scoring.score_store(store, ["DEMO_CALM"])
    scores = _by_target(store)
    for target in ("1", "2", "3"):
        assert scores[target]["reason"] == "outcome_retrieved_before_forecast", target
    assert scores["4"]["status"] == scores["5"]["status"] == "scored"
    assert scores["cum"]["reason"] == "outcome_retrieved_before_forecast"


def test_the_five_session_total_uses_its_own_band_and_base(tmp_path):
    store, bars, record = _setup(tmp_path, known=100, later=slice(100, 105))
    scoring.score_store(store, ["DEMO_CALM"])
    total = _by_target(store)["cum"]
    level = bars["close"] * bars["split_factor"].cumprod()
    expected = level.iloc[104] / level.iloc[99] - 1
    assert total["realized_return"] == pytest.approx(expected)
    tau = record["cumulative_band"]
    code = 2 if expected > tau else 0 if expected < -tau else 1
    assert total["outcome"] == code


def test_a_gap_found_by_the_calendar_is_never_scored_as_one_session(tmp_path):
    store, _, _ = _setup(tmp_path, known=100, later=slice(101, 105))  # session 100 missing
    calendar = lambda a, b: pd.bdate_range(a, b, tz="UTC").as_unit("us")
    scoring.score_store(store, ["DEMO_CALM"], calendar=calendar)
    scores = _by_target(store)
    assert (
        scores["2"]["status"] == "unscorable" and scores["2"]["reason"] == "previous_close_missing"
    )
    assert scores["cum"]["reason"] == "window_incomplete"
    assert scores["3"]["status"] == "scored"


def test_scores_are_write_once_and_scoring_again_adds_nothing(replayed):
    store, _ = replayed
    assert scoring.score_store(store, synthetic.DEMO_IDS) == 0
    first = store.scores("DEMO_CALM")[0]
    with pytest.raises(sqlite3.IntegrityError):
        store.record_scores([first])


# --- the ledger ---


def test_the_ledger_covers_every_record_and_verifies(replayed):
    store, _ = replayed
    entries, problem = store.verify_ledger()
    forecasts = sum(len(store.forecasts(s)) for s in synthetic.DEMO_IDS)
    assert problem is None
    assert entries == forecasts + len(store.scores()) + len(store.job_runs())


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("UPDATE forecasts SET record = replace(record, '\"B1\"', '\"B2\"')", "record"),
        ("UPDATE ledger_entries SET payload_sha256 = '0' WHERE seq = 2", "hash"),
        ("DELETE FROM ledger_entries WHERE seq = 3", "sequence"),
        ("UPDATE scores SET outcome = 2, rps = 0.0 WHERE rowid = 1", "score record"),
        ("DELETE FROM scores WHERE rowid = 1", "missing score"),
        ("UPDATE job_runs SET status = 'ok', record = replace(record, 'ok', 'failed')", "record"),
        (
            (
                "INSERT INTO forecasts SELECT forecast_id || 'x', symbol, as_of_session, "
                "created_at, data_class, supersedes, record FROM forecasts LIMIT 1"
            ),
            "no ledger entry",
        ),
    ],
)
def test_editing_the_store_outside_the_lab_breaks_the_chain(replayed, tmp_path, tamper, message):
    store, _ = replayed
    copy_path = tmp_path / DB_NAME
    copy_path.write_bytes(store.path.read_bytes())
    tmp_path.chmod(0o700)
    connection = sqlite3.connect(copy_path)
    for trigger in [
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    ]:
        connection.execute(f"DROP TRIGGER {trigger}")
    connection.execute(tamper)
    connection.commit()
    connection.close()
    _, problem = StockLabStore(copy_path).verify_ledger()
    assert problem is not None and message in problem


# --- the replay ---


def test_the_replay_fits_only_on_sessions_before_it_starts(replayed):
    store, _ = replayed
    evaluation = store.latest_evaluation(DataClass.SYNTHETIC)
    start = store.forecasts("DEMO_CALM")[0]["as_of_session"]
    assert evaluation["config"]["outcome_end"] == start
    # Half-open windows end on the day after the start; outcomes after it are censored.
    day_after = (pd.Timestamp(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert all(fold["test_end"] <= day_after for fold in evaluation["folds"])
    for stock_id in synthetic.DEMO_IDS:
        history = store.forecasts(stock_id)
        assert len(history) == 7  # the start and six replay days
        # One forecast per as-of session: a new day's forecast replaces nothing.
        assert [f["supersedes"] for f in history] == [None] * 7
        assert len({f["as_of_session"] for f in history}) == 7
        # The store recorded each one on the simulated clock, when it was made: none is backdated.
        recorded = store.forecast_recorded_at(stock_id)
        assert all(recorded[f["forecast_id"]] == f["created_at"] for f in history)
        assert history[-1]["as_of_session"] == "2024-12-31"
        assert all(f["data_class"] == "synthetic" for f in history)


# --- monitoring and Model Health ---


def test_psi_is_near_zero_for_one_distribution_and_large_for_a_shift():
    rng = np.random.RandomState(0)
    development = rng.normal(size=5000)
    assert monitoring.psi(development, rng.normal(size=500)) < 0.05
    assert monitoring.psi(development, rng.normal(1.0, 1.0, size=500)) > 0.25
    assert monitoring.psi(development[:50], development[:30]) is None


def test_each_replay_day_writes_one_valid_health_input(replayed):
    _, inputs = replayed
    files = sorted((inputs / ARTIFACT).glob("*.json"))
    assert len(files) == 6
    latest = load_latest(inputs / ARTIFACT, ARTIFACT)
    assert latest.synthetic and latest.document["ledger"]["verified"]
    assert latest.document["lineage"]["feature_version"] == "stock_lab_features_v1"
    with pytest.raises(FileExistsError):
        monitoring.write_monitor(inputs, json.loads(files[-1].read_text()))
    checks = {c.check_id: c for c in stock_lab_checks(inputs / ARTIFACT, allow_synthetic=True)}
    assert checks["forecast.stock_lab_ledger"].state is HealthState.HEALTHY
    assert checks["forecast.stock_lab_scores"].reason == "too_few_scored"
    private = stock_lab_checks(inputs / ARTIFACT)  # the private scope: synthetic is never healthy
    assert all(c.state is not HealthState.HEALTHY for c in private)


@pytest.mark.parametrize(
    ("change", "check_id", "state", "reason"),
    [
        (
            lambda d: d["job"].update(status="partial", failed=["X"]),
            "forecast.stock_lab_job",
            "degraded",
            "job_partial",
        ),
        (
            lambda d: d["job"].update(status="skipped", reason="approvals"),
            "forecast.stock_lab_job",
            "degraded",
            "job_skipped",
        ),
        (
            lambda d: d["scores"].update(rps=0.3),
            "forecast.stock_lab_scores",
            "degraded",
            "worse_than_baseline",
        ),
        (
            lambda d: d["scores"].update(n=MIN_SCORED - 1),
            "forecast.stock_lab_scores",
            "insufficient_evidence",
            "too_few_scored",
        ),
        (
            lambda d: d.update(freshness=[]),
            "data.stock_lab_freshness",
            "insufficient_evidence",
            "no_watchlist",
        ),
        (
            lambda d: d["freshness"][0].update(sessions_behind=None),
            "data.stock_lab_freshness",
            "degraded",
            "stale_prices",
        ),
        (
            lambda d: d.update(inputs_drift=[]),
            "forecast.stock_lab_drift",
            "insufficient_evidence",
            "no_drift_measured",
        ),
        (
            lambda d: d.pop("ledger"),
            "forecast.stock_lab_ledger",
            "unavailable",
            "unrecognized_format",
        ),
        (
            lambda d: d.pop("symbols"),
            "data.stock_lab_tickers",
            "insufficient_evidence",
            "not_recorded",
        ),
        (
            lambda d: d["symbols"].update(tickers=[]),
            "data.stock_lab_tickers",
            "insufficient_evidence",
            "no_watchlist",
        ),
        (
            lambda d: d["symbols"]["tickers"][0].update(status="unverified", reason="not_checked"),
            "data.stock_lab_tickers",
            "insufficient_evidence",
            "tickers_not_checked",
        ),
        (
            lambda d: d["symbols"]["tickers"][1].update(status="ambiguous"),
            "data.stock_lab_tickers",
            "degraded",
            "tickers_not_supported",
        ),
        (
            lambda d: d["symbols"]["tickers"][1].update(status="inactive"),
            "data.stock_lab_tickers",
            "degraded",
            "tickers_not_supported",
        ),
        (
            lambda d: d["symbols"]["tickers"][0].update(status="probably_fine"),
            "data.stock_lab_tickers",
            "unavailable",
            "unrecognized_format",
        ),
    ],
)
def test_the_health_checks_judge_what_the_job_recorded(tmp_path, change, check_id, state, reason):
    document = health_fixtures.stock_lab_monitor()
    change(document)
    health_fixtures.write_input(tmp_path, document)
    checks = {c.check_id: c for c in stock_lab_checks(tmp_path / ARTIFACT, allow_synthetic=True)}
    assert (checks[check_id].state.value, checks[check_id].reason) == (state, reason)


# --- the daily and schedule scripts ---


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-10-16T23:59:00Z", "2026-10-15"),  # 19:59 in New York: today's close not yet
        ("2026-10-17T00:01:00Z", "2026-10-16"),  # 20:01 in New York
        ("2026-10-18T15:00:00Z", "2026-10-16"),  # a Sunday
        ("2026-11-27T03:00:00Z", "2026-11-25"),  # Thanksgiving evening: the holiday is skipped
    ],
)
def test_the_expected_session_follows_the_nyse_calendar(now, expected):
    assert daily.expected_session(pd.Timestamp(now)).strftime("%Y-%m-%d") == expected


def test_the_real_job_scores_records_itself_skipped_and_never_forecasts(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    out = io.StringIO()
    code = daily.main(["--health-inputs", str(tmp_path / "inputs")], store_factory=lambda: store,
                      clock=lambda: pd.Timestamp("2026-10-17T01:00:00Z"), out=out)  # fmt: skip
    assert code == 0, out.getvalue()
    run = store.latest_job_run()
    assert run["status"] == "skipped" and "A4" in run["reason"] and run["data_class"] == "real"
    assert run["expected_session"] == "2026-10-16"
    document = load_latest(tmp_path / "inputs" / ARTIFACT, ARTIFACT).document
    assert document["synthetic"] is False and document["schedule"]["next_expected_at"]
    assert store.verify_ledger() == (1, None)


@pytest.mark.parametrize("flag", ["--authorized", "--inference-window", "--record-real"])
def test_the_real_update_flags_are_refused_until_the_owners_decisions(tmp_path, flag):
    store = StockLabStore(tmp_path / DB_NAME)
    out = io.StringIO()
    assert (
        daily.main([flag, "--health-inputs", str(tmp_path)], store_factory=lambda: store, out=out)
        == 2
    )
    assert "A1, A2 and A4" in out.getvalue() and store.latest_job_run() is None


@pytest.mark.parametrize("sessions", ["0", "61"])
def test_the_replay_length_is_bounded(tmp_path, sessions):
    out = io.StringIO()
    store = StockLabStore(tmp_path / DB_NAME)
    assert daily.main(["--demo-replay", sessions], store_factory=lambda: store, out=out) == 2


def test_the_schedule_is_printed_never_installed(tmp_path):
    out = io.StringIO()
    assert schedule.main([], out=out) == 0
    plan = plistlib.loads(out.getvalue().encode())
    assert plan["Label"] == schedule.LABEL
    assert plan["ProgramArguments"][1].endswith("scripts/stock_lab_daily.py")
    times = plan["StartCalendarInterval"]
    assert [t["Weekday"] for t in times] == [1, 2, 3, 4, 5]
    assert {(t["Hour"], t["Minute"]) for t in times} == {(20, 30)}
    assert schedule.local_time("America/Los_Angeles") == (17, 30, 0)
    tokyo = plistlib.loads(schedule.plist("Asia/Tokyo").encode())["StartCalendarInterval"]
    assert [t["Weekday"] for t in tokyo] == [2, 3, 4, 5, 6]  # Tuesday-Saturday morning there
    assert {(t["Hour"], t["Minute"]) for t in tokyo} == {(10, 30)}
    assert "launchctl" not in Path(schedule.__file__).read_text().split('"""', 2)[2]  # never run


def test_the_freshness_counts_sessions_behind_the_expected_one(tmp_path):
    from stock_agent.stock_lab.job import freshness

    store = StockLabStore(tmp_path / DB_NAME)
    bars = synthetic.demo_bars("DEMO_CALM").iloc[:50]
    store.record_prices("DEMO_CALM", bars, source="f", retrieved_at=T0,
                        data_class=DataClass.SYNTHETIC, vintage_note="v")  # fmt: skip
    expected = bars["session"].iloc[-1] + pd.offsets.BDay(2)
    calendar = lambda a, b: pd.bdate_range(a, b, tz="UTC")
    rows = freshness(store, ["DEMO_CALM", "DEMO_SWING"], expected, calendar)
    assert rows[0]["sessions_behind"] == 2 and rows[1]["sessions_behind"] is None


def test_a_real_job_never_reports_a_synthetic_success(replayed, tmp_path):
    store, _ = replayed
    inputs = tmp_path / "inputs"
    code = daily.main(["--health-inputs", str(inputs)], store_factory=lambda: store,
                      clock=lambda: pd.Timestamp("2026-10-17T01:00:00Z"), out=io.StringIO())  # fmt: skip
    assert code == 0
    document = load_latest(inputs / ARTIFACT, ARTIFACT).document
    assert document["schedule"]["last_success_at"] is None  # the replay's runs are synthetic
    assert document["ledger"]["head"] == store.ledger_head()


def test_a_second_replay_into_the_same_store_is_refused_before_writing(replayed):
    store, _ = replayed
    before = store.verify_ledger()
    out = io.StringIO()
    assert daily.main(["--demo-replay", "3"], store_factory=lambda: store, out=out) == 2
    assert "STOCK_AGENT_STOCK_LAB_HOME" in out.getvalue() and store.verify_ledger() == before


def test_a_job_that_stopped_is_failed_as_overdue(tmp_path):
    from datetime import UTC, datetime

    document = health_fixtures.stock_lab_monitor()
    document["schedule"]["next_expected_at"] = "2025-01-02T01:00:00.000000Z"
    health_fixtures.write_input(tmp_path, document)
    late = datetime(2025, 1, 3, 12, tzinfo=UTC)
    checks = {
        c.check_id: c for c in stock_lab_checks(tmp_path / ARTIFACT, allow_synthetic=True, now=late)
    }
    assert (checks["forecast.stock_lab_job"].state, checks["forecast.stock_lab_job"].reason) == (
        HealthState.FAILED,
        "job_overdue",
    )
    on_time = datetime(2025, 1, 2, 3, tzinfo=UTC)
    checks = {
        c.check_id: c
        for c in stock_lab_checks(tmp_path / ARTIFACT, allow_synthetic=True, now=on_time)
    }
    assert checks["forecast.stock_lab_job"].state is HealthState.HEALTHY


def test_overlapping_targets_count_once_per_session_for_the_rolling_score(tmp_path):
    document = health_fixtures.stock_lab_monitor()
    document["scores"].update(n=300, sessions=20)  # 300 targets but only 20 sessions settled
    health_fixtures.write_input(tmp_path, document)
    checks = {c.check_id: c for c in stock_lab_checks(tmp_path / ARTIFACT, allow_synthetic=True)}
    assert checks["forecast.stock_lab_scores"].reason == "too_few_scored"


def test_a_forecast_dated_in_the_future_is_refused(tmp_path):
    store, _, record = _setup(tmp_path, known=100, later=slice(100, 101))
    future = {**record, "created_at": "2099-01-01T00:00:00.000000Z"}
    future["forecast_id"] = artifact.forecast_id(future)
    with pytest.raises(StoreRefused, match="future"):
        store.record_forecast(future)


@pytest.mark.parametrize(
    ("now", "due"),
    [
        ("2026-10-16T20:00:00Z", "2026-10-17T00:00:00.000000Z"),  # Friday 16:00 ET: tonight
        ("2026-10-17T01:00:00Z", "2026-10-20T00:00:00.000000Z"),  # Friday 21:00 ET: Monday
    ],
)
def test_the_next_expected_update_is_tonight_before_the_close_is_due(now, due):
    assert daily.next_expected(pd.Timestamp(now)) == due


def test_a_calendar_refusal_is_recorded_as_a_failed_run(tmp_path, monkeypatch):
    from stock_agent.calendars.nyse import CalendarRefused

    def refuse(now):
        raise CalendarRefused("outside_coverage", "past 2027")

    monkeypatch.setattr(daily, "expected_session", refuse)
    store = StockLabStore(tmp_path / DB_NAME)
    assert (
        daily.main(
            ["--health-inputs", str(tmp_path)], store_factory=lambda: store, out=io.StringIO()
        )
        == 1
    )
    run = store.latest_job_run()
    assert run["status"] == "failed" and "calendar" in run["reason"]


def test_a_monitor_file_without_symbols_is_still_a_valid_v1_0_file(tmp_path):
    # `symbols` is additive: a file written before it keeps every other check's verdict.
    document = health_fixtures.stock_lab_monitor()
    with_symbols = {
        c.check_id: (c.state, c.reason)
        for c in stock_lab_checks(
            health_fixtures.write_input(tmp_path / "with", document).parent, allow_synthetic=True
        )
    }
    document.pop("symbols")
    assert document["schema_version"] == "1.0"
    without = {
        c.check_id: (c.state, c.reason)
        for c in stock_lab_checks(
            health_fixtures.write_input(tmp_path / "without", document).parent,
            allow_synthetic=True,
        )
    }
    tickers = "data.stock_lab_tickers"
    assert without.pop(tickers) == (HealthState.INSUFFICIENT_EVIDENCE, "not_recorded")
    assert with_symbols.pop(tickers) == (HealthState.HEALTHY, None)
    assert without == with_symbols
